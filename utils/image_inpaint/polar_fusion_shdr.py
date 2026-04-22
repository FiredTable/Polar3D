import os
import cv2
import skimage
import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from pathlib import Path
from datetime import datetime
from sklearn.cluster import KMeans, MiniBatchKMeans
from typing import Optional
from scipy.ndimage import gaussian_filter

from utils.polarization_analyser import PolarizationAnalyser
from utils.image_inpaint.utils.canopy_cluster import Canopy
from utils.common import imsave, list2bhwc, imnorm
from utils.image import apply_clahe, inpaint_by_diffusion
from utils.image_fusion.DWTFusion import DWTFuser
from utils.image_fusion.FFTFusion import FFTFuser
from utils.image_fusion.poisson_fusion import poisson_fusion

__all__ = [
    'PolarFusionSHDR'
]


def _merge_mask(base_spec_mask, pseudo_spec_mask, min_size=None, visualize=False):
    """
    如果 base_spec_mask 和 pseudo_spec_mask 中的连通域相邻近，则使 mask 包含该连通域
    :param base_spec_mask: 基础 mask
    :param pseudo_spec_mask: 伪扩展 mask
    """
    # 对 base_spec_mask 和 pseudo_spec_mask 进行小连通域的消除
    if min_size is None:
        min_size = base_spec_mask.shape[0] * base_spec_mask.shape[1] / 2e4

    denoise_base_mask = skimage.morphology.remove_small_objects(base_spec_mask, min_size)
    denoise_pseudo_mask = skimage.morphology.remove_small_objects(pseudo_spec_mask, min_size)

    ext_spec_mask = denoise_base_mask

    extent_regions = skimage.measure.label(denoise_pseudo_mask)
    num_extent_regions = len(np.unique(extent_regions))
    for k in range(1, num_extent_regions):
        kth_extent_region = np.zeros_like(denoise_pseudo_mask, dtype=bool)
        kth_extent_region[extent_regions == k] = True

        # 判断 kth_extent_region 是否与 denoise_base_mask 相交
        # 如果相交，则 kth_extent_region 必定覆盖 denoise_base_mask 中的某个连通域
        overlapping_region = kth_extent_region & denoise_base_mask

        if not np.any(overlapping_region):
            continue

        # kth_extent_region 的面积应该小于 overlapping_region 面积的 10 倍
        area_overlapping_region = np.sum(overlapping_region.astype(np.uint8))
        area_kth_extent_region = np.sum(kth_extent_region.astype(np.uint8))
        if area_kth_extent_region > 10 * area_overlapping_region:
            continue

        ext_spec_mask[kth_extent_region] = True

    if visualize:
        plt.figure('_merge_mask')
        plt.subplot(131)
        plt.imshow(base_spec_mask)
        plt.title('base mask')
        plt.subplot(132)
        plt.imshow(pseudo_spec_mask)
        plt.title('pseudo_spec_mask')
        plt.subplot(133)
        plt.imshow(ext_spec_mask)
        plt.title('extent mask')
        plt.show(block=True)

    return ext_spec_mask


class PolarFusionSHDR:
    def __init__(self, polar_analyser: PolarizationAnalyser, save_dir: Path):
        self.polar_analyser = polar_analyser
        self.canopy_centers = None
        self.kmeans_centers = None
        self.label_image = None
        self.rho_centers = None
        self.iun_centers = None
        self.cluster_thresh1 = None
        self.cluster_thresh2 = None
        self.high_spec_mask = None
        self.spec2diffuse_mask = None
        self.ext_spec_mask = None
        self.mask_on_img = None
        self.runtime_cluster = None
        self.diffuse_image = None
        self.spec_image = None
        self.clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))

        if save_dir is None:
            time_str = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")
            self.save_dir = Path(os.path.join('runs', time_str))
        else:
            self.save_dir = save_dir

    @staticmethod
    def _canopy_cluster(dataset, canopy_t1, canopy_t2):
        """使用canopy聚类方法对数据进行聚类分析"""
        canopy = Canopy(dataset)
        canopy.setThreshold(canopy_t1, canopy_t2)
        canopy.canopies = canopy.clustering()

        canopy_centers = np.array([sublist[0] for sublist in canopy.canopies])
        return canopy_centers

    @staticmethod
    def _kmeans_cluster(dataset, init_centers, mini_batch=True):
        """使用kmeans聚类方法对数据进行聚类分析"""
        if mini_batch:
            kmeans_model = MiniBatchKMeans(n_clusters=init_centers.shape[0], init=init_centers)
        else:
            kmeans_model = KMeans(n_clusters=init_centers.shape[0], init=init_centers)

        kmeans_model.fit(dataset)

        return kmeans_model.cluster_centers_, kmeans_model.labels_

    def polar_clustering(self, canny_t1=0.6, canny_t2=0.4, mini_batch=False, save=False):
        """对偏振特征进行聚类，得到强度聚类中心和偏振度聚类中心"""
        st_time = time.time()

        # 对 iun 和 rho 进行去噪
        iun = gaussian_filter(self.polar_analyser.iun, sigma=1.0)
        rho = gaussian_filter(self.polar_analyser.rho, sigma=1.0)

        # 计算偏振特征聚类中心点
        pixel_num = iun.shape[0] * iun.shape[1]
        polar_dataset = np.concatenate((iun.reshape((pixel_num, 1)), rho.reshape((pixel_num, 1))), axis=1)
        self.canopy_centers = self._canopy_cluster(polar_dataset, canny_t1, canny_t2)
        self.kmeans_centers, labels = self._kmeans_cluster(polar_dataset, self.canopy_centers, mini_batch=mini_batch)

        self.label_image = labels.reshape(iun.shape[0], iun.shape[1])

        self.runtime_cluster = time.time() - st_time
        print(f'polar clustering costs {self.runtime_cluster:.4f} s')

    def stokes_clustering(self, canopy_t1=0.6, canopy_t2=0.4, mini_batch=True, save=False):
        """对斯托克斯参量进行聚类，得到强度聚类中心和偏振度聚类中心"""
        st_time = time.time()

        # 对斯托克斯参量进行最大值归一化，避免出现过多的聚类中心点
        stokes = gaussian_filter(self.polar_analyser.stokes / np.max(self.polar_analyser.stokes), sigma=1.0)

        # 计算斯托克斯聚类中心点
        stokes_dataset = stokes.reshape((-1, stokes.shape[2]))
        self.canopy_centers = self._canopy_cluster(stokes_dataset, canopy_t1, canopy_t2)
        self.kmeans_centers, labels = self._kmeans_cluster(stokes_dataset, self.canopy_centers, mini_batch=mini_batch)

        # 计算对应的偏振特征中心
        self.label_image = labels.reshape(stokes.shape[0], stokes.shape[1])

        self.runtime_cluster = time.time() - st_time
        print(f'stokes clustering costs {self.runtime_cluster:.4f} s')

        if save:
            imsave(self.label_image, 'label_image.png', self.save_dir, cmap_name='coolwarm', norm=True)

    def estimate_cluster_thresh(self, r1=1.0, r2=0.7):
        uint8_iun = (255 * self.polar_analyser.iun).astype(np.uint8)
        uint8_rho = (255 * self.polar_analyser.rho).astype(np.uint8)
        thresh_iun, _ = cv2.threshold(uint8_iun, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh_rho, _ = cv2.threshold(uint8_rho, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        cluster_thresh1 = r1 * 0.5 * (thresh_iun + thresh_rho) / 255.
        cluster_thresh2 = r2 * cluster_thresh1

        return cluster_thresh1, cluster_thresh2

    def label2centers(self):
        rho_centers = np.array([np.mean(self.polar_analyser.rho[self.label_image == k])
                                for k in range(self.kmeans_centers.shape[0])])
        iun_centers = np.array([np.mean(self.polar_analyser.iun[self.label_image == k])
                                for k in range(self.kmeans_centers.shape[0])])
        return rho_centers, iun_centers

    @staticmethod
    def detect_outliers(img, max_area: Optional[int] = None, block_size: int = 129,
                        c: int = -60, threshold_type=cv2.THRESH_BINARY):
        # 使用自适应阈值处理获取高光掩码
        uint8_img = (255 * img).astype(np.uint8)

        if img.ndim == 2:
            detect_mask = cv2.adaptiveThreshold(
                uint8_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, threshold_type, block_size, c)
        elif img.ndim == 3:
            detect_mask = []
            for k in range(img.shape[-1]):
                detect_mask.append(
                    cv2.adaptiveThreshold(uint8_img[:, :, k], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          threshold_type, block_size, c))
            detect_mask = np.stack(detect_mask, axis=-1)
            detect_mask = np.max(detect_mask, axis=-1)
        else:
            raise ValueError

        detect_mask = detect_mask.astype(bool)

        if max_area is not None:
            # 对感兴趣区域的最大面积进行限制
            tmp_mask = skimage.morphology.remove_small_objects(detect_mask, max_area)
            detect_mask = np.logical_xor(detect_mask, tmp_mask)

        return detect_mask

    def get_specular_region(self, thresh, c):
        """计算镜面反射区域"""
        thresh_iun = np.percentile(self.polar_analyser.iun, 10)
        thresh_rho = np.percentile(self.polar_analyser.rho, 10)

        specular_mask = np.zeros_like(self.label_image, dtype=bool)

        spec_assess = []
        for k in range(self.rho_centers.shape[0]):
            spec_assess.append(0.55 * self.iun_centers[k] / np.max(self.iun_centers) +
                               0.45 * self.rho_centers[k] / np.max(self.rho_centers))
            if (spec_assess[k] > thresh) and (self.iun_centers[k] > thresh_iun) and (self.rho_centers[k] > thresh_rho):
                specular_mask[self.label_image == k] = True

        # 对感兴趣区域的面积进行限制
        min_area = int(np.sqrt(self.label_image.size) / 100.) ** 2
        max_area = int(np.sqrt(self.label_image.size) / 3.) ** 2
        outliers_max_area = int(np.sqrt(self.label_image.size) / 30.) ** 2
        block_size = int(np.sqrt(self.label_image.size) / 10.) * 2 + 1
        dilate_radius = int(np.sqrt(self.label_image.size) / 200.)

        outliers_mask = self.detect_outliers(self.polar_analyser.iun, max_area=outliers_max_area,
                                             block_size=block_size, c=c)
        specular_mask[outliers_mask] = True

        se = skimage.morphology.disk(radius=dilate_radius)
        specular_region = skimage.morphology.binary_dilation(specular_mask, se)

        filtered_specular_region = skimage.morphology.remove_small_objects(specular_region, min_area)

        return filtered_specular_region, outliers_mask

    def detect_specular_highlight(self, cluster_src='polar',
                                  canopy_t1=0.5, canopy_t2=0.2, mini_batch=False,
                                  r1=1.2, r2=0.8,
                                  visualize=False, save=False):
        st_time = time.time()

        if cluster_src == 'stokes':
            self.stokes_clustering(canopy_t1=canopy_t1, canopy_t2=canopy_t2, mini_batch=mini_batch, save=save)
        elif cluster_src == 'polar':
            self.polar_clustering(canny_t1=canopy_t1, canny_t2=canopy_t2, mini_batch=mini_batch, save=save)

        self.cluster_thresh1, self.cluster_thresh2 = self.estimate_cluster_thresh(r1=r1, r2=r2)
        self.rho_centers, self.iun_centers = self.label2centers()

        # 计算镜面反射区域
        self.high_spec_mask, outliers_mask1 = self.get_specular_region(thresh=self.cluster_thresh1, c=-80)
        self.spec2diffuse_mask, outliers_mask2 = self.get_specular_region(thresh=self.cluster_thresh2, c=-60)
        self.ext_spec_mask = _merge_mask(self.high_spec_mask, self.spec2diffuse_mask, visualize=visualize)

        runtime = time.time() - st_time
        print(f'Detect specular highlight runtime: {runtime:.4f} s')

        save_path = os.path.join(self.save_dir, 'detect')
        if save:
            self.save_label_results(save_path, with_text=False)
            self.save_label_results(save_path, with_text=True)
            imsave(self.high_spec_mask, 'high_spec_mask.png', save_path)
            imsave(self.spec2diffuse_mask, 'spec2diffuse_mask.png', save_path)
            imsave(self.ext_spec_mask, 'ext_spec_mask.png', save_path)
            mask_on_img = self.overlay2mask(self.polar_analyser.avg_images)
            imsave(mask_on_img, 'mask on img.png', save_path)
            imsave(outliers_mask1, 'outliers_mask1.png', save_path)
            imsave(outliers_mask2, 'outliers_mask2.png', save_path)
            # self.save_cluster_centers(save_path)

    def overlay2mask(self, img, alpha=0.1):
        # img 需要是三通道图像
        if img.ndim == 2:
            img = np.stack((img,) * 3, axis=-1)

        mask_on_img = img.copy()
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if self.ext_spec_mask[i, j]:
                    mask_on_img[i, j, 0] = alpha * img[i, j, 0] + (1 - alpha) * 243.0 / 255.0
                    mask_on_img[i, j, 1] = alpha * img[i, j, 1] + (1 - alpha) * 210.0 / 255.0
                    mask_on_img[i, j, 2] = alpha * img[i, j, 2] + (1 - alpha) * 102.0 / 255.0
                if self.high_spec_mask[i, j]:
                    mask_on_img[i, j, 0] = alpha * img[i, j, 0] + (1 - alpha) * 150.0 / 255.0
                    mask_on_img[i, j, 1] = alpha * img[i, j, 1] + (1 - alpha) * 195.0 / 255.0
                    mask_on_img[i, j, 2] = alpha * img[i, j, 2] + (1 - alpha) * 125.0 / 255.0

        return mask_on_img

    def save_label_results(self, save_path, with_text=False):
        """显示聚类结果"""
        # 离散的 colorbar：
        #   https://blog.csdn.net/dreaming_coder/article/details/106833727
        plt.figure('show_cluster_results', figsize=(12, 8))
        plt.clf()  # 关键修复：清除已有绘图元素

        lut = np.max(self.label_image) + 1
        cmap = plt.get_cmap('coolwarm', lut=lut)
        bounds = np.linspace(-0.5, np.max(self.label_image) + 0.5, lut + 1)
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        if with_text:
            text_blocks = []
            if hasattr(self, 'rho_centers') and hasattr(self, 'iun_centers'):
                rho_str = "\n".join([
                    f"Rho Center {i}: {c.round(3)}"
                    for i, c in enumerate(self.rho_centers)
                ])
                iun_str = "\n".join([
                    f"Iun Center {i}: {c.round(3)}"
                    for i, c in enumerate(self.iun_centers)
                ])
                text_blocks.append(("Rho Centers", rho_str))
                text_blocks.append(("Intensity Centers", iun_str))

            if hasattr(self, 'cluster_thresh1') and hasattr(self, 'cluster_thresh2'):
                thresh_str = (
                    f"Threshold1: {self.cluster_thresh1:.3f}\n"
                    f"Threshold2: {self.cluster_thresh2:.3f}"
                )
                text_blocks.append(("Decision Thresholds", thresh_str))

            # 组织文本布局
            text_y = 0.85  # 起始Y坐标
            text_x = 0.02  # 左侧边距

            for title, content in text_blocks:
                full_text = f"{title}:\n{content}"
                plt.figtext(
                    x=text_x,
                    y=text_y,
                    s=full_text,
                    fontsize=8,
                    family='monospace',
                    bbox=dict(
                        facecolor='white',
                        alpha=0.8,
                        edgecolor='red' if 'Threshold' in title else 'gray'
                    )
                )
                text_y -= 0.18  # 垂直间距调整

        plt.imshow(self.label_image, cmap=cmap, norm=norm)
        plt.colorbar(ticks=np.arange(0, lut))
        plt.axis('off')

        if with_text:
            save_name = 'label_image_with_ticks_with_text.png'
        else:
            save_name = 'label_image_with_ticks.png'

        os.makedirs(save_path, exist_ok=True)

        plt.savefig(
            os.path.join(save_path, save_name),
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.3,
            transparent=False
        )
        plt.close()

    def save_cluster_centers(self, save_path):
        """显示聚类中心"""
        plt.figure('save_cluster_centers')

        iun = self.polar_analyser.iun
        rho = self.polar_analyser.rho

        x1 = np.arange(0, np.max(iun), 0.005)
        x2 = np.arange(0, np.max(rho), 0.005)
        x1G, x2G = np.meshgrid(x1, x2)
        XGrid = np.vstack((x1G.ravel(), x2G.ravel())).T

        iun_centers = PolarizationAnalyser.stokes2iun(self.kmeans_centers)
        rho_centers = PolarizationAnalyser.stokes2rho(self.kmeans_centers)

        kmeans_model = KMeans(n_clusters=self.kmeans_centers.shape[0],
                              init=np.vstack((iun_centers, rho_centers)).T, max_iter=1)
        idx2Region = kmeans_model.fit_predict(XGrid)

        colors = plt.cm.summer(np.linspace(0, 1, self.kmeans_centers.shape[0]))
        for k in range(self.kmeans_centers.shape[0]):
            plt.scatter(XGrid[idx2Region == k, 0], XGrid[idx2Region == k, 1],
                        color=colors[k], marker='.', label=f'Cluster {k}')
        plt.scatter(iun, rho, s=2, c='black', marker='.')
        plt.scatter(iun_centers, rho_centers, s=45, c='#FF6F00', marker='*', facecolors='none')

        plt.xlim([0, np.max(iun)])
        plt.ylim([0, np.max(rho)])

        # plt.legend(loc="upper right")
        plt.savefig(os.path.join(save_path, 'cluster centers.png'), transparent=True, dpi=1200)
        plt.close()

    def remove_specular_highlight(self, enhance=False, blending_method='normal', visualize=False, save=False):
        st_time = time.time()

        # 基于小波分解方法得到初始融合图像，作为镜面高光区域的图像源
        images = [self.polar_analyser.min_images, self.polar_analyser.avg_images]
        images = list2bhwc(images)
        dwt_fuser = DWTFuser([0.8, 0.2], [0.8, 0.2], 1,
                            wavelet='db10', caf_type='mean', cxf_type='mean')
        dwt_fused_image = imnorm(dwt_fuser.fusion(images, norm_mode=None, visualize=visualize), mode='min-max')
        if enhance:
            dwt_fused_image = apply_clahe(dwt_fused_image, self.clahe)
        specular_source = dwt_fused_image.clip(min=0, max=1)

        # 直方图均衡化算法得到增强图像，作为非镜面高光区域的图像源
        non_specular_source = self.polar_analyser.iun.copy()
        if enhance:
            non_specular_source = apply_clahe(self.polar_analyser.iun, self.clahe)

        # 基于泊松融合方法消除图像高光
        blending_image = poisson_fusion(specular_source, non_specular_source, self.ext_spec_mask,
                                        method=blending_method, visualize=False)

        # 采用 inpaint 算法消除高光
        block_size = int(np.sqrt(self.label_image.size) / 10.) * 2 + 1
        spec_mask = self.detect_outliers(blending_image, max_area=200, block_size=block_size,
                                         c=-60, threshold_type=cv2.THRESH_BINARY)
        diffuse_image = inpaint_by_diffusion(blending_image, spec_mask)

        # 得到镜面反射分量
        spec_image = (self.polar_analyser.iun - diffuse_image).clip(min=0, max=1)

        save_dir = os.path.join(self.save_dir, f'enh_{enhance}-blending_{blending_method}')
        if save:
            imsave(self.polar_analyser.min_images, 'min_image.png', save_dir, norm=True)
            imsave(self.polar_analyser.avg_images, 'avg_image.png', save_dir, norm=True)
            imsave(specular_source, 'init_fused_image.png', save_dir, norm=True)
            imsave(non_specular_source, 'ehe-avg_image.png', save_dir, norm=False)
            imsave(blending_image, 'blending_image.png', save_dir, norm=False)
            imsave(diffuse_image, 'diffuse_image.png', save_dir, norm=False)
            imsave(spec_image, 'spec_image.png', save_dir, norm=False)

        runtime = time.time() - st_time
        print(f'remove specular highlight costs {runtime:.2f} s')

        self.diffuse_image = diffuse_image
        self.spec_image = spec_image
