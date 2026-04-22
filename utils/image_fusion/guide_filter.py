import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter, gaussian_filter, laplace

from utils.common import imnorm

__all__ = ['GuideFilterFuser']


class GuideFilterFuser:
    def __init__(self, sigma_r=5, average_filter_size=31, r_1=45, r_2=7, eps_1=0.3, eps_2=10e-6):
        """
        初始化引导滤波融合器
        :param sigma_r: 显著性图高斯滤波的标准差
        :param average_filter_size: 基础层均匀滤波的核大小
        :param r_1: 引导滤波的第一个半径
        :param r_2: 引导滤波的第二个半径
        :param eps_1: 引导滤波的第一个正则化参数
        :param eps_2: 引导滤波的第二个正则化参数
        """
        self.sigma_r = sigma_r
        self.average_filter_size = average_filter_size
        self.r_1 = r_1
        self.r_2 = r_2
        self.eps_1 = eps_1
        self.eps_2 = eps_2

    @staticmethod
    def bhwc_imnorm(images: np.ndarray, norm_mode=None):
        normalized_images = np.zeros_like(images, dtype=np.float32)
        for k in range(images.shape[0]):
            normalized_images[k, :] = imnorm(images[k, :], mode=norm_mode)
        return normalized_images

    def fusion(self, images, norm_mode=None, visualize=False):
        images = self.bhwc_imnorm(images, norm_mode=norm_mode)

        B, H, W, C = images.shape

        # 基础层（均匀滤波）
        base_layers = np.stack(
            [uniform_filter(images[:, :, :, i], size=self.average_filter_size, mode='reflect') for i in range(C)],
            axis=-1)

        # 细节层
        detail_layers = images - base_layers

        # 显著性图（对多通道图像求和）
        saliencies = np.stack(
            [gaussian_filter(np.abs(laplace(np.sum(images[b], axis=2), mode='reflect')), self.sigma_r, mode='reflect')
             for b in range(B)], axis=0)
        mask = np.float32(np.argmax(saliencies, axis=0))

        # 引导滤波
        guided_filters = []
        for b in range(B):
            gf1 = cv2.ximgproc.createGuidedFilter(images[b], self.r_1, self.eps_1)
            gf2 = cv2.ximgproc.createGuidedFilter(images[b], self.r_2, self.eps_2)
            guided_filters.append((gf1, gf2))

        # 应用引导滤波
        weights_r1 = []
        weights_r2 = []
        for b in range(B):
            mask_b = (mask == b).astype(np.float32)

            # weight_r1 = guided_filters[b][0].filter(saliencies[b]).clip(min=0)
            # weight_r2 = guided_filters[b][1].filter(saliencies[b]).clip(min=0)

            weight_r1 = guided_filters[b][0].filter(mask_b)  # 基础层的权重
            weight_r2 = guided_filters[b][1].filter(mask_b)  # 细节层的权重

            # 将权重添加到列表中
            weights_r1.append(weight_r1)
            weights_r2.append(weight_r2)

        weights_r1 = np.stack(weights_r1, axis=0)  # 形状为 (B, H, W)
        weights_r2 = np.stack(weights_r2, axis=0)

        # 融合基础层和细节层
        fused_base = np.zeros_like(base_layers[0])
        fused_detail = np.zeros_like(detail_layers[0])

        for b in range(B):
            fused_base += base_layers[b] * weights_r1[b][..., np.newaxis]
            fused_detail += detail_layers[b] * weights_r2[b][..., np.newaxis]

        # 归一化权重
        sum_weights_r1 = np.sum(weights_r1, axis=0)[..., np.newaxis]
        sum_weights_r2 = np.sum(weights_r2, axis=0)[..., np.newaxis]

        zero_mask1 = sum_weights_r1 <= 1e-4
        fused_base = np.where(zero_mask1, 0.5, fused_base / (sum_weights_r1 + 1e-8))

        zero_mask2 = sum_weights_r2 <= 1e-4
        fused_detail = np.where(zero_mask2, 0.5, fused_detail / (sum_weights_r2 + 1e-8))

        # 重建图像
        fused_image = fused_base + fused_detail
        fused_image = np.clip(fused_image, 0, 1)  # 限制像素值在[0, 1]范围内

        if visualize:
            fig, axes = plt.subplots(5, B, figsize=(15, 10))  # 创建5行，每行num_images个子图

            # 显示原始图像
            for ax, img in zip(axes[0], images):
                ax.imshow(img)
                ax.axis('off')
            axes[0, B // 2].set_title('Original Images')

            # 显示基础层
            for ax, base in zip(axes[1], base_layers):
                ax.imshow(base)
                ax.axis('off')
            axes[1, B // 2].set_title('Base Layers')

            # 显示基础层对应的权重
            for ax, weight in zip(axes[2], weights_r1):
                ax.imshow(weight, cmap='gray')
                ax.axis('off')
            axes[2, B // 2].set_title('Weights for Base Layers')

            # 显示细节层
            for ax, detail in zip(axes[3], detail_layers):
                ax.imshow(detail)
                ax.axis('off')
            axes[3, B // 2].set_title('Detail Layers')

            # 显示细节层对应的权重
            for ax, weight in zip(axes[4], weights_r2):
                ax.imshow(weight, cmap='gray')
                ax.axis('off')
            axes[4, B // 2].set_title('Weights for Detail Layers')

            plt.tight_layout()
            plt.show(block=True)

            plt.figure()
            plt.subplot(131)
            plt.imshow(fused_base, cmap='gray')
            plt.title('fused base')
            plt.subplot(132)
            plt.imshow(fused_detail, cmap='gray')
            plt.title('fused detail')
            plt.subplot(133)
            plt.imshow(fused_image, cmap='gray')
            plt.title('fused image')
            plt.show(block=True)

        return fused_image

