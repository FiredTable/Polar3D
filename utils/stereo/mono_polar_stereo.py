import os
import cv2
import torch
import time
import math
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftfreq, ifft2
from torchvision import transforms
from skimage import transform as sk_transform
from scipy.signal import medfilt2d
from scipy.ndimage import uniform_filter, gaussian_filter, binary_dilation
from typing import Optional, Union
from scipy.interpolate import interp1d
from numpy.lib.stride_tricks import as_strided

from utils.polarization_analyser import ReflectionPolarAnalyser
from utils.common import imsave, im_dtype2uint8, imnorm
from utils.tensor import img_tensor2np
from utils.U2Net import uint8_to_sample, RescaleT, ToTensorLab, U2NET, normPRED
from utils.stereo.AdelaiDepth import DepthModel, strip_prefix_if_present, scale_torch
from depth_anything_v2.dpt import DepthAnythingV2 

__all__ = [
    'PolarShaper'
]


class MultiScaleEvaluator:
    def __init__(self):
        self.eps = 1e-6
    
    @staticmethod
    def pad_to_multiple(image: np.ndarray, block_size=(32, 32)):
        """将图像填充为块大小的整数倍"""
        h, w = image.shape[:2]
        block_height, block_width = block_size

        pad_h = (block_height - h % block_height) % block_height
        pad_w = (block_width - w % block_width) % block_width

        pad_width = ((0, pad_h), (0, pad_w))
        padded_image = np.pad(image, pad_width=pad_width, mode='reflect')
        return padded_image
    
    @staticmethod
    def split_image(image: np.ndarray, block_size=(32, 32)):
        """将图像分割为块"""
        h, w = image.shape[:2]
        block_height, block_width = block_size

        num_blocks_h = h // block_height
        num_blocks_w = w // block_width

        blocks = image.reshape(num_blocks_h, block_height, num_blocks_w, block_width)
        blocks = blocks.transpose(0, 2, 1, 3)
        blocks = blocks.reshape(-1, block_height, block_width)

        return blocks
    
    @staticmethod
    def merge_blocks(blocks, original_size):
        """将块合并为原始图像大小"""
        h, w = original_size

        block_height, block_width = blocks.shape[1:3]

        num_blocks_h = math.ceil(h / block_height)
        num_blocks_w = math.ceil(w / block_width)

        merged = blocks.reshape(num_blocks_h, num_blocks_w, block_height, block_width)
        merged = merged.transpose(0, 2, 1, 3)
        merged = merged.reshape(num_blocks_h * block_height, num_blocks_w * block_width)

        return merged[:h, :w]
    
    def evaluate():
        pass


class PolarNoiseEvaluator(MultiScaleEvaluator):
    def __init__(self):
        super().__init__()
        self.eps = 1e-6
    
    def calc_entropy(self, blocks, bins=128):
        """计算每个块的信息熵"""
        blocks_num, block_height, block_width = blocks.shape
        entropy_map = np.zeros((blocks_num, block_height, block_width))
        for b in range(blocks_num):
            im = blocks[b]
            hist, bin_edges = np.histogram(im, bins=bins, range=(0.0, 1.0))
            prob = hist / (block_height * block_width)
            entropy = -np.sum(prob * np.log(prob + 1e-10))
            entropy_map[b, :, :] = np.abs(entropy)

        return entropy_map
    
    def calc_polar_noise_value(self, rho, entropy_map,
                               kernel_size, norm_method='clip',
                               beta=1.0):
        """计算偏振噪声值"""
        # 计算局部方差
        local_mean = uniform_filter(rho, size=kernel_size)
        local_var = uniform_filter(rho**2, size=kernel_size) - local_mean**2

        entropy = entropy_map.ravel()
        if norm_method == 'clip':
            entropy_division_point = entropy.mean() + entropy.std()  # 高信息熵的特征区域不被衰减
            entropy_norm = (entropy_map / entropy_division_point).clip(max=1.0)
        else:
            raise ValueError(f'Unknown norm method: {norm_method}')
        
        variances = local_var.ravel()
        if norm_method == 'clip':
            variances_division_point = variances.mean() - variances.std()  # 低信息熵的背景区域不被衰减
            local_var_norm = ((local_var - variances_division_point) / (local_var.max() - variances_division_point)).clip(min=self.eps)
        else:
            raise ValueError(f'Unknown norm method: {norm_method}')
        
        polar_noise = local_var_norm * (1 - entropy_norm)**beta
        
        return polar_noise

    def evaluate(self, vis, rho):
        block_len = int(np.floor(np.min(vis.shape[0:2]) / 64)) * 2 + 1
        block_size = (block_len, block_len)
        vis_padded = self.pad_to_multiple(vis, block_size=block_size)
        vis_blocks = self.split_image(vis_padded, block_size=block_size)
        entropy = self.calc_entropy(vis_blocks, bins=128)
        entropy_map = self.merge_blocks(entropy, original_size=vis.shape[0:2])
        polar_noise = self.calc_polar_noise_value(rho, entropy_map, kernel_size=3)
        return polar_noise
    
    
class SpecularEavluator(MultiScaleEvaluator):
    def __init__(self):
        super().__init__()
    
    @staticmethod
    def calc_specular_value(blocks: np.ndarray, mask: np.ndarray, avg: float):
        """
        根据分块，计算每个像素点的镜面值
        :param blocks: (N, H, W)
        :param mask: (N, H, W)
        :param avg: 全图像平均值
        """
        assert blocks.shape == mask.shape
        assert mask.dtype == bool

        masked_blocks = blocks * mask
        valid_counts = np.sum(mask, axis=(1, 2))
        masked_sums = np.sum(masked_blocks, axis=(1, 2))

        # 计算均值并处理全零掩码情况（避免除零错误）
        means = np.divide(masked_sums, valid_counts, 
                          where=valid_counts > 0,            # 仅对有效区域执行除法
                          out=np.zeros_like(masked_sums))    # 无效区域填充0

        means_expanded = means[:, np.newaxis, np.newaxis] 

        # 1.计算像素值与局部快均值的差值（自动广播）
        diff1 = blocks - means_expanded

        # 2.计算像素值与整体平均值的插值
        diff2 = blocks - avg

        # 计算镜面分数值，像素值小于均值的分数为0，像素值大于均值的分数为exp(v)-1
        specular_scores = np.maximum(np.exp(diff1) - 1, 0) + np.maximum(np.exp(diff2) - 1, 0)

        return specular_scores

    def evaluate(self, img):
        assert img.min() >= 0. and img.max() <= 1.

        mask = img >= 0.01
        avg = np.mean(img[mask])

        min_block_len = int(np.floor(np.min(img.shape[0:2]) / 16)) * 2 + 1
        max_block_len = int(np.floor(np.min(img.shape[0:2]) / 2)) * 2 + 1

        sepcular_maps = []
        block_len = min_block_len
        while block_len <= max_block_len: 
            block_size = (block_len, block_len)
            img_padded = self.pad_to_multiple(img, block_size=block_size)
            mask_padded = self.pad_to_multiple(mask, block_size=block_size)
            img_blocks = self.split_image(img_padded, block_size=block_size)
            mask_blocks = self.split_image(mask_padded, block_size=block_size)
            specular_scores = self.calc_specular_value(img_blocks, mask_blocks, avg)
            sepcular_maps.append(self.merge_blocks(specular_scores, original_size=img.shape[0:2]))

            block_len = 2 * block_len + 1

        # 最终的镜面分值为所有图像之和
        specular_map = np.sum(sepcular_maps, axis=0)
        return specular_map


class PolarShaper:
    def __init__(self, n_material, save_dir='.',
                 u2net_model_path='u2net.pth',
                 depth_model_path='depth_anything_v2_vitl.pth'):
        self.n_material = n_material
        self.save_dir = save_dir
        
        self.clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8))

        self.use_u2net = True
        if self.use_u2net:
            self.u2net = U2NET(3,1)
            self.u2net.load_state_dict(torch.load(u2net_model_path))
            self.u2net.cuda()
            self.u2net.eval()

        self.use_AdelaiDepth = False
        if self.use_AdelaiDepth:
            self.depth_model = DepthModel()
            checkpoint = torch.load(depth_model_path)
            self.depth_model.load_state_dict(
                strip_prefix_if_present(checkpoint['depth_model'], "module.depth_model."), 
                strict=True)
            self.depth_model.eval()
            self.depth_model.cuda()

        self.use_depth_anything_v2 = True
        if self.use_depth_anything_v2:
            model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            }
            self.depth_model = DepthAnythingV2(**model_configs['vitl'])
            self.depth_model.load_state_dict(torch.load(depth_model_path, map_location='cpu'))
            self.depth_model.eval()
            self.depth_model.cuda()

        self.eps=1e-7
        self.timer = {}

        # 构建偏振度->观测天顶角函数
        self.diffuse_dop2viewing_angle_func = self.create_diffuse_dop2viewing_angle_function(self.n_material)
        self.specular_dop2viewing_angle_func_rising, self.specular_dop2viewing_angle_func_falling =\
            self.create_specular_dop2viewing_angle_function(self.n_material)
    
    @staticmethod
    def create_diffuse_dop2viewing_angle_function(n_material, n_viewing=1.0, num=10001):
        # 曲线拟合
        reflection_polar_analyser = ReflectionPolarAnalyser(
            n_i=n_material, n_t=n_viewing, num=num)
        ndarray_theta_i, ndarray_dop = \
            reflection_polar_analyser.calc_refracted_dop_curve(save=False)
        ndarray_theta_t = reflection_polar_analyser.ndarray_theta_i2t()

        # 创建线性插值函数
        dop2viewing_angle_func = interp1d(
            ndarray_dop.real[0:len(ndarray_theta_t)], ndarray_theta_t,
            kind='linear', bounds_error=True)
        
        print(f'漫散射天顶角插值函数允许的偏振度输入范围为：\
              {dop2viewing_angle_func.x.min():.4f} ~ {dop2viewing_angle_func.x.max():.4f}')
        return dop2viewing_angle_func

    @staticmethod
    def create_specular_dop2viewing_angle_function(n_material, n_viewing=1.0, num=10001):
        # 计算观测角度和镜面反射反射偏振度的对应阵列
        reflection_polar_analyser = ReflectionPolarAnalyser(
            n_i=n_viewing, n_t=n_material, num=num)
        ndarray_theta_i, ndarray_dop = \
            reflection_polar_analyser.calc_reflected_dop_curve(save=False)
        
        # 找到最大镜面反射偏振度对应的观测天顶角索引
        max_dop_index = np.argmax(ndarray_dop.real, axis=0)

        # 创建线性插值函数
        # 1. 上升阶段
        dop2viewing_angle_func_rising = interp1d(
            ndarray_dop.real[0:max_dop_index], ndarray_theta_i[0:max_dop_index],
            kind='linear', bounds_error=True)

        print(f'镜面反射天顶角插值函数(上升阶段)允许的偏振度输入范围为：\
              {dop2viewing_angle_func_rising.x.min():.4f} ~ {dop2viewing_angle_func_rising.x.max():.4f}')
        
        # 2. 下降阶段
        dop2viewing_angle_func_falling = interp1d(
            ndarray_dop.real[max_dop_index:], ndarray_theta_i[max_dop_index:],
            kind='linear', bounds_error=True)
        
        print(f'镜面反射天顶角插值函数(下降阶段)允许的偏振度输入范围为：\
              {dop2viewing_angle_func_falling.x.min():.4f} ~ {dop2viewing_angle_func_falling.x.max():.4f}')
        
        return dop2viewing_angle_func_rising, dop2viewing_angle_func_falling
    
    def generate_mask_by_u2net(self, img: np.ndarray, u2net_thresh=0.1, vis_thresh=0.1):
        if len(img.shape) == 3:
            img = np.mean(img, axis=-1)
        uint8_image = self.clahe.apply(im_dtype2uint8(img))
        sample = uint8_to_sample(
            uint8_image, transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)]))
        
        with torch.no_grad():
            print("inferencing mask")
            inputs_test = sample['image'][None, ...]

            # 将input_test 的数据类型转化为float
            inputs_test = inputs_test.type(torch.FloatTensor)
            inputs_test = inputs_test.cuda()

            st_time = time.time()
            d1, d2, d3, d4, d5, d6, d7= self.u2net(inputs_test)
            print('U2Net Inference time: ', time.time() - st_time)

            # normalization
            pred = d1[:,0,:,:]
            pred = normPRED(pred)

            np_img = img_tensor2np(pred[0])
            resized_img = sk_transform.resize(
                np_img, (uint8_image.shape[0], uint8_image.shape[1]))
            u2net_mask = resized_img > u2net_thresh

        vis_mask = img > vis_thresh

        return u2net_mask & vis_mask

    @staticmethod
    def diffuse_dop_to_viewing_angle(rho, eta=1.6):
        """
        漫反射偏振度->观测天顶角函数
        :param rho: 漫反射偏振度
        :param eta: 折射率
        """
        numerator = 2*rho + 2*eta**2*rho - 2*eta**2 + eta**4 + rho**2 + 4*eta**2*rho**2 - eta**4*rho**2 - 4*eta**3*rho*np.sqrt((1-rho)*(1+rho)) + 1
        denominator = eta**4*rho**2 + 2*eta**4*rho + eta**4 + 6*eta**2*rho**2 + 4*eta**2*rho - 2*eta**2 + rho**2 + 2*rho + 1
        cos_theta = np.sqrt(numerator / denominator)
        viewing_angle = np.arccos(cos_theta)
        return viewing_angle
    
    @staticmethod
    def normal_to_depth(normal: np.ndarray):
        p = -normal[:, :, 0] / normal[:, :, -1].clip(min=1e-6)
        q = -normal[:, :, 1] / normal[:, :, -1].clip(min=1e-6)

        u, v = fftfreq(p.shape[0]), fftfreq(p.shape[1])  # 生成离散傅里叶变换（DFT）的频率分量
        V, U = np.meshgrid(v, u)  # 生成二维频率网格

        P_Forier_map = fft2(p)
        Q_Forier_map = fft2(q)

        Z_forier_map = (1j * U * P_Forier_map + 1j * V * Q_Forier_map) \
            / (np.power(U, 2) + np.power(V, 2) + 1e-6)
    
        return ifft2(Z_forier_map).real 
        
    @staticmethod
    def depth_to_pointcloud(Z: np.ndarray, output_dimension=3, xy_scale=2):
        height, width = Z.shape[0:2]
        U, V = np.meshgrid(np.arange(width), np.arange(height))
        points_3d = np.stack([U, V, Z], axis=-1)

        x_min, x_max = points_3d[:, :, 0].min(), points_3d[:, :, 0].max()
        y_min, y_max = points_3d[:, :, 1].min(), points_3d[:, :, 1].max()
        z_min, z_max = points_3d[:, :, 2].min(), points_3d[:, :, 2].max()

        # 归一化 X/Y 到 [z_min, z_max]
        points_3d[:, :, 0] = xy_scale * (points_3d[:, :, 0] - x_min) / (x_max - x_min) * (z_max - z_min) + z_min
        points_3d[:, :, 1] = xy_scale * (points_3d[:, :, 1] - y_min) / (y_max - y_min) * (z_max - z_min) + z_min
        points_3d[:, :, 1] = height / width * points_3d[:, :, 1]

        if output_dimension == 3:
            return points_3d  # (H, W, 3)
        elif output_dimension == 2:
            return points_3d.reshape(-1, 3)  # (H*W, 3)

    @staticmethod
    def normal_from_depth(depth: np.ndarray, TAP=15, VHRATIO=2, 
                          D=1e-10, KSIZE=7, SIGMA=1., EMETHOD=0):
        """
        从法线恢复深度
        :param depth: 深度图
        :param TAP: 插值滤波器的长度，影响梯度计算的平滑度.
        :param VHRATIO: 垂直/水平缩放比例，调整高度数据的比例敏感度.
        :param D: 设置微分时的差分值
        :pramam KSIZE: 高斯模糊核尺寸，用于平滑高度图减少噪声
        :param SIGMA: 高斯模糊的标准差，控制模糊强度
        :param EMETHOD: 图像扩展方式, 0:镜像填充, 1:边缘重复填充
        """
        TAP2 = TAP // 2
        assert TAP >= 3 and TAP % 2 == 1 
        assert VHRATIO >= 0.
        assert (KSIZE == 0) or (KSIZE >= 3 and KSIZE % 2 == 1) 
        assert 0 <= D <= 0.1

        height = depth / VHRATIO
        EXP = TAP2 + KSIZE // 2
        if EMETHOD == 1:
            # for x axis
            exp1 = np.tile(height[:, [0]], (1, EXP))
            exp2 = np.tile(height[:, [-1]], (1, EXP))
            height = np.concatenate([exp1, height, exp2], 1)

            # for y axis
            exp1 = np.tile(height[[0], :], (EXP, 1))
            exp2 = np.tile(height[[-1], :], (EXP, 1))
            height = np.concatenate([exp1, height, exp2], 0)
        else:
            # for x axis
            exp1 = height[:, -1-EXP:-1]
            exp2 = height[:, 0:EXP]
            height = np.concatenate([exp1, height, exp2], 1)

            # for y axis
            exp1 = height[-1-EXP:-1, :]
            exp2 = height[0:EXP, :]
            height = np.concatenate([exp1, height, exp2], 0)

        if KSIZE != 0:
            # make Gaussian Blur kernel matrix
            vec = np.linspace(-float(KSIZE//2), float(KSIZE//2), KSIZE)
            xx, yy = np.meshgrid(vec, vec)
            kernel = np.exp(-0.5*(xx**2.+yy**2.)/SIGMA**2.)
            kernel /= np.sum(kernel)

            # convolve each pixels with Gaussian Blur kernel matrix
            new_shape = tuple(np.subtract(height.shape, kernel.shape)+1)
            view = as_strided(height, kernel.shape+new_shape, height.strides*2)
            height = np.einsum('ij,ijkl->kl', kernel, view)

        # make filter vector for interpolation as truncated sinc
        Vl = np.linspace(float(-TAP2), float(TAP2), TAP2*2+1)
        Vsn = np.sinc(Vl-D) # for negative nearest neighbor point
        Vsp = np.sinc(Vl+D) # for positive nearest neighbor point

        # make initialization vector for xyz vector arary elements 
        rsz = np.zeros((height.shape[0]-TAP2*2, height.shape[1]-TAP2*2))
        rsd = np.full((height.shape[0]-TAP2*2, height.shape[1]-TAP2*2), D*2)

        # Differentiate the heightmap and make xyz vector array
        #  for y-axis
        rsp_y = rsz.T.copy()
        rsn_y = rsz.T.copy()
        for x in range(height.shape[1]-TAP2*2):
            rsp_y[x] = np.convolve(height.T[x+TAP2], Vsp, mode='valid')
            rsn_y[x] = np.convolve(height.T[x+TAP2], Vsn, mode='valid')
        xyz_dy = np.stack([rsz, rsd, rsn_y.T-rsp_y.T], axis=-1)

        #  for x-axis
        rsp_x = rsz.copy()
        rsn_x = rsz.copy()
        for y in range(height.shape[0]-TAP2*2):
            rsp_x[y] = np.convolve(height[y+TAP2], Vsp, mode='valid')
            rsn_x[y] = np.convolve(height[y+TAP2], Vsn, mode='valid')
        xyz_dx = np.stack([rsd, rsz, rsp_x-rsn_x], axis=-1)

        # make normal-map image and z-angle-map image
        cross = np.cross(xyz_dx, xyz_dy)
        norm = np.linalg.norm(cross, axis=-1)
        normal = cross/np.stack([norm, norm, norm], axis=-1)
        return normal

    def monocular_estimation(self, img: np.ndarray):
        img = im_dtype2uint8(img)
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        if self.use_AdelaiDepth:    
            resized_img = cv2.resize(img, (448, 448))
            img_torch = scale_torch(resized_img)[None, :, :, :].cuda()
            pred_depth = self.depth_model(img_torch).cpu().detach().numpy().squeeze()
            resized_pred_depth = cv2.resize(pred_depth, (img.shape[1], img.shape[0]))
            return resized_pred_depth
        elif self.use_depth_anything_v2:
            depth = self.depth_model.infer_image(img, input_size=518)
            return depth
        else:
            raise ValueError

    def generate_spec_mask(self, vis, rho, diffuse_dop_range,
                           vis_thresh: Union[int, str]='Otsu'):
        if len(vis.shape) == 3:
            vis = np.mean(vis, axis=-1)
        
        # 1.异常高亮区域为镜面反射区域
        specular_evaluator = SpecularEavluator()
        specular_score_map = specular_evaluator.evaluate(vis)

        uint8_specular_scores = im_dtype2uint8(imnorm(specular_score_map, mode='min-max'))
        if vis_thresh == 'Otsu':
            _, highlight_mask = cv2.threshold(src=uint8_specular_scores, thresh=0, maxval=255, 
                                              type=cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            _, highlight_mask = cv2.threshold(src=uint8_specular_scores, thresh=vis_thresh, maxval=255, 
                                              type=cv2.THRESH_BINARY)
        highlight_mask = highlight_mask > 0
        
        # 2.超过漫反射可能达到的最大偏振度的区域为镜面反射区域
        highpolar_mask = rho > diffuse_dop_range[1]

        spec_mask = highlight_mask | highpolar_mask
        return spec_mask

    @staticmethod
    def overlay_mask(img: np.ndarray, mask: np.ndarray, alpha=0.3):
        mask_on_img = img.copy()
        
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if mask[i, j]:
                    mask_on_img[i, j, 0] = alpha * img[i, j, 0] + (1 - alpha) * 168.0 / 255.0
                    mask_on_img[i, j, 1] = alpha * img[i, j, 1] + (1 - alpha) * 215.0 / 255.0
                    mask_on_img[i, j, 2] = alpha * img[i, j, 2] + (1 - alpha) * 125.0 / 255.0
                    
        return mask_on_img

    @staticmethod
    def disambiguate_viewing_angle(viewing_angle_rising, viewing_angle_falling, viewing_angle_refer):
        # 由于 viewing_angle 的范围在 [0, pi/2]，因此不需要考虑圆周性质
        diff1 = np.abs(viewing_angle_rising - viewing_angle_refer)
        diff2 = np.abs(viewing_angle_falling - viewing_angle_refer)

        mask = diff1 < diff2
        return np.where(mask, viewing_angle_rising, viewing_angle_falling)
    
    @staticmethod
    def disambiguate_azimuth_angle(phi, spec_mask, azimuth_angle_refer):
        disambiguated_phi = np.zeros_like(phi, dtype=np.float32)

        # 处理漫反射区域 (非spec_mask)
        if np.any(~spec_mask):
            # 生成候选角度：phi 和 phi+π
            phi_diff = phi[~spec_mask]
            candidates_diff = np.stack([
                phi_diff, 
                (phi_diff + np.pi) % (2 * np.pi)  # 保证范围在 [0, 2π]
            ], axis=-1)  # [N, 2]
            
            # 比较候选角度与估计值的差异
            est_azimuth_diff = azimuth_angle_refer[~spec_mask][:, None]  # [N,1]
            diffs = np.abs(candidates_diff - est_azimuth_diff) 

            # 考虑角度圆周性，取最小差异
            diffs = np.minimum(diffs, 2*np.pi - diffs)

            # 选择差异最小的候选
            best_idx = np.argmin(diffs, axis=-1)
            disambiguated_phi[~spec_mask] = np.take_along_axis(
                candidates_diff, best_idx[:, None], axis=-1).squeeze()
            
        # 处理镜面反射区域 (spec_mask)
        if np.any(spec_mask):
            # 生成候选角度：phi-π/2 和 phi+π/2
            phi_spec = phi[spec_mask]
            candidates_spec = np.stack([
                (phi_spec - np.pi/2) % (2*np.pi),
                (phi_spec + np.pi/2) % (2*np.pi)
            ], axis=-1)  # [M, 2]
            
            # 比较候选角度与估计值的差异
            est_azimuth_spec = azimuth_angle_refer[spec_mask][:, None]  # [M,1]
            diffs = np.abs(candidates_spec - est_azimuth_spec)
            diffs = np.minimum(diffs, 2*np.pi - diffs)  # 处理圆周差异
            
            # 选择差异最小的候选
            best_idx = np.argmin(diffs, axis=-1)
            disambiguated_phi[spec_mask] = np.take_along_axis(
                candidates_spec, best_idx[:, None], axis=-1).squeeze()
            
        return disambiguated_phi

    def shape_from_polarization(self, vis, rho, phi, color_map, save=False):
        # 生成目标掩膜
        if self.use_u2net:
            u2net_mask = self.generate_mask_by_u2net(vis, u2net_thresh=0.1, vis_thresh=0.01)
            imsave(u2net_mask, 'u2net_mask.png', self.save_dir) if save else None
        
        # 基于单目估计深度和法线
        monocular_depth = self.monocular_estimation(vis)
        monocular_depth[u2net_mask == 0] = np.min(monocular_depth[u2net_mask])
        monocular_depth = gaussian_filter(monocular_depth, sigma=0.5)
        monocular_normal = self.normal_from_depth(monocular_depth, VHRATIO=0.05)
        monocular_theta = np.arccos(monocular_normal[:, :, -1])
        monocular_alpha = np.arctan2(monocular_normal[:, :, 0], monocular_normal[:, :, 1])
        if save:
            imsave(monocular_depth, 'monocular_depth.png', self.save_dir, norm=True, cmap_name='turbo')
            imsave(monocular_normal, 'monocular_normal.png', self.save_dir, norm=True)
            imsave(monocular_theta, 'monocular_theta.png', self.save_dir, norm=True, cmap_name='twilight_shifted')
            imsave(monocular_alpha, 'monocular_alpha.png', self.save_dir, norm=True, cmap_name='twilight_shifted')

            monocular_points_3d = self.depth_to_pointcloud(monocular_depth, output_dimension=3, xy_scale=2)
            monocular_point_cloud = PointCloud(monocular_points_3d.reshape(-1, 3), mask=None)
            monocular_point_cloud.colorize(color_map)
            monocular_point_cloud.save_pcd(file_name='monocular_point_cloud.pcd', save_dir=self.save_dir)
            o3d.visualization.draw_geometries([monocular_point_cloud.pcd])

        vis[u2net_mask==0] = 1e-6
        vis[vis < 1e-6] = 1e-6
        rho[u2net_mask==0] = 1e-6
        rho[rho < 1e-6] = 1e-6
        phi[u2net_mask==0] = 1e-6

        # 生成偏振噪声掩膜
        evaluate_polar_noise = False
        if evaluate_polar_noise:
            polar_noise_evaluator = PolarNoiseEvaluator()
            polar_noise = polar_noise_evaluator.evaluate(vis, rho)
            imsave(polar_noise, 'polar_noise.png', self.save_dir, norm=True, cmap_name='turbo') if save else None

        # 生成镜面反射掩膜
        diffuse_dop_range = [self.diffuse_dop2viewing_angle_func.x.min(), 
                             self.diffuse_dop2viewing_angle_func.x.max()]
        spec_mask = self.generate_spec_mask(vis, rho, diffuse_dop_range, vis_thresh=220)
        spec_mask = binary_dilation(spec_mask, iterations=2)
        if save:
            imsave(spec_mask, 'spec_mask.png', self.save_dir)
            spec_on_img = self.overlay_mask(color_map, spec_mask)
            imsave(spec_on_img, 'spec_on_img.png', self.save_dir)

        # 根据偏振度计算观测天顶角
        # 漫反射区域观测天顶角
        viewing_angle_diffuse = np.zeros_like(vis, dtype=np.float32)
        viewing_angle_diffuse[rho < self.diffuse_dop2viewing_angle_func.x.max()] = \
            self.diffuse_dop2viewing_angle_func(rho[rho < self.diffuse_dop2viewing_angle_func.x.max()])
        imsave(viewing_angle_diffuse, 'viewing_angle_diffuse.png', 
               self.save_dir, norm=True, cmap_name='twilight_shifted') if save else None
        
        # 镜面反射区域观测天顶角
        viewing_angle_specular_rising = self.specular_dop2viewing_angle_func_rising(rho)
        imsave(viewing_angle_specular_rising, 'viewing_angle_specular_rising.png',
               self.save_dir, norm=True, cmap_name='twilight_shifted') if save else None  
        
        viewing_angle_specular_falling = self.specular_dop2viewing_angle_func_falling(rho)
        imsave(viewing_angle_specular_falling, 'viewing_angle_specular_falling.png',
               self.save_dir, norm=True, cmap_name='twilight_shifted') if save else None
        
        viewing_angle_specular = self.disambiguate_viewing_angle(
            viewing_angle_specular_rising, viewing_angle_specular_falling, monocular_theta)
        imsave(viewing_angle_specular, 'viewing_angle_specular.png',
               self.save_dir, norm=True, cmap_name='twilight_shifted') if save else None
        
        # 整合漫反射和镜面反射区域观测天顶角的计算结果
        viewing_angle = viewing_angle_diffuse.copy()
        viewing_angle[spec_mask] = viewing_angle_specular[spec_mask]
        imsave(viewing_angle, 'viewing_angle.png', self.save_dir, norm=True, cmap_name='twilight_shifted') if save else None

        # 根据偏振角计算法线方位角
        azimuth_angle = self.disambiguate_azimuth_angle(phi, spec_mask, monocular_alpha)
        imsave(azimuth_angle, 'azimuth_angle.png', self.save_dir, norm=True, cmap_name='twilight_shifted') if save else None

        # 根据观测天顶角和偏振角计算法线方位
        theta = (viewing_angle_diffuse).copy()
        alpha = (monocular_alpha).copy()
        normal = np.stack([np.sin(theta) * np.cos(alpha),
                           np.sin(theta) * np.sin(alpha),
                           np.cos(theta)], axis=-1)
        normal[:, :, 0] = medfilt2d(normal[:, :, 0], kernel_size=3)
        normal[:, :, 1] = medfilt2d(normal[:, :, 1], kernel_size=3)
        normal[:, :, 2] = medfilt2d(normal[:, :, 2], kernel_size=3)
        imsave(theta, 'theta.png', self.save_dir, norm=True, cmap_name='twilight_shifted') if save else None
        imsave(alpha, 'alpha.png', self.save_dir, norm=True, cmap_name='twilight_shifted') if save else None
        imsave(normal, 'normal.png', self.save_dir, norm=True) if save else None

        # 根据法线方位计算相对深度
        depth = self.normal_to_depth(normal)
        depth /= np.max(np.abs(depth))
        depth[u2net_mask == 0] = np.min(depth[u2net_mask])
        depth = gaussian_filter(depth, sigma=3)
        imsave(depth, 'depth.png', self.save_dir, norm=True, cmap_name='turbo') if save else None

        points_3d = self.depth_to_pointcloud(depth, output_dimension=3, xy_scale=4)
        point_cloud = PointCloud(points_3d.reshape(-1, 3), mask=None)
        point_cloud.colorize(color_map)
        o3d.visualization.draw_geometries([point_cloud.pcd])
        point_cloud.save_pcd(file_name='point_cloud.pcd', save_dir=self.save_dir) if save else None

        return point_cloud

