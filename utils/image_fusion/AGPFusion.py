import os
import glob
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import cv2
import skimage.io
import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects
from skimage.feature import local_binary_pattern
from tqdm import tqdm
from torchvision import models
from kornia.filters import GuidedBlur
from torchvision.transforms.functional import gaussian_blur
from datetime import datetime
from pathlib import Path
from pytorch_grad_cam import GradCAMPlusPlus
from typing import Union

from ..image import batch_bm3d
from utils.tensor import tensor_imsave, tensor_imshow, img_tensor2np, img_np2tensor, tensor2np
from utils.common import im_dtype2uint8, imnorm, imsave

__all__ = [
    'SobelAttentionHead',
    'AdaptiveLocalVarianceAttentionHead',
    'AGPFuser'
]


def poisson_solver(img: torch.Tensor):
    # 获取图像的尺寸
    batch_size, channels, height, width = img.shape

    # 创建频率网格
    ky = torch.fft.fftfreq(height, d=1.0).view(-1, 1).to(img.device)
    kx = torch.fft.fftfreq(width, d=1.0).view(1, -1).to(img.device)

    # 计算频率的平方和
    k_squared = kx ** 2 + ky ** 2
    k_squared[0, 0] = 1.0  # 避免除以零

    # 对图像进行傅里叶变换
    img_fft = torch.fft.fft2(img)

    # 在频率域中求解泊松方程
    result_fft = img_fft / (-4 * torch.pi ** 2 * k_squared)

    # 将结果转换回空间域
    result = torch.fft.ifft2(result_fft).real

    return result


class NoiseLearner:
    def __init__(self, dataset_path: Path):
        self.dataset_path = dataset_path
        self.save_path = dataset_path / "local_variances.pt"
        self.threshold = None

    def compute(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        extensions = ['*.bmp', '*.png', '*.jpg']
        images_list = [file for ext in extensions for file in glob.glob(os.path.join(self.dataset_path, ext))]

        all_vars = []
        for k in tqdm(range(len(images_list))):
            im = imnorm(skimage.io.imread(images_list[k], as_gray=True))
            tensor_im = img_np2tensor(im).to(device)

            channels, height, width = tensor_im.shape
            block_sizes = AGPFuser.get_block_sizes((height, width))
            kernel_size = block_sizes['medium_scale'][0][0]

            pad = torch.nn.ReflectionPad2d(kernel_size // 2)
            padded = pad(tensor_im)
            local_mean = F.avg_pool2d(padded, kernel_size, stride=kernel_size // 2)
            local_var = F.avg_pool2d(padded ** 2, kernel_size, stride=kernel_size // 2) - local_mean ** 2
            all_vars.append(local_var.ravel())

        all_vars_tensor = torch.cat(all_vars).cpu()
        torch.save(all_vars_tensor, self.save_path)

    def visualize(self):
        if not self.save_path.exists():
            raise FileNotFoundError(f"Local variances data not found at {self.save_path}, run compute() first")

        variances = torch.load(self.save_path).numpy().flatten().clip(min=0)
        self.threshold = np.mean(variances) + 5 * np.std(variances)
        print(f'learned threshold is {self.threshold}')

        plt.figure(figsize=(12, 6))
        max_var = variances.max()
        bins = np.linspace(0, max_var, 500)

        plt.hist(variances, bins=bins, alpha=0.7, edgecolor='none')
        plt.title("Local Variance Distribution (Linear Scale)")
        plt.xlabel("Variance Value")
        plt.ylabel("Frequency")
        plt.grid(True, which='both', alpha=0.4)
        plt.axvline(self.threshold, color='red', label=f'Noise Threshold = {self.threshold:.2e}')
        plt.legend()
        plt.show(block=True)


# graditude attention head
class SobelAttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

        self.pad = nn.ReflectionPad2d(1)

    def forward(self, x):
        x_padded = self.pad(x)
        grad_x = F.conv2d(x_padded, self.sobel_x.to(x.device), padding=0)
        grad_y = F.conv2d(x_padded, self.sobel_y.to(x.device), padding=0)
        gradient_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2)

        attention_scores = gradient_magnitude
        return attention_scores


class LaplacianAttentionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.laplacian = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32).view(1, 1, 3, 3)
        self.pad = nn.ReflectionPad2d(1)

    def forward(self, x):
        x_padded = self.pad(x)
        laplacian_response = F.conv2d(x_padded, self.laplacian.to(x.device), padding=0)

        attention_scores = torch.abs(laplacian_response)
        return attention_scores


# texture attention head
class GaborAttentionHead(nn.Module):
    def __init__(self, kernel_size=5, sigma=2.0, theta=0, lambd=5.0, gamma=1.0):
        """
        :param kernel_size:
        :param sigma: 控制 Gabor 滤波器的高斯包络的宽度，决定了滤波器的尺度。通常取值范围为 0.5 到 5.0
        :param theta: 控制 Gabor 滤波器的方向，决定了滤波器对哪个方向的纹理敏感。
        :param lambd: 控制 Gabor 滤波器的正弦分量的波长，决定了滤波器的频率。通常取值范围为 1.0 到 10.0。
        :param gamma: 控制 Gabor 滤波器的高斯包络的纵横比，决定了滤波器的形状。通常取值范围为 0.2 到 1.0
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.theta = theta
        self.lambd = lambd
        self.gamma = gamma

        # 创建 Gabor 滤波器
        self.gabor_kernel = self._create_gabor_kernel()

    def _create_gabor_kernel(self):
        kernel = torch.zeros(self.kernel_size, self.kernel_size)
        radius = (self.kernel_size - 1) // 2
        for x in range(-radius, radius + 1):
            for y in range(-radius, radius + 1):
                x_theta = x * math.cos(self.theta) + y * math.sin(self.theta)
                y_theta = -x * math.sin(self.theta) + y * math.cos(self.theta)
                kernel[x + radius, y + radius] = (
                        math.exp(-(x_theta ** 2 + self.gamma ** 2 * y_theta ** 2) / (2 * self.sigma ** 2))
                        * math.cos(2 * math.pi * x_theta / self.lambd))
        kernel = kernel / kernel.sum()
        return kernel.view(1, 1, self.kernel_size, self.kernel_size)

    def forward(self, x):
        gabor_response = F.conv2d(x, self.gabor_kernel.to(x.device), padding=self.kernel_size // 2)
        attention_scores = torch.abs(gabor_response)  # 取绝对值
        return attention_scores


# GLCM 计算时间较久
class GLCMAttentionHead(nn.Module):
    def __init__(self, gray_levels=16, distance=1, angles=(0, 45, 90, 135)):
        super().__init__()
        self.gray_levels = gray_levels
        self.distance = distance
        self.angles = angles

    def _compute_glcm(self, image):
        # 将图像量化为 gray_levels 个灰度级
        quantized_image = (image * (self.gray_levels - 1)).to(torch.uint8)

        # 初始化 GLCM
        glcm = torch.zeros((self.gray_levels, self.gray_levels), dtype=torch.float32).to(image.device)

        # 计算 GLCM
        for angle in self.angles:
            if angle == 0:  # 水平方向
                shifted_image = torch.roll(quantized_image, shifts=self.distance, dims=1)
            elif angle == 45:  # 对角线方向
                shifted_image = torch.roll(quantized_image, shifts=(self.distance, -self.distance), dims=(0, 1))
            elif angle == 90:  # 垂直方向
                shifted_image = torch.roll(quantized_image, shifts=self.distance, dims=0)
            elif angle == 135:  # 反对角线方向
                shifted_image = torch.roll(quantized_image, shifts=(self.distance, self.distance), dims=(0, 1))
            else:
                raise ValueError("Unsupported angle. Choose from [0, 45, 90, 135].")

            # 统计灰度共生矩阵
            for i in range(self.gray_levels):
                for j in range(self.gray_levels):
                    glcm[i, j] += torch.sum((quantized_image == i) & (shifted_image == j))

        # 归一化 GLCM
        glcm = glcm / glcm.sum()
        return glcm

    def _extract_texture_features(self, glcm, prop='contrast'):
        if prop == 'contrast':
            results = torch.sum(
                glcm * (torch.arange(self.gray_levels).view(-1, 1) - torch.arange(self.gray_levels)) ** 2)
        elif prop == 'energy':
            results = torch.sum(glcm ** 2)
        elif prop == 'entropy':
            results = -torch.sum(glcm * torch.log(glcm + 1e-10))
        else:
            raise ValueError('%s is an invalid property' % prop)
        return results

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        attention_scores = torch.zeros_like(x)
        for b in tqdm(range(batch_size)):  # 耗时
            for c in range(channels):
                glcm = self._compute_glcm(x[b, c])
                entropy = self._extract_texture_features(glcm, prop='entropy')
                attention_scores[b, c] = entropy

        return attention_scores


class CannyAttentionHead(nn.Module):
    def __init__(self, low_threshold=100, high_threshold=200):
        """
        :param low_threshold: 梯度值小于 low_threshold 的像素会被直接去除
        :param high_threshold: 像素的梯度值大于 high_threshold 的点会被保留为边缘
        """
        super().__init__()
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold

    def forward(self, x):
        channels, height, width = x.shape
        attention_scores = torch.zeros_like(x)

        for c in range(channels):
            image_np = x[c].cpu().numpy()
            image_np = (image_np * 255).astype(np.uint8)
            edges = cv2.Canny(image_np, self.low_threshold, self.high_threshold)
            attention_scores[c] = torch.from_numpy(edges / 255.0).to(x.device)

        return attention_scores


class GradCAMPPAttentionHead(nn.Module):
    def __init__(self, model, target_layer):
        super().__init__()
        self.model = model
        self.target_layer = target_layer
        self.cam = GradCAMPlusPlus(model=self.model, target_layers=self.target_layer, reshape_transform=None)

    def forward(self, x, targets=None):
        if x.shape[0] == 1:
            x = x.expand(3, -1, -1).unsqueeze(0)
        grayscale_cam = self.cam(input_tensor=x, targets=targets)
        grayscale_cam[grayscale_cam < 0.1] = 0
        grad_cam = torch.from_numpy(grayscale_cam).to(self.cam.device)

        attention_scores = grad_cam
        return attention_scores


class AdaptiveLocalVarianceAttentionHead(nn.Module):
    def __init__(self, kernel_size, alpha=5.0, beta=0.5, eps=1e-8):
        """
        Args:
           kernel_size (int): 计算局部方差的窗口大小
           alpha (float): 控制 Sigmoid 函数斜率的参数
           beta (float): 控制熵权重的参数（默认 1.0）
           eps (float): 防止除零的小量（默认 1e-8）
        """
        super().__init__()
        self.kernel_size = kernel_size
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

        self.pad = nn.ReflectionPad2d(kernel_size // 2)

    def forward(self, x, entropy_map):
        x_padded = self.pad(x)
        local_mean = F.avg_pool2d(x_padded, kernel_size=self.kernel_size, stride=1)
        local_var = F.avg_pool2d(x_padded ** 2, kernel_size=self.kernel_size, stride=1) - local_mean ** 2

        norm_method = 'clip'

        entropy = entropy_map.ravel()
        if norm_method == 'clip':
            entropy_division_point = entropy.mean() + entropy.std()  # 高信息熵的特征区域不被衰减
            entropy_norm = (entropy_map / entropy_division_point).clip(max=1.0)
        elif norm_method == 'min-max':
            entropy_norm = (entropy_map - entropy.min()) / (entropy.max() - entropy.min())
        else:
            raise ValueError(f'Unknown norm method: {norm_method}')

        variances = local_var.ravel()
        if norm_method == 'clip':
            variances_division_point = variances.mean() - variances.std()  # 低信息熵的背景区域不被衰减
            local_var_norm = ((local_var - variances_division_point) / (local_var.max() - variances_division_point)).clip(min=self.eps)
        elif norm_method == 'min-max':
            local_var_norm = (local_var - variances.min()) / (variances.max() - variances.min())
        else:
            raise ValueError(f'Unknown norm method: {norm_method}')

        preliminary_scores = local_var_norm * (1 - entropy_norm) ** self.beta
        # attention_scores = torch.sigmoid(self.alpha * (preliminary_scores - 0.5))

        return preliminary_scores


class EntropyAttentionHead(nn.Module):
    def __init__(self, bins=256):
        super().__init__()
        self.bins = bins

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        attention_scores = torch.zeros_like(x)

        for b in range(batch_size):
            for c in range(channels):
                im = x[b, c]
                hist = torch.histc(im, bins=self.bins, min=0.0, max=1.0)
                prob = hist / (height * width)
                entropy = -torch.sum(prob * torch.log(prob + 1e-10))
                attention_scores[b, c] = torch.abs(entropy)
        return attention_scores


class LBPAttentionHead(nn.Module):
    def __init__(self, radius=1, n_points=8, method='uniform'):
        super().__init__()
        self.radius = radius
        self.n_points = n_points
        self.method = method

    def lbp(self, x):
        """
        计算 LBP 特征并返回归一化的 LBP 特征图
        :param x: 输入的图像张量 (channels, height, width)
        :return: LBP 特征图 (channels, height, width)
        """
        channels, height, width = x.size()
        lbp_maps = []

        for c in range(channels):
            img = img_tensor2np(x[c])
            uint8_img = (img * 255).astype(np.uint8)
            lbp_map = local_binary_pattern(uint8_img, self.n_points, self.radius, method=self.method) / 255.

            # 将 LBP 映射从 NumPy 转换回 PyTorch 张量
            lbp_map = torch.tensor(lbp_map, dtype=torch.float32).unsqueeze(0).to(x.device)
            lbp_maps.append(lbp_map)

        lbp_maps = torch.cat(lbp_maps, dim=0)
        return lbp_maps

    def forward(self, x):
        """
        前向传播，计算 LBP 特征图
        :param x: 输入图像张量 (channels, height, width)
        :return: LBP 特征图作为注意力得分 (channels, height, width)
        """
        lbp_map = self.lbp(x)

        attention_scores = lbp_map
        return attention_scores


class AGPFuser(nn.Module):
    """physical attention guided polarization image fusion"""
    def __init__(self,
                 save_dir: Union[Path, str] = None,
                 use_gah=True, use_tah=True, use_eah=True, use_sah=True, use_nah=True, use_ms_enh=True,
                 noise_method='soft_mask'):
        super().__init__()

        # 均值滤波参数
        self.average_filter_size = 31

        # 导向滤波参数
        self.r_1 = 45
        self.r_2 = 7
        self.eps_1 = 0.3
        self.eps_2 = 10e-6

        if save_dir is None:
            time_str = datetime.strftime(datetime.now(), "%Y-%m-%d_%H-%M-%S")
            self.save_dir = Path(os.path.join('runs', time_str))
        else:
            self.save_dir = save_dir

        # 启用功能模块
        self.use_gah = use_gah
        self.use_tah = use_tah
        self.use_eah = use_eah
        self.use_sah = use_sah
        self.use_nah = use_nah
        self.use_ms_enh = use_ms_enh
        self.noise_method = noise_method

        self.runtime = {}

    @staticmethod
    def pad_to_multiple(image_tensor: torch.Tensor, block_size=(32, 32)):
        c, h, w = image_tensor.shape
        block_height, block_width = block_size

        pad_h = (block_height - h % block_height) % block_height
        pad_w = (block_width - w % block_width) % block_width

        padded_image = F.pad(image_tensor, (0, pad_w, 0, pad_h), mode='reflect')
        return padded_image

    @staticmethod
    def split_image(image_tensor: torch.Tensor, block_size=(32, 32)):
        c, h, w = image_tensor.shape
        block_height, block_width = block_size

        num_blocks_h = h // block_height
        num_blocks_w = w // block_width

        blocks = (image_tensor
                  .unfold(1, block_height, block_height)
                  .unfold(2, block_width, block_width)
                  .contiguous()
                  .view(c, num_blocks_h, num_blocks_w, block_height, block_width)
                  .permute(1, 2, 0, 3, 4)
                  .contiguous()
                  .view(-1, c, block_height, block_width))

        return blocks

    @staticmethod
    def merge_blocks(blocks, original_size):
        c, h, w = original_size

        block_height, block_width = blocks.shape[2], blocks.shape[3]

        num_blocks_h = math.ceil(h / block_height)
        num_blocks_w = math.ceil(w / block_width)

        merged_tensor = (blocks
                         .view(num_blocks_h, num_blocks_w, c, block_height, block_width)
                         .permute(2, 0, 3, 1, 4)
                         .contiguous()
                         .view(c, num_blocks_h * block_height, num_blocks_w * block_width))

        if merged_tensor.shape[1] > h or merged_tensor.shape[2] > w:
            merged_tensor = merged_tensor[:, :h, :w]

        return merged_tensor

    @staticmethod
    def get_block_sizes(shape: tuple):
        height, width = shape
        short_side = min(height, width)

        min_block_size = 3
        max_block_size = short_side // 2
        max_block_size = max(max_block_size, min_block_size)

        block_sizes = []
        size = min_block_size
        while size <= max_block_size:
            block_sizes.append((size, size))
            size = 2 * size - 1

        threshold1 = short_side // 64
        threshold2 = short_side // 8

        small_scale = [bs for bs in block_sizes if bs[0] <= threshold1]
        medium_scale = [bs for bs in block_sizes if threshold1 < bs[0] <= threshold2]
        large_scale = [bs for bs in block_sizes if bs[0] > threshold2]

        return {
            'small_scale': small_scale,
            'medium_scale': medium_scale,
            'large_scale': large_scale
        }

    @staticmethod
    def tensor_imnorm(images: torch.Tensor, norm_mode=None):
        images = images.to(torch.float32)
        if norm_mode is None:
            return images
        elif norm_mode == 'image':
            # 对每张图像进行独立的归一化
            batch_size = images.shape[0]
            normalized_images = torch.zeros_like(images)
            for b in range(batch_size):
                image = images[b, :]
                min_val = image.min()
                max_val = image.max()
                if max_val - min_val == 0:
                    normalized_images[b, :] = torch.zeros_like(image)
                else:
                    normalized_images[b, :] = (image - min_val) / (max_val - min_val)
            return normalized_images
        elif norm_mode == 'batch':
            # 对整个batch进行归一化
            min_val = images.min()
            max_val = images.max()
            if max_val - min_val == 0:
                return torch.zeros_like(images)
            else:
                return (images - min_val) / (max_val - min_val)
        else:
            raise ValueError('unsupported norm mode')

    def decompose_image(self, images: torch.Tensor, method: str, save=False):
        if method == 'poisson':
            base_layer = poisson_solver(images)
        elif method == 'gaussian':
            base_layer = gaussian_blur(images, kernel_size=[15, 15], sigma=[3, 3])
        elif method == 'avg_pool':
            base_layer = F.avg_pool2d(images, self.average_filter_size, stride=1,
                                      padding=self.average_filter_size // 2, count_include_pad=False)
        else:
            raise ValueError(f'Invalid decomposition method: {method}')

        detail_layer = images - base_layer

        if save:
            for b in range(images.shape[0]):
                tensor_imsave(base_layer[b], f'img{b}_base_layer.png', save_path=self.save_dir)
                tensor_imsave(detail_layer[b], f'img{b}_detail_layer.png', save_path=self.save_dir, norm=True)

        return base_layer, detail_layer

    def compute_attention_maps(self, images: torch.Tensor, vis_src_idx=0, save=False):
        batch_size, channels, height, width = images.shape
        block_sizes = self.get_block_sizes((height, width))

        attention_maps = {}
        for image_id in range(batch_size):
            attention_maps[f'img{image_id}'] = {}

            if self.use_sah:
                st = time.time()
                # 计算语义注意力头
                attention_maps[f'img{image_id}']['semantic_attention_maps'] = {}

                # GradCAMPP 注意力头：zao/vgg16/inception_v3/densenet121/vit_b_16
                model = models.efficientnet_b0(weights=None)
                model_weights_path = os.path.join('..', 'models', 'efficientnet_b0_rwightman-3dd342df.pth')
                model.load_state_dict(torch.load(model_weights_path))
                model = model.to(images.device)
                target_layer = [model.features[-1]]
                head = GradCAMPPAttentionHead(model=model, target_layer=target_layer)
                head_name = head.__class__.__name__
                attention_maps[f'img{image_id}']['semantic_attention_maps'][head_name] = []
                head_map = head(images[image_id, :])
                attention_maps[f'img{image_id}']['semantic_attention_maps'][head_name].append(head_map)
                if save:
                    tensor_imsave(head_map, f'img{image_id}_{head_name}.png', self.save_dir, norm=True)

                self.runtime[f'semantic_attention_maps_img{image_id}'] = time.time() - st

            if self.use_tah:
                st = time.time()
                # 纹理注意力图
                attention_maps[f'img{image_id}']['texture_attention_maps'] = {}

                # LBP 注意力头：设置不同尺寸的 radius 和 n_points 实现多尺度分析
                head_name = LBPAttentionHead.__name__
                attention_maps[f'img{image_id}']['texture_attention_maps'][head_name] = []
                for block_size in [(1, 1)]:
                    bs_head = LBPAttentionHead(radius=block_size[0], n_points=4 * block_size[0] ** 2 + 4 * block_size[0],
                                               method='uniform')
                    bs_head_map = bs_head(images[image_id, :])
                    attention_maps[f'img{image_id}']['texture_attention_maps'][head_name].append(bs_head_map)
                    if save:
                        tensor_imsave(bs_head_map, f'img{image_id}_{head_name}_block{block_size}.png',
                                      self.save_dir, norm=True)

                # Canny 注意力头：先计算整体的 Canny 图像，然后再进行多尺度的池化平均
                head = CannyAttentionHead()
                head_name = head.__class__.__name__
                attention_maps[f'img{image_id}']['texture_attention_maps'][head_name] = []
                head_map = head(images[image_id, :])
                for block_size in block_sizes['medium_scale']:
                    bs_head_map = F.avg_pool2d(head_map, block_size, stride=1,
                                               padding=block_size[0] // 2, count_include_pad=False)
                    attention_maps[f'img{image_id}']['texture_attention_maps'][head_name].append(bs_head_map)
                    if save:
                        tensor_imsave(bs_head_map, f'img{image_id}_{head_name}_block{block_size}.png',
                                      self.save_dir, norm=True)

                if_use_gabor = False
                if if_use_gabor:
                    # Gabor 注意力头：通过设置不同大小的kernel_size实现多尺度分析（对超参数过于敏感）
                    head_name = GaborAttentionHead.__name__
                    attention_maps[f'img{image_id}']['texture_attention_maps'][head_name] = []
                    for block_size in block_sizes['medium_scale']:
                        head = GaborAttentionHead(kernel_size=block_size[0])
                        head_map = head(images[image_id, :])
                        attention_maps[f'img{image_id}']['texture_attention_maps'][head_name].append(head_map)
                        if save:
                            tensor_imsave(head_map, f'img{image_id}_{head_name}_block{block_size}.png',
                                          self.save_dir, norm=True)

                if_use_glcm = False
                if if_use_glcm:
                    # GLCM 注意力头：把图像分割成多种大小的图像块，针对每个图像块计算 GLCM 信息熵（费时！）
                    head = GLCMAttentionHead(gray_levels=8)
                    head_name = head.__class__.__name__
                    attention_maps[f'img{image_id}']['texture_attention_maps'][head_name] = []
                    for block_size in block_sizes['medium_scale']:
                        image_padded = self.pad_to_multiple(images[image_id, :], block_size)
                        blocks = self.split_image(image_padded, block_size)
                        blocks_map = head(blocks)
                        merged_map = self.merge_blocks(blocks_map, original_size=(channels, height, width))
                        attention_maps[f'img{image_id}']['texture_attention_maps'][head_name].append(merged_map)
                        if save:
                            tensor_imsave(merged_map, f'img{image_id}_{head_name}_block{block_size}.png',
                                          self.save_dir, norm=True)
                self.runtime[f'texture_attention_maps_img{image_id}'] = time.time() - st

            if self.use_gah:
                st = time.time()
                # 计算梯度注意力图
                attention_maps[f'img{image_id}']['gradient_attention_maps'] = {}

                # Sobel 注意力头：先计算整体的 Sobel 图像，然后再进行多尺度的池化平均
                head = SobelAttentionHead()
                head_name = head.__class__.__name__
                attention_maps[f'img{image_id}']['gradient_attention_maps'][head_name] = []
                head_map = head(images[image_id, :])
                for block_size in block_sizes['small_scale']:
                    bs_head_map = F.avg_pool2d(head_map, block_size, stride=1,
                                               padding=block_size[0] // 2, count_include_pad=False)
                    attention_maps[f'img{image_id}']['gradient_attention_maps'][head_name].append(bs_head_map)
                    if save:
                        tensor_imsave(bs_head_map, f'img{image_id}_{head_name}_block{block_size}.png',
                                      self.save_dir, norm=True)

                # Laplacian 注意力头：先计算整体的 Sobel 图像，然后再进行多尺度的池化平均
                head = LaplacianAttentionHead()
                head_name = head.__class__.__name__
                attention_maps[f'img{image_id}']['gradient_attention_maps'][head_name] = []
                head_map = head(images[image_id, :])
                for block_size in block_sizes['small_scale']:
                    bs_head_map = F.avg_pool2d(head_map, block_size, stride=1,
                                               padding=block_size[0] // 2, count_include_pad=False)
                    attention_maps[f'img{image_id}']['gradient_attention_maps'][head_name].append(bs_head_map)
                    if save:
                        tensor_imsave(bs_head_map, f'img{image_id}_{head_name}_block{block_size}.png',
                                      self.save_dir, norm=True)

                self.runtime[f'gradient_attention_maps_img{image_id}'] = time.time() - st

            entropy_map = None
            if self.use_eah:
                st = time.time()
                # 计算信息注意力图
                attention_maps[f'img{image_id}']['infor_attention_maps'] = {}

                # 信息熵注意力头：把图像分割成多种大小的图像块，针对每个图像块计算信息熵
                head = EntropyAttentionHead()
                head_name = head.__class__.__name__
                attention_maps[f'img{image_id}']['infor_attention_maps'][head_name] = []
                for block_size in block_sizes['medium_scale']:
                    image_padded = self.pad_to_multiple(images[image_id, :], block_size)
                    blocks = self.split_image(image_padded, block_size)
                    blocks_map = head(blocks)
                    merged_map = self.merge_blocks(blocks_map, original_size=(channels, height, width))
                    attention_maps[f'img{image_id}']['infor_attention_maps'][head_name].append(merged_map)
                    if save:
                        tensor_imsave(merged_map, f'img{image_id}_{head_name}_block{block_size}.png',
                                      self.save_dir, norm=True)
                entropy_map = attention_maps[f'img0']['infor_attention_maps'][head_name][0]

                self.runtime[f'infor_attention_maps_img{image_id}'] = time.time() - st

            if self.use_nah:
                st = time.time()
                # 将可见光图像的信息熵图像作为参考图像，以计算偏振特征图像的噪声
                if entropy_map is None:
                    block_size = block_sizes['medium_scale'][0]
                    image_padded = self.pad_to_multiple(images[vis_src_idx, :], block_size)
                    blocks = self.split_image(image_padded, block_size)
                    head = EntropyAttentionHead()
                    blocks_map = head(blocks)
                    merged_map = self.merge_blocks(blocks_map, original_size=(channels, height, width))
                    entropy_map = merged_map

                # 计算噪声注意力图
                attention_maps[f'img{image_id}']['noise_attention_maps'] = {}

                # 自适应局部方差注意力头
                head = AdaptiveLocalVarianceAttentionHead(kernel_size=block_sizes['medium_scale'][0][0])
                head_name = head.__class__.__name__
                attention_maps[f'img{image_id}']['noise_attention_maps'][head_name] = []
                head_map = head(images[image_id, :], entropy_map)
                attention_maps[f'img{image_id}']['noise_attention_maps'][head_name].append(head_map)
                if save:
                    tensor_imsave(head_map, f'img{image_id}_{head_name}_block{head.kernel_size}.png',
                                  self.save_dir, norm=True)
                    tensor_imsave(self.histogram_equalization(head_map),
                                  f'img{image_id}_{head_name}_block{head.kernel_size}_stretched.png',
                                  self.save_dir, norm=True)

                self.runtime[f'noise_attention_maps_img{image_id}'] = time.time() - st

        return attention_maps

    @staticmethod
    def histogram_equalization(x, bins=256):
        # 转换到CPU和NumPy（若需GPU实现可使用cumsum替代）
        x_np = img_tensor2np(x)

        # 计算直方图和累积分布
        hist, _ = np.histogram(x_np, bins=bins)
        cdf = hist.cumsum()
        cdf = (cdf - cdf.min()) / (cdf.max() - cdf.min())

        # 均衡化映射
        equalized = np.interp(x_np.flatten(), np.linspace(x_np.min(), x_np.max(), bins), cdf)
        return torch.from_numpy(equalized.reshape(x.shape)).to(x.device)

    @staticmethod
    def gamma_correction(x, gamma=2.0):
        """非线性亮度调节"""
        x_normalized = (x - x.min()) / (x.max() - x.min())
        return torch.pow(x_normalized, gamma)

    @staticmethod
    def adaptive_contrast_stretch(x, percentile_low=0.1, percentile_high=99.9):
        """基于百分位的动态范围拉伸"""
        # 计算指定百分位值
        low = torch.quantile(x, percentile_low/100)
        high = torch.quantile(x, percentile_high/100)

        # 线性拉伸到[0,1]
        x_stretched = (x - low) / (high - low)
        return torch.clamp(x_stretched, 0, 1)

    @staticmethod
    def get_ostu_mask(image: torch.Tensor, offset_ratio=0.3) -> torch.Tensor:
        """
        使用 OSTU 算法得到图像的掩膜
        :param offset_ratio: 偏置量系数
        :param image: 输入的图像张量 (C, H, W)
        :return: OSTU 掩膜 (C, H, W)
        """
        np_image = image.detach().cpu().numpy()
        ostu_masks = []
        for c in range(np_image.shape[0]):
            norm_image = imnorm(np_image[c], mode='min-max')
            uint8_image = im_dtype2uint8(norm_image)
            ostu_threshold, _ = cv2.threshold(uint8_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

            offset = (255. - ostu_threshold) * offset_ratio

            adjusted_threshold = ostu_threshold + offset
            _, ostu_mask = cv2.threshold(uint8_image, adjusted_threshold, 255, cv2.THRESH_BINARY)

            min_area = int(np.sqrt(ostu_mask.size) / 200.) ** 2
            ostu_mask = remove_small_objects(ostu_mask.astype(bool), min_size=min_area)
            ostu_masks.append(torch.tensor(ostu_mask).to(image.device))

        ostu_masks = torch.stack(ostu_masks, dim=0)
        return ostu_masks

    @staticmethod
    def enhance_significant_features(value_map, saliency_mask, sigma=2.0):
        """根据特征强度动态地生成增强系数，够平滑地增强显著特征"""
        coefficient = torch.exp(-value_map / (2 * sigma**2)) * saliency_mask
        new_value_map = value_map + coefficient * value_map
        return new_value_map

    @staticmethod
    def compute_decay_coefficient(decay_map: torch.Tensor, quantile1: float, quantile2: float):
        """
        根据 decay_map 的值计算衰减系数
        Args:
            decay_map (torch.Tensor): 输入的 decay_map
            quantile1 (float): 第一个分位数（0.683 分位数）
            quantile2 (float): 第二个分位数（0.955 分位数）
        Returns:
            torch.Tensor: 衰减系数矩阵 decay_coefficient
            torch.Tensor: 噪声掩膜 noise_mask
        """
        decay_coefficient = torch.zeros_like(decay_map)

        # 区间 [0, quantile1]: 衰减系数为 1
        mask1 = (decay_map <= quantile1)
        decay_coefficient[mask1] = 1.0

        # 区间 [quantile1, quantile2]: 衰减系数从 1 线性衰减到 0
        mask2 = (decay_map > quantile1) & (decay_map <= quantile2)
        decay_coefficient[mask2] = 1.0 - (decay_map[mask2] - quantile1) / (quantile2 - quantile1)
        decay_coefficient[mask2].clip(min=0)

        # 区间 > quantile2: 衰减系数为 0
        mask3 = (decay_map > quantile2)
        decay_coefficient[mask3] = 0.0

        noise_mask = torch.zeros_like(decay_map, dtype=torch.float32)
        noise_mask[mask3] = 1
        noise_mask[mask2] = 0.5

        return decay_coefficient, noise_mask

    def compute_weight_maps(self, attention_maps, main_src_idx=0, vis_src_idx=0, save=False):
        value_maps = {}
        decay_maps = {}
        image_ids = list(attention_maps.keys())

        head_num = 0
        for image_id in image_ids:
            value_maps[image_id] = None
            decay_maps[image_id] = None

            for map_type, heads in attention_maps[image_id].items():
                # map_type: str = 'noise_attention_maps' or 'semantic_attention_maps' or 'texture_attention_maps'
                # heads: dict = {'GradCAMPPAttentionHead':list = [Tensor]}
                #   or {'LBPAttentionHead':list = [Tensor], 'CannyAttentionHead':list = [Tensor]}
                if map_type == 'noise_attention_maps':
                    for head_name, head_maps in heads.items():
                        # head_name: str = 'AdaptiveLocalVarianceAttentionHead'
                        # head_maps: list = [Tensor1, Tensor2, ...]
                        if decay_maps[image_id] is None:
                            example_tensor = next(iter(head_maps))
                            decay_maps[image_id] = torch.zeros_like(example_tensor, dtype=torch.float32)
                        attention_map = torch.sum(torch.stack(head_maps), dim=0)
                        total_attention_map = torch.zeros_like(attention_map, dtype=torch.float32)
                        for other_image_id in image_ids:
                            if other_image_id != image_id:
                                other_attention_map = torch.sum(
                                    torch.stack(attention_maps[other_image_id][map_type][head_name]), dim=0)
                                total_attention_map += other_attention_map
                        denominator = torch.max(attention_map + total_attention_map)
                        head_decay_map = attention_map / denominator
                        decay_maps[image_id] += head_decay_map
                else:
                    head_num += len(heads)
                    for head_name, head_maps in heads.items():
                        if value_maps[image_id] is None:
                            example_tensor = next(iter(head_maps))
                            value_maps[image_id] = torch.zeros_like(example_tensor, dtype=torch.float32)

                        attention_map = torch.sum(torch.stack(head_maps), dim=0)
                        total_attention_map = torch.zeros_like(attention_map, dtype=torch.float32)
                        for other_image_id in image_ids:
                            if other_image_id != image_id:
                                other_attention_map = torch.sum(
                                    torch.stack(attention_maps[other_image_id][map_type][head_name]), dim=0)
                                total_attention_map += other_attention_map
                        denominator = torch.max(attention_map + total_attention_map)
                        head_value_map = attention_map / denominator
                        value_maps[image_id] += head_value_map

        if save:
            for image_id in image_ids:
                tensor_imsave(value_maps[image_id], f'{image_id}_value_map.png', self.save_dir, norm=True)
                if self.use_nah:
                    tensor_imsave(decay_maps[image_id], f'{image_id}_decay_map.png', self.save_dir, norm=True)

        if self.use_nah:
            # 对同源的decay_values进行统计，计算阈值
            all_decay_values = []
            for image_id in image_ids:
                all_decay_values.append(decay_maps[image_id].flatten().cpu())
            all_decay_values = torch.cat(all_decay_values, dim=0)

            draw_decay_values_histogram = False
            if draw_decay_values_histogram:
                plt.hist(all_decay_values.numpy(), bins=100)
                plt.show(block=True)

            # sigma-0.683, 2sigma-0.955, 3sigma-0.997
            quantile1 = np.quantile(all_decay_values, 0.683)
            quantile2 = np.quantile(all_decay_values, 0.955)
            quantile3 = np.quantile(all_decay_values, 0.977)

            noise_masks = {}
            for image_id in image_ids:
                if image_id == image_ids[vis_src_idx]:
                    # 对可见光图像不进行噪声衰减
                    decay_coefficient = torch.ones_like(decay_maps[image_id], dtype=torch.float32)
                    noise_mask = torch.zeros_like(decay_maps[image_id], dtype=torch.float32)
                else:
                    # 对偏振特征图像采用噪声衰减
                    decay_coefficient, noise_mask = (
                        self.compute_decay_coefficient(decay_maps[image_id], quantile1, quantile2))
                    
                noise_masks[image_id] = noise_mask
                if self.noise_method == 'hard_mask':
                    value_maps[image_id][noise_mask > 0] = 0.001
                elif self.noise_method == 'soft_mask':
                    value_maps[image_id] *= decay_coefficient ** np.ceil(np.sqrt(head_num / 2))
                else:
                    raise ValueError('Invalid noise_method: {}'.format(self.noise_method))

                if save:
                    tensor_imsave(noise_mask, f'{image_id}_noise_mask.png', self.save_dir, norm=False)
                    tensor_imsave(value_maps[image_id], f'{image_id}_decayed_value_map.png',
                                    self.save_dir, norm=True)

        if self.use_ms_enh:
            # 根据注意力值进行显著性度量，得到显著性区域掩膜
            saliency_mask = self.get_ostu_mask(value_maps[image_ids[main_src_idx]], offset_ratio=0.3)
            new_value_map = self.enhance_significant_features(value_maps[image_ids[main_src_idx]], saliency_mask)
            value_maps[image_ids[main_src_idx]] = new_value_map
            if save:
                tensor_imsave(saliency_mask, f'main_saliency_mask.png', self.save_dir, norm=True)
                tensor_imsave(new_value_map, f'enhanced_value_map.png', self.save_dir, norm=True)

        # 对 value_maps 进行归一化得到融合权重
        weight_maps = {}

        example_tensor = next(iter(value_maps.values()))
        denominator = torch.zeros_like(example_tensor, dtype=torch.float32)

        for image_id in image_ids:
            denominator += torch.exp(value_maps[image_id])

        for image_id in image_ids:
            numerator = torch.exp(value_maps[image_id])
            weight_maps[image_id] = numerator / denominator

            if self.use_nah:
                if self.noise_method == 'hard_mask':
                    weight_maps[image_id][noise_masks[image_id] > 0] = 0.01

            if save:
                tensor_imsave(weight_maps[image_id], f'{image_id}_weight_map.png', self.save_dir, norm=False)

        tensors = [weight_maps[key] for key in weight_maps.keys()]
        stacked_weight_maps = torch.cat(tensors, dim=0).unsqueeze(1)
        return stacked_weight_maps

    def compute_guided_weight_maps(self, base_layers: torch.Tensor, detail_layers: torch.Tensor,
                                   weight_maps: torch.Tensor, save):
        guided_blur_r1 = GuidedBlur(kernel_size=self.r_1, eps=self.eps_1)
        guided_blur_r2 = GuidedBlur(kernel_size=self.r_2, eps=self.eps_2)

        weights_base = guided_blur_r1(base_layers, weight_maps)
        weights_detail = guided_blur_r2(base_layers + detail_layers, weight_maps)

        if save:
            for b in range(weights_base.shape[0]):
                tensor_imsave(weights_base[b], f'base_guided_weight_img{b}.png', self.save_dir, norm=False)
                tensor_imsave(weights_detail[b], f'detail_guided_weight_img{b}.png', self.save_dir, norm=False)

        return weights_base, weights_detail

    def forward(self, images: torch.Tensor, main_src_idx=0, vis_src_idx=0,
                norm_mode: str = None,
                save: bool = False, visualize: bool = False):
        batch_size, _, height, width = images.shape
        images = self.tensor_imnorm(images, norm_mode)

        if_use_bm3d = False
        if if_use_bm3d:
            images = self.bm3d_denoise(images)

        if save:
            for k in range(batch_size):
                tensor_imsave(images[k], f'img{k}.png', self.save_dir, norm=False)

        # 根据原始图像计算注意力图
        st1 = time.time()
        attention_maps = self.compute_attention_maps(images, vis_src_idx, save)
        self.runtime['compute_attention_maps'] = time.time() - st1

        # 根据注意力图计算权重图像
        st2 = time.time()
        weight_maps = self.compute_weight_maps(attention_maps, main_src_idx, vis_src_idx, save)
        self.runtime['compute_weight_maps'] = time.time() - st2

        # 将图像分解为基本层和细节层
        st3 = time.time()
        base_layers, detail_layers = self.decompose_image(images, method='avg_pool', save=save)

        # 使用分解图像对权重图像进行引导滤波
        weights_base, weights_detail = self.compute_guided_weight_maps(
            base_layers, detail_layers, weight_maps, save)

        # 融合基础层和细节层
        fused_base = torch.zeros_like(base_layers[0])
        fused_detail = torch.zeros_like(detail_layers[0])

        for b in range(batch_size):
            fused_base += base_layers[b] * weights_base[b]  # 权重形状为 (1, H, W)，自动广播到 (C, H, W)
            fused_detail += detail_layers[b] * weights_detail[b]

        fused_image = fused_base + fused_detail
        fused_image = torch.clamp(fused_image, 0, 1)  # 限制像素值在[0, 1]范围内
        self.runtime['guide filter fusion'] = time.time() - st3

        if visualize:
            # 创建子图布局
            fig, axes = plt.subplots(1, batch_size + 1, figsize=(15, 5))  # 1 行，N+1 列

            # 显示待融合的图像
            for i in range(batch_size):
                np_image = img_tensor2np(images[i])
                axes[i].imshow(np_image)
                axes[i].set_title(f"Image {i + 1}")
                axes[i].axis("off")  # 关闭坐标轴

            # 显示融合结果
            np_image = img_tensor2np(fused_image)
            axes[-1].imshow(np_image)
            axes[-1].set_title("Fused Image")
            axes[-1].axis("off")  # 关闭坐标轴

            # 调整子图间距
            plt.tight_layout()
            plt.show(block=True)

        if save:
            tensor_imsave(fused_image, f'APGFusion.png', self.save_dir, norm=False)

        return fused_image

    def apply_clahe(self, image: torch.Tensor, clahe: cv2.CLAHE, save: bool = False):
        numpy_image = img_tensor2np(image)
        uint8_image = im_dtype2uint8(numpy_image)

        if numpy_image.ndim == 2:
            enhanced_image = imnorm(clahe.apply(uint8_image), mode=None)
        elif numpy_image.ndim == 3:
            enhanced_image = []
            for k in range(image.shape[-1]):
                enhanced_image.append(imnorm(clahe.apply(image[:, :, k]), mode=None))
            enhanced_image = np.stack(enhanced_image, axis=-1)
        else:
            raise ValueError("Image must be 2D or 3D.")

        if save:
            imsave(enhanced_image, 'clahe_image.png', save_path=self.save_dir, norm=True)

        return enhanced_image

    @staticmethod
    def bm3d_denoise(image: torch.Tensor):
        np_image = tensor2np(image).transpose((0, 2, 3, 1))
        results = batch_bm3d(np_image, sigma_psd=0.1)
        tensor_results = torch.from_numpy(results.transpose((0, 3, 1, 2)))
        return tensor_results


def main():
    dataset_path = Path(os.path.join('..', '..', 'datasets', 'natural_images'))
    learner = NoiseLearner(dataset_path)
    # learner.compute()
    learner.visualize()


if __name__ == '__main__':
    main()
