import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision.transforms.functional import gaussian_blur
from kornia.filters import GuidedBlur, laplacian
from utils.tensor import img_tensor2np

__all__ = ['GuideFilterFuserTensor']


class GuideFilterFuserTensor:
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

    def fusion(self, images, visualize=False):
        batch_size, channels, height, width = images.shape

        # 基础层（均匀滤波）
        base_layers = F.avg_pool2d(
            images, kernel_size=self.average_filter_size, stride=1,
            padding=self.average_filter_size // 2, count_include_pad=False
        )

        # 细节层
        detail_layers = images - base_layers

        # 显著性图
        saliencies = torch.sum(
            gaussian_blur(
                torch.abs(laplacian(images,
                                    kernel_size=3,
                                    border_type="reflect")),
                kernel_size=[3, 3],
                sigma=[self.sigma_r]
            ),
            dim=1,
            keepdim=True
        )

        # 引导滤波
        guided_blur_r1 = GuidedBlur(kernel_size=self.r_1, eps=self.eps_1)
        guided_blur_r2 = GuidedBlur(kernel_size=self.r_2, eps=self.eps_2)

        # 应用引导滤波
        weights_r1 = guided_blur_r1(images, saliencies).clip(min=0)
        weights_r2 = guided_blur_r2(images, saliencies).clip(min=0)

        # 归一化权重
        sum_weights_r1 = torch.sum(weights_r1, dim=0, keepdim=True)
        sum_weights_r2 = torch.sum(weights_r2, dim=0, keepdim=True)

        weights_base = weights_r1 / (sum_weights_r1 + 1e-8)
        weights_detail = weights_r2 / (sum_weights_r2 + 1e-8)

        # 融合基础层和细节层
        fused_base = torch.zeros_like(base_layers[0])
        fused_detail = torch.zeros_like(detail_layers[0])

        for b in range(batch_size):
            fused_base += base_layers[b] * weights_base[b]  # 权重形状为 (1, H, W)，自动广播到 (C, H, W)
            fused_detail += detail_layers[b] * weights_detail[b]

        fused_image = fused_base + fused_detail
        fused_image = torch.clamp(fused_image, 0, 1)  # 限制像素值在[0, 1]范围内

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

        return fused_image

