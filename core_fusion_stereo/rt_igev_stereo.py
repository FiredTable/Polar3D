import torch.nn.functional as F
import torch.nn as nn
from core_fusion.model import APFNet
from core_stereo_rt.rt_igev_stereo import IGEVStereo

__all__ = [
    'PolarIGEVStereo'
]
import torch


def rgb_to_ycbcr(images: torch.Tensor) -> torch.Tensor:
    """
    将 RGB 图像转换为 YCbCr 格式
    输入: 形状为 [B, 3, H, W] 的 RGB 图像张量，值范围 [0, 1]
    输出: 形状为 [B, 3, H, W] 的 YCbCr 图像张量
    """
    if images.size(1) != 3:
        raise ValueError("输入图像必须有 3 个通道 (RGB)")
    
    # 分离 RGB 通道
    r = images[:, 0, :, :]
    g = images[:, 1, :, :]
    b = images[:, 2, :, :]
    
    # 转换矩阵 (ITU-R BT.601 标准)
    # Y 分量
    y = 0.299 * r + 0.587 * g + 0.114 * b
    
    # Cb 分量
    cb = -0.1687 * r - 0.3313 * g + 0.5 * b + 0.5
    
    # Cr 分量
    cr = 0.5 * r - 0.4187 * g - 0.0813 * b + 0.5
    
    # 组合通道并返回
    return torch.stack([y, cb, cr], dim=1)


def ycbcr_to_rgb(images: torch.Tensor) -> torch.Tensor:
    y = images[:, 0, :, :]
    cb = images[:, 1, :, :] - 0.5
    cr = images[:, 2, :, :] - 0.5
    
    r = y + 1.402 * cr
    g = y - 0.3441 * cb - 0.7141 * cr
    b = y + 1.772 * cb
    
    return torch.stack([r, g, b], dim=1).clamp(0, 1)


class PFIGEVStereo(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.fusion_model = APFNet()
        self.stereo_model = IGEVStereo(args)

    def forward(self, left_img1, right_img1, left_img2, right_img2, iters=12, flow_init=None, test_mode=False):
        """ Estimate disparity between pair of frames """
        # the range of input for APFNet is [0, 1]
        left_img1 = (left_img1 / 255.0).contiguous()
        left_img2 = (left_img2 / 255.0).contiguous()
        right_img1 = (right_img1 / 255.0).contiguous()
        right_img2 = (right_img2 / 255.0).contiguous()

        # input for APFNet is in grayscale
        is_color = False
        if left_img1.shape[1] == 3:  # the inputs is [B, 3, H, W]
            is_color = True
            left_ycbcr1 = rgb_to_ycbcr(left_img1)
            right_ycbcr1 = rgb_to_ycbcr(right_img1)

            # the range of output for APFNet is [0, 1] and in grayscale
            left_img = self.fusion_model(left_ycbcr1[:, 0:1, :, :], left_img2[:, 0:1, :, :])
            right_img = self.fusion_model(right_ycbcr1[:, 0:1, :, :], right_img2[:, 0:1, :, :])
        else:
            left_img = self.fusion_model(left_img1, left_img2)
            right_img = self.fusion_model(right_img1, right_img2)

        # input fro IGEV is in RGB
        if is_color:
            left_ycbcr = torch.cat((left_img, left_ycbcr1[:, 1:3, :, :]), dim=1)
            left_img = ycbcr_to_rgb(left_ycbcr)
            right_ycbcr = torch.cat((right_img, right_ycbcr1[:, 1:3, :, :]), dim=1)
            right_img = ycbcr_to_rgb(right_ycbcr)
        else:
            left_img = torch.cat((left_img, left_img, left_img), dim=1)
            right_img = torch.cat((right_img, right_img, right_img), dim=1)

        # the range of input for IGEV is [0, 255]
        left_img = left_img * 255.0
        right_img = right_img * 255.0

        if test_mode:
            disp_up = self.stereo_model(left_img, right_img, iters, flow_init, test_mode)
            return disp_up, left_img, right_img
        
        # the range of output for IGEV is [0, args.max_disp]
        init_disp, disp_preds = self.stereo_model(left_img, right_img, iters, flow_init, test_mode)
        return init_disp, disp_preds, left_img, right_img

        