import torch
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path
from utils.common import imsave
from typing import Union

__all__ = [
    'tensor2np',
    'list2tensor',
    'img_tensor2np',
    'tensor_imshow',
    'img_np2tensor',
    'tensor_imsave',
    'setup_seed'
]

eps = 1e-7


def tensor2np(tensor: torch.Tensor):
    return tensor.detach().cpu().numpy()


def list2tensor(images: list):
    images_np = np.array(images)
    images_tensor = torch.from_numpy(images_np).unsqueeze(1)
    return images_tensor


def img_tensor2np(img: torch.Tensor):
    if len(img.shape) == 2:
        # 二维图像张量
        np_img = tensor2np(img)
    elif len(img.shape) == 3:
        if img.shape[0] == 1:
            # 三维灰度图像张量
            np_img = tensor2np(img[0])
        elif img.shape[0] == 3:
            # 三维彩色图像张量
            np_img = tensor2np(img)
            np_img = np_img.transpose(1, 2, 0)
        else:
            raise ValueError
    else:
        raise ValueError('input torch requires 2d or 3d')

    return np_img


def tensor_imshow(img: torch.Tensor, cmap='gray', title: str = None):
    np_img = img_tensor2np(img)
    matplotlib.use('TkAgg')
    plt.figure()
    plt.imshow(np_img, cmap)
    plt.title(title)
    plt.show(block=True)


def img_np2tensor(image: np.ndarray):
    """灰度图像转为tensor张量"""
    if len(image.shape) == 2 or (len(image.shape) == 3 and image.shape[2] == 1):
        image = image.reshape(1, image.shape[0], image.shape[1])
        return torch.from_numpy(image)
    elif len(image.shape) == 3 and image.shape[2] == 3:
        return torch.from_numpy(image.transpose(2, 0, 1))
    else:
        raise ValueError("image shape error")


def tensor_imsave(img: torch.Tensor, save_name: str, save_path:Union[Path, str], cmap_name=None, norm=False):
    """
    保存二维或三维张量图像
    :param img: 张量图像
    :param save_name: 保存名称
    :param save_path: 保存路径
    :param cmap_name: 伪彩色方案名称，例如'twilight_shifted'
    :param norm: 可选，是否归一化
    """
    np_img = img_tensor2np(img)
    imsave(np_img, save_name, save_path, cmap_name=cmap_name, norm=norm)


def setup_seed(seed=0, benchmark=False, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = deterministic
