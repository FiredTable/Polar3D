import numpy as np
import cv2
import bm3d
from tqdm import tqdm
import skimage
from .common import imnorm, im_dtype2uint8

__all__ = [
    'detect_by_adaptive_threshold',
    'inpaint_by_diffusion',
    'apply_clahe',
    'batch_bm3d'
]


def detect_by_adaptive_threshold(img: np.ndarray, min_area=None, max_area=None, block_size=129, c=-60):
    """
    采用自适应阈值分割方法检测异常像素
    :param img: image, dtype=np.ndarray
    :param min_area: 感兴趣的最小面积范围，None 则表示不设置
    :param max_area: 感兴趣的最大面积范围，None 则表示不设置
    :param block_size: 邻域大小为 blockSize×blockSize 的奇数块
    :param c: 从加权和中减去的常数值，用于较亮的异常点检测
    """
    # 使用自适应阈值处理获取高光掩码
    uint8_img = (255 * img).astype(np.uint8)

    if img.ndim == 2:
        detect_mask = cv2.adaptiveThreshold(
            uint8_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, c)
    elif img.ndim == 3:
        detect_mask = []
        for k in range(img.shape[-1]):
            detect_mask.append(
                cv2.adaptiveThreshold(uint8_img[:, :, k], 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, block_size, c))
        detect_mask = np.stack(detect_mask, axis=-1)
        detect_mask = np.max(detect_mask, axis=-1)
    else:
        raise ValueError

    detect_mask = detect_mask.astype(bool)

    if min_area is not None:
        # 对感兴趣区域的最小面积进行限制
        detect_mask = skimage.morphology.remove_small_objects(detect_mask, min_area)

    if max_area is not None:
        # 对感兴趣区域的最大面积进行限制
        tmp_mask = skimage.morphology.remove_small_objects(detect_mask, max_area)
        detect_mask = np.logical_xor(detect_mask, tmp_mask)

    return detect_mask


def inpaint_by_diffusion(img, mask):
    """采用扩散方法对图像中异常点进行消除"""
    if mask.dtype == bool:
        mask = (255 * mask).astype(np.uint8)

    if img.dtype != np.uint8:
        norm_img = imnorm(img, 'min-max')
        uint8_img = (255 * norm_img).astype(np.uint8)
    else:
        uint8_img = img.copy()

    inpaint_img = cv2.inpaint(uint8_img, mask, 5, cv2.INPAINT_TELEA)

    min_val = np.nanmin(img)
    max_val = np.nanmax(img)
    inpaint_img = imnorm(inpaint_img, mode=None) * (max_val - min_val) + min_val

    return inpaint_img


def apply_clahe(image: np.ndarray, clahe: cv2.CLAHE):
    """
    对比度限制自适应直方图均衡化算法
    :param image: float[0,1]
    :param clahe: cv2.CLAHE
    :return: enhanced_image : float[0,1]
    """
    image = im_dtype2uint8(image)

    if image.ndim == 2:
        enhanced_image = clahe.apply(image).astype(np.float32) / 255
    elif image.ndim == 3:
        enhanced_image = []
        for k in range(image.shape[-1]):
            enhanced_image.append(clahe.apply(image[:, :, k]).astype(np.float32) / 255)
        enhanced_image = np.stack(enhanced_image, axis=-1)
    else:
        raise ValueError

    return enhanced_image


def batch_bm3d(img: np.ndarray, sigma_psd=0.1):
    """batch bm3d denoise [B,H,W,C]"""
    results = []
    batch_size, height, width, channel = img.shape
    for b in tqdm(range(batch_size)):
        for c in range(channel):
            results.append(bm3d.bm3d(img[b, :, :, c], sigma_psd=sigma_psd))
    results = np.stack(results, axis=0).reshape(batch_size, height, width, channel)
    return results
