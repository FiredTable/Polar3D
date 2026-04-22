import cv2
import numpy as np

__all__ = ['yang2010real']


def yang2010real(img):
    assert img.dtype == np.float64 or img.dtype == np.float32, 'Input img is not type double.'
    assert img.min() >= 0 and img.max() <= 1, 'Input img is not within [0, 1] range.'
    n_row, n_col, n_ch = img.shape
    assert n_row > 1 and n_col > 1, 'Input img has a singleton dimension.'
    assert n_ch == 3, 'Input img is not a RGB image.'

    total = np.sum(img, axis=2)

    sigma = img / total[:, :, None]
    sigma[np.isnan(sigma)] = 0

    sigmaMin = np.min(sigma, axis=2)
    sigmaMax = np.max(sigma, axis=2)

    lambda_ = np.ones_like(img) / 3
    lambda_ = (sigma - sigmaMin[:, :, None]) / (3 * (lambda_ - sigmaMin[:, :, None]))
    lambda_[np.isnan(lambda_)] = 1 / 3

    lambdaMax = np.max(lambda_, axis=2)

    SIGMAS = int(0.25 * min(n_row, n_col))  # 双边滤波器的空间域半径
    SIGMAR = 0.04 * 255  # 双边滤波器的颜色域标准差，转换为8位图像的范围
    THR = 0.03

    iter_num = 0
    while True:
        sigmaMaxF = cv2.ximgproc.jointBilateralFilter(
            (lambdaMax * 255).astype(np.uint8),  # 指导图像
            (sigmaMax * 255).astype(np.uint8),  # 待滤波的图像
            d=SIGMAS,
            sigmaColor=SIGMAR,
            sigmaSpace=SIGMAS
        )
        sigmaMaxF = sigmaMaxF.astype(np.float64) / 255

        if np.count_nonzero(sigmaMaxF - sigmaMax > THR) == 0:
            break
        sigmaMax = np.maximum(sigmaMax, sigmaMaxF)

        iter_num += 1
        if iter_num > 10:
            break

    img_max = np.max(img, axis=2)

    den = (1 - 3 * sigmaMax)
    img_s = (img_max - sigmaMax * total) / den
    img_s[den == 0] = np.max(img_s[den != 0])

    img_d = np.clip(img - img_s[:, :, None], 0, 1)

    return img_d

