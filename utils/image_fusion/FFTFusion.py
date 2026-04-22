import cv2
import numpy as np
import matplotlib.pyplot as plt

from scipy.fftpack import fft2, ifft2, fftshift, ifftshift

__all__ = ['FFTFuser']


class FFTFuser:
    def __init__(self, sigma=None):
        self.sigma = sigma

    def fusion(self, images:np.ndarray):
        batch_size, height, width, channels = images.shape

        if self.sigma is not None:
            mask = self.gaussian_low_pass_filter((height, width), sigma=self.sigma)
        else:
            mask = np.ones((height, width))

        freq_ims = np.zeros((batch_size, height, width, channels), dtype=np.complex128)
        for b in range(batch_size):
            for c in range(channels):
                freq_channel = fftshift(fft2(images[b, :, :, c]))
                filtered = freq_channel * mask
                freq_ims[b, :, :, c] = filtered

        fused_freq = np.sum(freq_ims, axis=0)

        fused_image = np.zeros((height, width, channels), dtype=np.float32)
        for c in range(channels):
            fused_image[:, :, c] = np.abs(ifft2(ifftshift(fused_freq[:, :, c])))

        fused_image = cv2.normalize(fused_image, None, 1, 0, cv2.NORM_MINMAX)
        return fused_image

    def visualize(self, freq_ims, images, fused_image, fused_freq):
        magnitude_spectrum_images = self.calc_magnitude_spectrum(freq_ims)

        batch_size = images.shape[0]
        fig, axes = plt.subplots(2, batch_size + 1)
        for b in range(batch_size):
            ax_top = axes[0, b]
            ax_top.imshow(images[b])

            ax_bottom = axes[1, b]
            ax_bottom.imshow(np.mean(magnitude_spectrum_images[b, :, :, :], axis=-1), cmap='gray')

        ax_top = axes[0, batch_size]
        ax_top.imshow(fused_image)
        ax_bottom = axes[1, batch_size]
        ax_bottom.imshow(np.log(np.abs(fused_freq)), cmap='gray')

        plt.tight_layout()
        plt.show(block=True)

        return fused_image

    @staticmethod
    def gaussian_low_pass_filter(shape, sigma):
        """创建高斯低通滤波器"""
        rows, cols = shape
        crow, ccol = rows // 2, cols // 2
        x, y = np.ogrid[:rows, :cols]
        distance_squared = (x - crow) ** 2 + (y - ccol) ** 2
        mask = np.exp(-distance_squared / (2 * (sigma ** 2)))
        return mask

    @staticmethod
    def calc_magnitude_spectrum(freq_ims: np.ndarray):
        batch_size, height, width, channels = freq_ims.shape
        magnitude_spectrum_images = np.zeros((batch_size, height, width, channels), dtype=np.float32)
        for b in range(batch_size):
            for c in range(channels):
                magnitude_spectrum_images[b, :, :, c] = np.log(np.abs(freq_ims[b, :, :, c]))
        return magnitude_spectrum_images
