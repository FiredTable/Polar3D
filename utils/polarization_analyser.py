import cmath
import re
import glob
import os
import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage.io
import tkinter as tk
from PIL import ImageTk
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from typing import Optional
from natsort import natsorted
from utils.common import imsave, imnorm, resize_pil_image, np2pil, cv2_imread

__all__ = [
    'PolarDatasetManager',
    'PolarizationAnalyser',
    'ImageCropper',
    'ClickToGetPixelPosition',
    'MCPolarDatasetManager',
    'MCPolarizationAnalyser',
    'pdm_fp2lp',
    'stokes2rgb',
    'ReflectionPolarAnalyser']

eps = np.finfo(np.float64).eps


class PolarDatasetManager:
    """偏振图像数据管理器"""
    DTYPE_DOT_FP = 0
    DTYPE_DOT_LP = 1
    DTYPE_DOF_LP = 2

    def __init__(self, dataset_type, save_path: Optional[Path] = None):
        self.dataset_type = dataset_type
        self.save_path = save_path
        self.polarization_images = None
        self.rgb_image = None

        if dataset_type == PolarDatasetManager.DTYPE_DOT_LP or dataset_type == PolarDatasetManager.DTYPE_DOF_LP:
            self.lp_angles = None
        if dataset_type == PolarDatasetManager.DTYPE_DOT_FP:
            self.lp_angles = None
            self.qwp_angles = None

    def import_dof_polarization_image(self, imfile):
        file_extension = os.path.splitext(imfile)[1]
        assert file_extension.lower() in ('.jpg', '.bmp', '.png')

        if os.path.exists(imfile):
            dof_polarization_image = cv2_imread(imfile)
            if dof_polarization_image.ndim == 3:
                dof_polarization_image = cv2.cvtColor(dof_polarization_image, cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError('image path does not exist')

        self.polarization_images, self.lp_angles = PolarDatasetManager.demosaicing(dof_polarization_image)

    @staticmethod
    def demosaicing(dofp_polarization_image: np.ndarray):
        height = dofp_polarization_image.shape[0]
        width = dofp_polarization_image.shape[1]
        polarization_images = np.zeros((height, width, 4), dtype=np.float64)
        lp_angles = np.array([0, 45, 90, 135]) * np.pi / 180

        code_bg = getattr(cv2, f"COLOR_BayerBG2BGR")
        code_gr = getattr(cv2, f"COLOR_BayerGR2BGR")
        img_debayer_bg = cv2.cvtColor(dofp_polarization_image, code_bg)
        img_debayer_gr = cv2.cvtColor(dofp_polarization_image, code_gr)
        img_000, _, img_090 = cv2.split(img_debayer_bg)
        img_045, _, img_135 = cv2.split(img_debayer_gr)

        polarization_images[:, :, 0] = img_000.astype(float) / 255
        polarization_images[:, :, 1] = img_045.astype(float) / 255
        polarization_images[:, :, 2] = img_090.astype(float) / 255
        polarization_images[:, :, 3] = img_135.astype(float) / 255

        return polarization_images, lp_angles

    def import_dot_full_polarization_images(self, imdir):
        extensions = ['*.bmp', '*.png']
        polar_images_list = [file for ext in extensions for file in glob.glob(os.path.join(imdir, ext))]

        if not polar_images_list:
            raise ValueError('图像数据路径为空，请检查图像路径或图像格式！')

        # 导入偏振图像数据集
        polarization_images = []
        qwp_angles = []
        lp_angles = []
        rgb_image = None

        for k in range(len(polar_images_list)):
            im = skimage.io.imread(polar_images_list[k])
            gray_im = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY) if len(im.shape) == 3 else im

            # 尝试根据图像名称得到波片方位角度
            img_name = os.path.basename(polar_images_list[k])
            name_without_ext = os.path.splitext(img_name)[0]
            try:
                numbers = re.findall(r'\d+', name_without_ext)
                numbers = [int(num) for num in numbers]
                lp_angles.append(np.deg2rad(numbers[0]))
                qwp_angles.append(np.deg2rad(numbers[1]))
                polarization_images.append(gray_im.astype(float) / 255)
            except IndexError:
                if name_without_ext == 'rgb':
                    rgb_image = skimage.io.imread(polar_images_list[k]).astype(float) / 255
                print(f"Warning: Skipping '{os.path.basename(polar_images_list[k])}'")

        self.lp_angles = np.array(lp_angles)
        self.qwp_angles = np.array(qwp_angles)
        self.polarization_images = np.stack(polarization_images, axis=-1)
        self.rgb_image = rgb_image

    def import_dot_linear_polarization_images(self, imdir):
        assert self.dataset_type == PolarDatasetManager.DTYPE_DOT_LP
        print('import linear polarization images...')

        extensions = ['*.bmp', '*.png']
        polar_images_list = natsorted([file for ext in extensions for file in glob.glob(os.path.join(imdir, ext))])

        if not polar_images_list:
            raise ValueError('图像数据路径为空，请检查图像路径或图像格式！')

        # 导入偏振图像数据集
        lp_angles = []
        polarization_images = []

        for k in range(len(polar_images_list)):
            img = cv2_imread(polar_images_list[k])
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            polarization_images.append(img/255.)

            # 根据图像名称得到偏振方位角度
            image_name = os.path.basename(polar_images_list[k])
            number_str = ''.join(filter(str.isdigit, image_name))
            lp_angles.append(int(number_str) * np.pi / 180)

        print('数据集中共有 ', len(polar_images_list), ' 张图像')

        self.polarization_images = np.array(polarization_images).transpose(1, 2, 0)
        self.lp_angles = np.array(lp_angles)

        sorted_indices = np.argsort(lp_angles)
        self.polarization_images = self.polarization_images[:, :, sorted_indices]
        self.lp_angles = self.lp_angles[sorted_indices]

    def crop_polarization_images(self, crop_coordinates=None, aspect_ratio=None):
        image = np.mean(self.polarization_images, axis=2)
        cropper = ImageCropper(image, crop_coordinates, aspect_ratio)
        crop_coordinates = cropper.crop_coordinates

        polarization_images = []
        for k in range(self.polarization_images.shape[2]):
            polarization_images.append(ImageCropper.imcrop(
                self.polarization_images[:, :, k], crop_coordinates, aspect_ratio))

        self.polarization_images = np.array(polarization_images).transpose(1, 2, 0)

        if self.rgb_image is not None:
            self.rgb_image = ImageCropper.imcrop(self.rgb_image, crop_coordinates, aspect_ratio)


class PolarizationAnalyser:
    """偏振图像分析工具"""
    def __init__(self, polar_dataset: PolarDatasetManager, resize_ratio=1.):
        """
        偏振分析，计算图像斯托克斯参量，计算偏振特征参量
        :param polar_dataset: 偏振图像数据管理器
        :param resize_ratio: 图像缩放比
        """
        self.polar_dataset = polar_dataset
        self.resize_ratio = resize_ratio

        self.stokes: Optional[np.ndarray] = None
        self.mask: Optional[np.ndarray] = None

        self.rho: Optional[np.ndarray] = None
        self.phi: Optional[np.ndarray] = None
        self.chi: Optional[np.ndarray] = None
        self.iun: Optional[np.ndarray] = None
        self.imin: Optional[np.ndarray] = None
        self.imax: Optional[np.ndarray] = None

        self.avg_images: Optional[np.ndarray] = None
        self.min_images: Optional[np.ndarray] = None
        self.max_images: Optional[np.ndarray] = None

    def calc_stokes(self, highlight_removal=False, save=False):
        images = self.polar_dataset.polarization_images

        start_time = time.time()
        if self.polar_dataset.dataset_type == PolarDatasetManager.DTYPE_DOT_FP:
            lp_angles = self.polar_dataset.lp_angles
            qwp_angles = self.polar_dataset.qwp_angles
            assert len(qwp_angles.shape) == 1
            stokes, mask = self.calc_full_stokes(images, lp_angles, qwp_angles, highlight_removal)
        elif (self.polar_dataset.dataset_type == PolarDatasetManager.DTYPE_DOT_LP
              or self.polar_dataset.dataset_type == PolarDatasetManager.DTYPE_DOF_LP):
            lp_angles = self.polar_dataset.lp_angles
            assert len(lp_angles.shape) == 1
            stokes, mask = self.calc_linear_stokes(images, lp_angles, highlight_removal)
        else:
            raise ValueError
        # print(f'highlight removal is {highlight_removal}, calc stokes cost {time.time() - start_time:.4f} s')

        if self.resize_ratio != 1.:
            stokes = self._reduce_resolution(stokes)
            images = self._reduce_resolution(images)
            mask = self._reduce_resolution(mask)

        self.stokes = stokes
        self.polar_dataset.polarization_images = images
        self.mask = mask

        if save:
            imsave(stokes[:, :, 0], f's0.png', self.polar_dataset.save_path, norm=True)
            imsave(stokes[:, :, 1], f's1.png', self.polar_dataset.save_path, norm=True)
            imsave(stokes[:, :, 2], f's2.png', self.polar_dataset.save_path, norm=True)
            if self.polar_dataset.dataset_type == PolarDatasetManager.DTYPE_DOT_FP:
                imsave(stokes[:, :, 3], f's3.png',
                       self.polar_dataset.save_path, norm=True)
            imsave(mask, f'mask.png', self.polar_dataset.save_path, norm=True)

    def calc_polar_features(self, s0_threshold=0., save=False):
        if self.stokes is None:
            self.calc_stokes()

        self.rho = self.stokes2rho(self.stokes, s0_threshold)
        self.phi = self.stokes2phi(self.stokes, s0_threshold)
        if self.polar_dataset.dataset_type == PolarDatasetManager.DTYPE_DOT_FP:
            self.chi = self.stokes2chi(self.stokes)

        self.iun = self.stokes2iun(self.stokes)
        self.imin = self.stokes2imin(self.stokes)
        self.imax = self.stokes2imax(self.stokes)

        if save:
            imsave(self.rho, 'rho.png', self.polar_dataset.save_path, norm=True)
            imsave(self.phi, 'phi.png', self.polar_dataset.save_path, cmap_name='twilight_shifted', norm=True)
            if self.chi is not None:
                imsave(self.chi, 'chi.png', self.polar_dataset.save_path, cmap_name='twilight_shifted', norm=True)
            imsave(self.iun, 'iun.png', self.polar_dataset.save_path, norm=True)
            imsave(self.imin, 'imin.png', self.polar_dataset.save_path, norm=True)
            imsave(self.imax, 'imax.png', self.polar_dataset.save_path, norm=True)

    def calc_inten_features(self, save=False):
        self.avg_images = self.images2iun(self.polar_dataset.polarization_images)
        self.min_images = self.images2imin(self.polar_dataset.polarization_images)
        self.max_images = self.images2imax(self.polar_dataset.polarization_images)

        if save:
            imsave(self.avg_images, 'avg_images.png', self.polar_dataset.save_path, norm=True)
            imsave(self.min_images, 'min_images.png', self.polar_dataset.save_path, norm=True)
            imsave(self.max_images, 'max_images.png', self.polar_dataset.save_path, norm=True)

    @staticmethod
    def calc_full_stokes(images, lp_angles, qwp_angles, highlight_removal):
        mask = np.ones((images.shape[0], images.shape[1]))

        assert len(lp_angles) == len(qwp_angles)

        values = images.reshape(images.shape[0] * images.shape[1], images.shape[2])
        stokes = PolarizationAnalyser.solve_full_stokes_contradictory_equation(lp_angles, qwp_angles, values)

        s0 = stokes[:, 0]
        s1 = stokes[:, 1]
        s2 = stokes[:, 2]
        s3 = stokes[:, 3]

        # 针对高光像素进行特殊处理
        if highlight_removal:
            tolerance1 = 0.04
            tolerance2 = 0.02
            values_max = np.max(values, axis=1)
            special_pixel_index = np.where(values_max >= 1 - tolerance1)[0]

            dark_is_also_invalid = False
            if dark_is_also_invalid:
                values_min = np.min(values, axis=1)
                special_pixel_index = np.where((values_max >= 1 - tolerance1) | (values_min <= 0))[0]

            for k in tqdm(range(len(special_pixel_index))):
                values_pt = values[special_pixel_index[k], :]
                valid_index = np.where((values_pt <= 1 - tolerance2) & (values_pt >= 0))[0]
                if len(values_pt) / 3 > len(valid_index) > 2:
                    valid_index = PolarizationAnalyser._expand_valid_index(valid_index)

                if len(valid_index) >= 4:
                    valid_values = values_pt[valid_index]
                    valid_lp_angles = lp_angles[valid_index]
                    valid_qwp_angles = qwp_angles[valid_index]

                    valid_stokes = PolarizationAnalyser.solve_full_stokes_contradictory_equation(
                        valid_lp_angles, valid_qwp_angles, valid_values)

                    s0[special_pixel_index[k]] = valid_stokes[0]
                    s1[special_pixel_index[k]] = valid_stokes[1]
                    s2[special_pixel_index[k]] = valid_stokes[2]
                    s3[special_pixel_index[k]] = valid_stokes[3]
                else:
                    row = special_pixel_index[k] // images.shape[1]
                    col = special_pixel_index[k] % images.shape[1]
                    mask[row, col] = 0.

        s0[s0 <= 0] = eps
        s1[np.abs(s1) >= np.abs(s0)] = s0[np.abs(s1) >= np.abs(s0)]
        s2[np.abs(s2) >= np.abs(s0)] = s0[np.abs(s2) >= np.abs(s0)]
        s3[np.abs(s3) >= np.abs(s0)] = s0[np.abs(s3) >= np.abs(s0)]

        stokes = stokes.reshape((images.shape[0], images.shape[1], 4))
        return stokes, mask

    @staticmethod
    def solve_full_stokes_contradictory_equation(lp_angles, qwp_angles, values):
        """
        计算全斯托克斯矛盾方程组
        :param lp_angles: 线偏振片角度数组
        :param qwp_angles: 四分之一波片角度数组
        :param values: (width_height * polar_chans)的二维数组 或 一维数组
        :return: stokes: (4 * width_height) 的值数组
        """
        if values.ndim == 2:
            # 如果 values 是 (width_height * polar_chans) 的二维值数组，转化为 (polar_chans * width_height) 的二维值数组
            values = values.transpose()

        ones_column = np.ones((len(qwp_angles), 1))
        cos_cos_column = (np.cos(2 * qwp_angles) * np.cos(2 * lp_angles - 2 * qwp_angles)).reshape(-1, 1)
        sin_cos_column = (np.sin(2 * qwp_angles) * np.cos(2 * lp_angles - 2 * qwp_angles)).reshape(-1, 1)
        sin_column = (np.sin(2 * lp_angles - 2 * qwp_angles)).reshape(-1, 1)
        A = 0.5 * np.hstack((ones_column, cos_cos_column, sin_cos_column, sin_column))
        x = np.linalg.pinv(A, rcond=1e-15) @ values
        stokes = x.transpose()
        return stokes

    @staticmethod
    def calc_linear_stokes(images, lp_angles, highlight_removal=False):
        mask = np.ones((images.shape[0], images.shape[1]))

        values = images.reshape(images.shape[0] * images.shape[1], images.shape[2])
        stokes = PolarizationAnalyser.solve_linear_stokes_contradictory_equation(lp_angles, values)

        s0 = stokes[:, 0]
        s1 = stokes[:, 1]
        s2 = stokes[:, 2]

        # 针对高光像素进行特殊处理
        if highlight_removal is True:
            tolerance1 = 0.04
            tolerance2 = 0.02
            values_max = np.max(values, axis=1)
            values_min = np.min(values, axis=1)
            special_pixel_index = np.where((values_max >= 1 - tolerance1) | (values_min <= 0))[0]

            for k in range(len(special_pixel_index)):
                values_pt = values[special_pixel_index[k], :]
                valid_index = np.where((values_pt <= 1 - tolerance2) & (values_pt >= 0))[0]
                if len(values_pt) / 3 > len(valid_index) > 2:
                    valid_index = PolarizationAnalyser._expand_valid_index(valid_index)

                if len(valid_index) >= 3:
                    valid_values = values_pt[valid_index]
                    valid_lp_angles = lp_angles[valid_index]

                    valid_stokes = PolarizationAnalyser.solve_linear_stokes_contradictory_equation(
                        valid_lp_angles, valid_values)

                    s0[special_pixel_index[k]] = valid_stokes[0]
                    s1[special_pixel_index[k]] = valid_stokes[1]
                    s2[special_pixel_index[k]] = valid_stokes[2]
                else:
                    row = special_pixel_index[k] // images.shape[1]
                    col = special_pixel_index[k] % images.shape[1]
                    mask[row, col] = 0.

        s0[s0 <= 0] = np.finfo(s0.dtype).eps
        s1[np.abs(s1) >= np.abs(s0)] = s0[np.abs(s1) >= np.abs(s0)]
        s2[np.abs(s2) >= np.abs(s0)] = s0[np.abs(s2) >= np.abs(s0)]

        stokes = stokes.reshape((images.shape[0], images.shape[1], 3))
        return stokes, mask

    @staticmethod
    def solve_linear_stokes_contradictory_equation(lp_angles, values):
        """
        求解线偏振矛盾方程组
        :param lp_angles: 线偏振片角度
        :param values: (width_height * polar_chans) 的值数组 或 一维数组
        :return: stokes: (3 * width_height) 的值数组
        """
        if values.ndim == 2:
            # 如果 values 是 (width_height * polar_chans) 的二维值数组，转化为 (polar_chans * width_height) 的二维值数组
            values = values.transpose()

        ones_column = np.ones((len(lp_angles), 1))
        cos_column = np.cos(2 * lp_angles).reshape(-1, 1)
        sin_column = np.sin(2 * lp_angles).reshape(-1, 1)
        A = 0.5 * np.hstack((ones_column, cos_column, sin_column))
        x = np.linalg.pinv(A, rcond=1e-15) @ values
        stokes = x.transpose()
        return stokes

    @staticmethod
    def _expand_valid_index(valid_index):
        """扩充有效值，认为有效值边缘处的无效值为有效的，例如[0,1,2,6,7] -> [0,1,2,3,5,6,7]"""
        continuous_index = np.arange(valid_index.min(), valid_index.max() + 1)
        missing_index = np.setdiff1d(continuous_index, valid_index)

        if missing_index.size == 0:
            # missing_index 是一个空的 np 数组
            return valid_index

        min_index = np.array([missing_index.min()])
        max_index = np.array([missing_index.max()])
        expanded_valid_index = np.sort(np.concatenate([valid_index, min_index, max_index]))
        return expanded_valid_index

    def _reduce_resolution(self, images: np.ndarray):
        f = self.resize_ratio
        resized_images = cv2.resize(images, dsize=None, fx=f, fy=f)
        return resized_images

    @staticmethod
    def stokes2rho(stokes, s0_threshold=0.):
        ndims = stokes.ndim

        if ndims == 2:
            rho = np.sqrt(np.sum(stokes[:, 1:] ** 2, axis=-1)) / stokes[:, 0].clip(min=eps)
            rho[stokes[:, 0] <= s0_threshold] = 0
        elif ndims == 3:
            rho = np.sqrt(np.sum(stokes[:, :, 1:] ** 2, axis=-1)) / stokes[:, :, 0].clip(min=eps)
            rho[stokes[:, :, 0] <= s0_threshold] = 0
        else:
            raise ValueError(
                "Unsupported dimensions for stokes array. Expected 2 or 3 dimensions, got {}.".format(ndims))

        rho[rho > 1] = 1
        return rho

    @staticmethod
    def stokes2phi(stokes, s0_threshold=0.):
        ndims = stokes.ndim

        if ndims == 2:
            phi = 0.5 * np.arctan2(stokes[:, 2], stokes[:, 1])
            phi[stokes[:, 0] <= s0_threshold] = 0
        elif ndims == 3:
            phi = 0.5 * np.arctan2(stokes[:, :, 2], stokes[:, :, 1])
            phi[stokes[:, :, 0] <= s0_threshold] = 0
        else:
            raise ValueError(
                "Unsupported dimensions for stokes array. Expected 2 or 3 dimensions, got {}.".format(ndims))

        return phi

    @staticmethod
    def stokes2phi_with_regularization(stokes, threshold=0.02):
        # 对s1和s2应用高斯滤波以减少噪声
        s1_smooth = gaussian_filter(stokes[:, :, 1], sigma=1)
        s2_smooth = gaussian_filter(stokes[:, :, 2], sigma=1)

        # 计算幅度A作为正则化项的一部分
        A = np.sqrt(s1_smooth ** 2 + s2_smooth ** 2)

        # 应用阈值处理，避免分母过小导致数值不稳定
        mask = A > threshold
        s1_thresholded = np.where(mask, s1_smooth, 0)
        s2_thresholded = np.where(mask, s2_smooth, 0)

        phi = 0.5 * np.arctan2(s1_thresholded, s2_thresholded)

        return phi

    @staticmethod
    def stokes2chi(stokes: np.ndarray) -> np.ndarray:
        # ellipticity angle ∈ [-pi/4, pi/4]
        chi = 0.5 * np.arctan2(stokes[:, :, 3], np.sqrt(stokes[:, :, 1] ** 2 + stokes[:, :, 2] ** 2))
        return chi

    @staticmethod
    def stokes2iun(stokes):
        ndims = stokes.ndim

        if ndims == 2:
            iun = stokes[:, 0] / 2
        elif ndims == 3:
            iun = stokes[:, :, 0] / 2
        else:
            raise ValueError(
                "Unsupported dimensions for stokes array. Expected 2 or 3 dimensions, got {}.".format(ndims))

        iun[iun > 1] = 1
        return iun

    @staticmethod
    def stokes2imin(stokes):
        imin = 0.5 * (stokes[:, :, 0] - np.sqrt(np.sum(stokes[:, :, 1:] ** 2, axis=-1)))
        imin[imin < 0] = 0
        return imin

    @staticmethod
    def stokes2imax(stokes):
        imax = 0.5 * (stokes[:, :, 0] + np.sqrt(np.sum(stokes[:, :, 1:] ** 2, axis=-1)))
        return imax

    @staticmethod
    def stokes2lp(stokes, lp_angle):
        s0 = stokes[:, :, 0]
        s1 = stokes[:, :, 1]
        s2 = stokes[:, :, 2]
        im_lp = 0.5 * (s0 + np.cos(2 * lp_angle) * s1 + np.sin(2 * lp_angle) * s2)
        return im_lp.clip(min=0, max=1)

    @staticmethod
    def images2iun(images):
        iun_ = np.mean(images, axis=2)
        return iun_

    @staticmethod
    def images2imin(images):
        imin_ = np.min(images, axis=2)
        imin_[imin_ > 1] = 1
        return imin_

    @staticmethod
    def images2imax(images):
        imax_ = np.max(images, axis=2)
        return imax_

    @staticmethod
    def dofp2iun(dofp: np.ndarray, resize_ratio: Optional[float] = None):
        polarization_images, lp_angles = PolarDatasetManager.demosaicing(dofp)
        if resize_ratio is not None:
            polarization_images = cv2.resize(polarization_images, dsize=None, fx=resize_ratio, fy=resize_ratio)
        s0 = np.sum(polarization_images, axis=2) / 2
        iun = s0 / 2
        iun[iun > 1] = 1
        return iun

    @staticmethod
    def dofp2rho(dofp: np.ndarray, resize_ratio: Optional[float] = None):
        polarization_images, lp_angles = PolarDatasetManager.demosaicing(dofp)
        if resize_ratio is not None:
            polarization_images = cv2.resize(polarization_images, dsize=None, fx=resize_ratio, fy=resize_ratio)
        s0 = np.sum(polarization_images, axis=2) / 2
        s1 = polarization_images[:, :, 0] - polarization_images[:, :, 2]
        s2 = polarization_images[:, :, 1] - polarization_images[:, :, 3]
        rho = np.sqrt(s1 ** 2 + s2 ** 2) / s0.clip(min=eps)
        rho[rho > 1] = 1
        return rho

    @staticmethod
    def dofp2phi(dofp: np.ndarray, s0_threshold=0., resize_ratio: Optional[float] = None):
        polarization_images, lp_angles = PolarDatasetManager.demosaicing(dofp)
        if resize_ratio is not None:
            polarization_images = cv2.resize(polarization_images, dsize=None, fx=resize_ratio, fy=resize_ratio)
        s0 = np.sum(polarization_images, axis=2) / 2
        s1 = polarization_images[:, :, 0] - polarization_images[:, :, 2]
        s2 = polarization_images[:, :, 1] - polarization_images[:, :, 3]
        phi = 0.5 * np.arctan2(s2, s1)
        phi[s0 <= s0_threshold] = 0
        return phi

    @staticmethod
    def dofp2imin(dofp: np.ndarray):
        polarization_images, lp_angles = PolarDatasetManager.demosaicing(dofp)
        s0 = np.sum(polarization_images, axis=2) / 2
        s1 = polarization_images[:, :, 0] - polarization_images[:, :, 2]
        s2 = polarization_images[:, :, 1] - polarization_images[:, :, 3]
        imin = 0.5 * (s0 - np.sqrt(s1 ** 2 + s2 ** 2))
        imin[imin < 0] = 0
        return imin

    def save_polar_images(self, save_path):
        imsave(self.stokes[:, :, 0], 'S0.png', save_path, norm=True)
        imsave(self.stokes[:, :, 1], 'S1.png', save_path, norm=True)
        imsave(self.stokes[:, :, 2], 'S2.png', save_path, norm=True)

        imsave(self.iun, 'iun.png', save_path)
        imsave(self.imin, 'imin.png', save_path)
        imsave(self.imax, 'imax.png', save_path)

        imsave(self.rho, 'rho.png', save_path)
        imsave(self.phi, 'phi.png', save_path, norm=True)

        if self.polar_dataset.dataset_type == PolarDatasetManager.DTYPE_DOT_FP:
            imsave(self.stokes[:, :, 3], 'S3.png', save_path, norm=True)
            imsave(self.chi, 'chi.png', save_path, norm=True)

    def click_show_polarization_curve(self):
        """点击图像中的一点，显示偏振曲线"""
        print('show_polarization_curve_at_a_pixel...')
        clicker = ClickToGetPixelPosition(np.mean(self.polar_dataset.polarization_images, axis=2))

        if self.polar_dataset.dataset_type == PolarDatasetManager.DTYPE_DOT_FP:
            self.plot_full_polarization_curve(clicker.coord_y, clicker.coord_x)
        elif (self.polar_dataset.dataset_type == PolarDatasetManager.DTYPE_DOT_LP or
              self.polar_dataset.dataset_type == PolarDatasetManager.DTYPE_DOF_LP):
            self.plot_linear_polarization_curve(clicker.coord_y, clicker.coord_x)
        else:
            raise ValueError('dataset type need to be expanded')

    def plot_full_polarization_curve(self, coord_y, coord_x):
        """绘制目标点的全偏振特征曲线"""
        values = self.polar_dataset.polarization_images[coord_y, coord_x, :]
        qwp_angles = self.polar_dataset.qwp_angles.copy()
        lp_angle = self.polar_dataset.lp_angles[0]

        plt.scatter(qwp_angles * 180 / np.pi, values, label='scatter')

        # 传统解析方法拟合曲线
        stokes = self.solve_full_stokes_contradictory_equation(lp_angle, qwp_angles, values)
        s0 = stokes[0]
        s1 = stokes[1]
        s2 = stokes[2]
        s3 = stokes[3]

        x = np.arange(qwp_angles.min(), qwp_angles.max(), 0.01)
        y1 = 0.5 * (s0 + np.cos(2 * x) * np.cos(2 * lp_angle - 2 * x) * s1
                    + np.sin(2 * x) * np.cos(2 * lp_angle - 2 * x) * s2
                    + np.sin(2 * lp_angle - 2 * x) * s3)
        plt.plot(x * 180 / np.pi, y1, label='traditional fitting')

        rho = np.sqrt(pow(s1, 2) + pow(s2, 2) + pow(s3, 2)) / s0
        print(f'传统方法曲线，对应的偏振度为{rho:.4f}')
        print(f'平均偏振强度值为 {np.mean(values):.4f}')

        # 对异常值进行特殊处理的拟合曲线
        res = 0.03
        valid_index = np.where((values < 1 - res) & (values > 0))[0]
        if len(values) / 3 > len(valid_index) > 2:
            valid_index = PolarizationAnalyser._expand_valid_index(valid_index)

        valid_values = values[valid_index]
        valid_qwp_angles = qwp_angles[valid_index]
        plt.scatter(valid_qwp_angles * 180 / np.pi, valid_values, label='valid scatter')

        valid_stokes = PolarizationAnalyser.solve_full_stokes_contradictory_equation(
            lp_angle, valid_qwp_angles, valid_values)

        valid_s0 = valid_stokes[0]
        valid_s1 = valid_stokes[1]
        valid_s2 = valid_stokes[2]
        valid_s3 = valid_stokes[3]

        y2 = 0.5 * (valid_s0 + np.cos(2 * x) * np.cos(2 * lp_angle - 2 * x) * valid_s1
                    + np.sin(2 * x) * np.cos(2 * lp_angle - 2 * x) * valid_s2
                    + np.sin(2 * lp_angle - 2 * x) * valid_s3)
        plt.plot(x * 180 / np.pi, y2, label='eliminate invalid values')

        valid_rho = (np.sqrt(pow(valid_s1, 2) + pow(valid_s2, 2) + pow(valid_s3, 2)) /
                     (valid_s0 + np.finfo(np.float32).eps))
        print(f'剔除无效测量点拟合曲线，对应的偏振度为 {valid_rho:.4f}')

        plt.title('Polarization Value vs Angles')
        plt.xlabel('Angles')
        plt.ylabel('Values')
        plt.xlim((qwp_angles.min() * 180 / np.pi, qwp_angles.max() * 180 / np.pi))
        plt.legend()
        plt.show(block=True)

        return qwp_angles, values, x, y1, y2

    def plot_linear_polarization_curve(self, coord_y, coord_x):
        """绘制目标点的线偏振特征曲线"""
        values = self.polar_dataset.polarization_images[coord_y, coord_x, :]
        lp_angles = self.polar_dataset.lp_angles

        plt.scatter(lp_angles * 180 / np.pi, values, label='scatter')

        # 传统解析方法拟合曲线
        stokes = self.solve_linear_stokes_contradictory_equation(lp_angles, values)
        s0 = stokes[0]
        s1 = stokes[1]
        s2 = stokes[2]

        x = np.arange(lp_angles.min(), lp_angles.max(), 0.01)
        y1 = 0.5 * (s0 + np.cos(2 * x) * s1 + np.sin(2 * x) * s2)
        plt.plot(x * 180 / np.pi, y1, label='traditional fitting')

        rho = np.sqrt(pow(s1, 2) + pow(s2, 2)) / s0
        print(f'传统方法曲线，对应的偏振度为{rho:.4f}')
        print(f'平均偏振强度值为 {np.mean(values):.4f}')

        # 对异常值进行特殊处理的拟合曲线
        res = 0.03
        valid_index = np.where((values < 1 - res) & (values > 0))[0]
        if len(values) / 3 > len(valid_index) > 2:
            valid_index = PolarizationAnalyser._expand_valid_index(valid_index)

        valid_values = values[valid_index]
        valid_lp_angles = lp_angles[valid_index]
        plt.scatter(valid_lp_angles * 180 / np.pi, valid_values, label='valid scatter')

        valid_stokes = PolarizationAnalyser.solve_linear_stokes_contradictory_equation(valid_lp_angles, valid_values)

        valid_s0 = valid_stokes[0]
        valid_s1 = valid_stokes[1]
        valid_s2 = valid_stokes[2]

        y2 = 0.5 * (valid_s0 + np.cos(2 * x) * valid_s1 + np.sin(2 * x) * valid_s2)
        plt.plot(x * 180 / np.pi, y2, label='eliminate invalid values')

        valid_rho = np.sqrt(pow(valid_s1, 2) + pow(valid_s2, 2)) / (valid_s0 + np.finfo(np.float32).eps)
        print(f'剔除无效测量点拟合曲线，对应的偏振度为 {valid_rho:.4f}')

        plt.title('Polarization Value vs Angles')
        plt.xlabel('Angles')
        plt.ylabel('Values')
        plt.xlim((lp_angles.min() * 180 / np.pi, lp_angles.max() * 180 / np.pi))
        plt.legend()
        plt.show(block=True)


class ImageCropper:
    """图片剪裁器，通过框选目标位置或指定目标位置剪裁图片"""

    def __init__(self, image, crop_coordinates: list, aspect_ratio=None, visualize=False):
        self.image = image
        self.crop_coordinates = crop_coordinates
        self.manual_crop_coordinates = []
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None

        if any(coord is None for coord in crop_coordinates):
            # 如果没有完全指定框选位置，则手动框选
            scale_ratio = 1
            while (image.shape[0] / scale_ratio) * (image.shape[1] / scale_ratio) > 1e6:
                scale_ratio += 1
            self.scale_ratio = scale_ratio
            self.pil_image = resize_pil_image(np2pil(self.image), self.scale_ratio)
            self.aspect_ratio = aspect_ratio

            self.root = tk.Tk()
            self.root.wm_title("图像剪裁器")
            self.canvas_width = self.pil_image.width
            self.canvas_height = self.pil_image.height
            self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height)
            self.canvas.pack()
            self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
            self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
            self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)

            self.photo = ImageTk.PhotoImage(self.pil_image)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

            self.root.mainloop()

        if None in crop_coordinates:
            self.update_crop_coordinates()

        self.image_cropped = self.imcrop(image, self.crop_coordinates, aspect_ratio)

        if visualize:
            self.show_cropped_image()

    @staticmethod
    def coordinates_absolute2ratio(absolute_coordinates: list, image_shape: tuple):
        """将绝对坐标值转化为相对比例位置"""
        height, width = image_shape[:2]
        center_x, center_y, length_x, length_y = absolute_coordinates
        ratio_coordinates = [center_x / width, center_y / height, length_x / width, length_y / height]
        return ratio_coordinates

    @staticmethod
    def coordinates_ratio2absolute(ratio_coordinates: list, image_shape: tuple):
        """将相对比例位置转化为绝对坐标值"""
        height, width = image_shape[:2]
        center_x, center_y, length_x, length_y = ratio_coordinates
        absolute_coordinates = [int(center_x * width), int(center_y * height),
                                int(length_x * width), int(length_y * height)]
        return absolute_coordinates

    def on_mouse_press(self, event):
        self.start_x = event.x
        self.start_y = event.y

    def on_mouse_drag(self, event):
        self.canvas.delete("crop_rectangle")
        current_x = event.x
        current_y = event.y
        self.canvas.create_rectangle(self.start_x, self.start_y, current_x, current_y,
                                     outline="red", tags="crop_rectangle")

    def on_mouse_release(self, event):
        self.canvas.delete("crop_rectangle")
        self.end_x = event.x
        self.end_y = event.y

        center_x = (self.start_x + self.end_x) / 2
        center_y = (self.start_y + self.end_y) / 2
        length_x = abs(self.start_x - self.end_x)
        length_y = abs(self.start_y - self.end_y)

        resized_crop_coordinates = (center_x, center_y, length_x, length_y)
        self.manual_crop_coordinates = [int(coord * self.scale_ratio) for coord in resized_crop_coordinates]

        self.root.quit()
        self.root.destroy()

    def update_crop_coordinates(self):
        """更新框选位置的区域"""
        none_indices = [index for index, value in enumerate(self.crop_coordinates) if value is None]
        for index in none_indices:
            self.crop_coordinates[index] = self.manual_crop_coordinates[index]

    @staticmethod
    def imcrop(image: np.ndarray, crop_coordinates: list, aspect_ratio=None):
        """
        根据鼠标框定的位置，剪裁图像
        :param image: 灰度图像或彩色图像[H,W,C]
        :param crop_coordinates: 剪裁区域[center_x, center_y, length_x, length_y]
        :param aspect_ratio: 剪裁图像的长宽比
        :return: image_cropped
        """
        height, width, left_up_x, left_up_y = ImageCropper.coordinates_center2corner(crop_coordinates, aspect_ratio)

        assert len(image.shape) == 2 or len(image.shape) == 3, 'only support gray and color image'
        image_cropped = image[left_up_y: left_up_y + height, left_up_x: left_up_x + width]

        return image_cropped

    @staticmethod
    def coordinates_center2corner(center_coordinates, aspect_ratio=None):
        """
        :param center_coordinates: [center_x, center_y, length_x, length_y]
        :param aspect_ratio: rect aspect ratio, type=float
        :return corner_coordinates: [height, width, left_up_x, left_up_y]
        """
        height = center_coordinates[3]
        width = center_coordinates[2]

        if aspect_ratio:
            if height < width:
                width = int(aspect_ratio * height)
            else:
                height = int(aspect_ratio * width)

        left_up_x = int(center_coordinates[0] - width / 2)
        left_up_y = int(center_coordinates[1] - height / 2)

        return height, width, left_up_x, left_up_y

    def show_cropped_image(self):
        plt.figure('cropped image')
        plt.imshow(self.image_cropped)
        plt.show(block=True)


class ClickToGetPixelPosition:
    def __init__(self, image):
        """ 点击图像中任意一点，得到该点的坐标"""
        self.image = imnorm(image, mode='min-max')

        scale_ratio = 1
        while (image.shape[0] / scale_ratio) * (image.shape[1] / scale_ratio) > 1e6:
            scale_ratio += 1

        self.scale_ratio = scale_ratio
        self.pil_image = resize_pil_image(np2pil(self.image), self.scale_ratio)

        self.coord_x = None
        self.coord_y = None

        self.root = tk.Tk()
        self.root.wm_title('请点击图像中的某一处，得到该点的坐标位置')
        self.canvas_width = self.pil_image.width
        self.canvas_height = self.pil_image.height
        self.canvas = tk.Canvas(self.root, width=self.canvas_width, height=self.canvas_height)
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.photo = ImageTk.PhotoImage(self.pil_image)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        self.root.mainloop()

    def on_mouse_press(self, event):
        """当鼠标左键在 canvas 上被按下时，这个方法会被调用"""
        self.coord_x = self.scale_ratio * event.x
        self.coord_y = self.scale_ratio * event.y

        self.root.quit()
        self.root.destroy()

        print(f"鼠标点击坐标位置： ({self.coord_x}, {self.coord_y})")


class MCPolarDatasetManager:
    """多通道偏振数据管理器"""
    DTYPE_DOT_FP = 0
    DTYPE_DOT_LP = 1
    DTYPE_DOF_LP = 2

    def __init__(self, dataset_type):
        self.dataset_type = dataset_type
        self.mc_polarization_images = None

        if self.dataset_type == MCPolarDatasetManager.DTYPE_DOT_LP:
            self.lp_angles = None

        self.channels = None
        self.save_path = None

    def import_mc_linear_polarization_data(self, mc_polarization_images, lp_angles):
        """
        :param mc_polarization_images: polarization images [H, W, B, C]
        :param lp_angles: linear polarization angles
        """
        assert self.dataset_type == MCPolarDatasetManager.DTYPE_DOT_LP
        self.mc_polarization_images = mc_polarization_images
        self.lp_angles = lp_angles
        self.channels = mc_polarization_images.shape[3]

    def crop_mc_polarization_images(self, crop_coordinates, aspect_ratio=None):
        mc_polarization_images = []
        for k in range(self.channels):
            cropper = ImageCropper(self.mc_polarization_images[:, :, :, k], crop_coordinates, aspect_ratio)
            mc_polarization_images.append(cropper.image_cropped)

        self.mc_polarization_images = np.array(mc_polarization_images).transpose(1, 2, 3, 0)


class MCPolarizationAnalyser:
    """多通道偏振解析器"""

    def __init__(self, mc_polar_dataset: MCPolarDatasetManager, resize_ratio=1.):
        self.mc_polar_dataset = mc_polar_dataset
        self.resize_ratio = resize_ratio
        self.channels = mc_polar_dataset.channels

        self.mc_polarization_images = mc_polar_dataset.mc_polarization_images
        self.mc_stokes = None
        self.mc_rho = None
        self.mc_phi = None
        self.mc_iun = None
        self.mc_imin = None
        self.mc_imax = None

        self.mc_iun_ = None
        self.mc_imin_ = None
        self.mc_imax_ = None

        self.polar_datasets = None
        self.polar_analysers = None

    def create_mc_polar_analyser(self):
        self.polar_datasets = []
        self.polar_analysers = []
        for k in range(self.channels):
            if self.mc_polar_dataset.dataset_type == MCPolarDatasetManager.DTYPE_DOT_LP:
                polar_dataset = PolarDatasetManager(self.mc_polar_dataset.dataset_type)
                polar_dataset.polarization_images = self.mc_polarization_images[:, :, :, k]
                polar_dataset.lp_angles = self.mc_polar_dataset.lp_angles
                polar_dataset.save_path = self.mc_polar_dataset.save_path
                polar_analyser = PolarizationAnalyser(polar_dataset, self.resize_ratio)

                self.polar_datasets.append(polar_dataset)
                self.polar_analysers.append(polar_analyser)

    def calc_stokes(self, highlight_removal=False, verbose=False, visualize=False):
        start_time = time.time()
        mc_stokes = []

        for k in range(self.channels):
            polar_analyser: PolarizationAnalyser = self.polar_analysers[k]
            polar_analyser.calc_stokes(highlight_removal)
            mc_stokes.append(polar_analyser.stokes)

        self.mc_stokes = np.array(mc_stokes).transpose(1, 2, 3, 0)

        if verbose:
            print(f'calc stokes cost {time.time() - start_time:.4f} s')

        if visualize:
            plt.figure('multi-channel stokes')
            plt.subplot(131)
            plt.imshow(np.mean(self.mc_stokes, axis=3)[:, :, 0])
            plt.title('multi-channel s0')
            plt.subplot(132)
            plt.imshow(np.mean(self.mc_stokes, axis=3)[:, :, 1])
            plt.title('multi-channel s1')
            plt.subplot(133)
            plt.imshow(np.mean(self.mc_stokes, axis=3)[:, :, 2])
            plt.title('multi-channel s2')
            plt.show(block=True)

        if self.resize_ratio != 1.:
            mc_polarization_images = []
            for k in range(self.channels):
                mc_polarization_images.append(self.polar_analysers[k].images)
            self.mc_polarization_images = np.array(mc_polarization_images).transpose(1, 2, 3, 0)

    def calc_polar_features(self, verbose=False, visualize=False):
        start_time = time.time()
        mc_rho = []
        mc_phi = []
        mc_iun = []
        mc_imin = []
        mc_imax = []

        for k in range(self.channels):
            polar_analyser: PolarizationAnalyser = self.polar_analysers[k]
            polar_analyser.calc_polar_features()

            mc_rho.append(polar_analyser.rho)
            mc_phi.append(polar_analyser.phi)
            mc_iun.append(polar_analyser.iun)
            mc_imin.append(polar_analyser.imin)
            mc_imax.append(polar_analyser.imax)

        self.mc_rho = np.array(mc_rho).transpose((1, 2, 0))
        self.mc_phi = np.array(mc_phi).transpose((1, 2, 0))
        self.mc_iun = np.array(mc_iun).transpose((1, 2, 0))
        self.mc_imin = np.array(mc_imin).transpose((1, 2, 0))
        self.mc_imax = np.array(mc_imax).transpose((1, 2, 0))

        if verbose:
            print(f'calc polar features cost {time.time() - start_time:.4f} s')

        if visualize:
            plt.figure('multi-channel polar features')
            plt.subplot(231)
            plt.imshow(np.mean(self.mc_rho, axis=-1))
            plt.title('multi-channel rho')
            plt.subplot(232)
            plt.imshow(np.mean(self.mc_phi, axis=-1))
            plt.title('multi-channel phi')
            plt.subplot(234)
            plt.imshow(np.mean(self.mc_iun, axis=-1))
            plt.title('multi-channel iun')
            plt.subplot(235)
            plt.imshow(np.mean(self.mc_imin, axis=-1))
            plt.title('multi-channel imin')
            plt.subplot(236)
            plt.imshow(np.mean(self.mc_imax, axis=-1))
            plt.title('multi-channel imax')
            plt.show(block=True)

    def calc_inten_features(self, verbose=False, visualize=False):
        start_time = time.time()
        mc_iun_ = []
        mc_imin_ = []
        mc_imax_ = []

        for k in range(self.channels):
            polar_analyser: PolarizationAnalyser = self.polar_analysers[k]
            polar_analyser.calc_inten_features(visualize=visualize)

            mc_iun_.append(polar_analyser.iun_)
            mc_imin_.append(polar_analyser.imin_)
            mc_imax_.append(polar_analyser.imax_)

        self.mc_iun_ = np.array(mc_iun_).transpose(1, 2, 0)
        self.mc_imin_ = np.array(mc_imin_).transpose(1, 2, 0)
        self.mc_imax_ = np.array(mc_imax_).transpose(1, 2, 0)

        if verbose:
            print(f'calc inten features cost {time.time() - start_time:.4f} s')

        if visualize:
            plt.figure('multi-channel inten features')
            plt.subplot(131)
            plt.imshow(np.mean(self.mc_iun_, axis=-1))
            plt.title('multi-channel iun_')
            plt.subplot(132)
            plt.imshow(np.mean(self.mc_imin_, axis=-1))
            plt.title('multi-channel imin_')
            plt.subplot(133)
            plt.imshow(np.mean(self.mc_imax_, axis=-1))
            plt.title('multi-channel imax_')
            plt.show(block=True)


def stokes2rgb(stokes: np.ndarray, norm=False):
    if norm:
        s0 = imnorm(stokes[:, :, 0], mode='min-max')
        s1 = imnorm(stokes[:, :, 1], mode='min-max')
        s2 = imnorm(stokes[:, :, 2], mode='min-max')
    else:
        s0 = stokes[:, :, 0] / 2
        s1 = stokes[:, :, 1] / 2
        s2 = stokes[:, :, 2] / 2

    rgb = np.stack((s0, s1, s2), axis=2)
    return rgb


def pdm_fp2lp(fp_dataset_manager: PolarDatasetManager):
    """polar dataset manager, translate from full polarization to linear polarization"""
    assert fp_dataset_manager.dataset_type == PolarDatasetManager.DTYPE_DOT_FP
    fp_dataset_analyser = PolarizationAnalyser(fp_dataset_manager)
    fp_dataset_analyser.calc_stokes(highlight_removal=False)
    stokes = fp_dataset_analyser.stokes

    i0 = PolarizationAnalyser.stokes2lp(stokes, np.deg2rad(0))
    i45 = PolarizationAnalyser.stokes2lp(stokes, np.deg2rad(45))
    i90 = PolarizationAnalyser.stokes2lp(stokes, np.deg2rad(90))
    i135 = PolarizationAnalyser.stokes2lp(stokes, np.deg2rad(135))

    polarization_images = np.stack((i0, i45, i90, i135), axis=-1)

    lp_dataset_manager = PolarDatasetManager(dataset_type=PolarDatasetManager.DTYPE_DOT_LP)
    lp_dataset_manager.polarization_images = polarization_images
    lp_dataset_manager.lp_angles = np.deg2rad(np.array([0., 45., 90., 135.]))
    lp_dataset_manager.save_path = fp_dataset_manager.save_path

    return lp_dataset_manager


class ReflectionPolarAnalyser:
    """反射偏振分析器"""
    DIELECTRIC = 0
    METAL = 1

    def __init__(self, n_i, n_t, num=1001, save_dir=Path(os.path.join('..', 'runs'))):

        self.n_i = complex(n_i)
        self.n_t = complex(n_t)

        if n_t.imag == 0:
            self.type = self.DIELECTRIC
        else:
            self.type = self.METAL
        print(f'n_i={self.n_i:.2f}, n_t={self.n_t:.2f}')

        self.num = num

        self.critical_angle = cmath.asin(self.n_t / self.n_i)
        self.brewster_angle = cmath.atan(self.n_t / self.n_i)
        print(f'type={"metal" if self.type else "dielectric"}: \t'
              f'Brewster angle is {np.rad2deg(self.brewster_angle.real):.2f} deg \t'
              f'Critical angle is {np.rad2deg(self.critical_angle.real):.2f} deg')

        self.ndarray_theta_i = np.linspace(0, np.pi / 2, num=self.num)

        self.save_dir = save_dir

    def _theta_i2t(self, theta_i):
        """复数形式的三角函数运算需要运用cmath库"""
        n_i = self.n_i
        n_t = self.n_t

        sin_theta_t = n_i * cmath.sin(theta_i) / n_t
        theta_t = cmath.asin(sin_theta_t)

        return theta_t

    def ndarray_theta_i2t(self):
        """计算入射角对应的折射角"""
        ndarray_theta_t = []
        for k in range(len(self.ndarray_theta_i)):
            theta_i = self.ndarray_theta_i[k]
            if theta_i > self.critical_angle.real:
                break
            sin_theta_t = self.n_i.real * cmath.sin(theta_i) / self.n_t.real
            theta_t = cmath.asin(sin_theta_t)
            ndarray_theta_t.append(theta_t.real)

        return np.array(ndarray_theta_t)


    def calc_rs_rp(self, theta_i):
        """计算振幅反射系数"""
        n_i = self.n_i
        n_t = self.n_t

        theta_t = self._theta_i2t(theta_i)

        if self.type == self.DIELECTRIC:
            r_s = (n_i * np.cos(theta_i) - n_t * np.cos(theta_t)) / (n_i * np.cos(theta_i) + n_t * np.cos(theta_t))
            r_p = (n_t * np.cos(theta_i) - n_i * np.cos(theta_t)) / (n_t * np.cos(theta_i) + n_i * np.cos(theta_t))
        else:  # self.type == METAL
            n = n_t.real
            chi = n_t.imag

            double_N_square = (n**2 - chi**2 - n_i**2 * np.sin(theta_i)**2 +
                               np.sqrt((n**2 - chi**2 - n_i**2 * np.sin(theta_i)**2)**2 + 4 * n**2 * chi**2))
            double_chi_prime_square = (-(n**2 - chi**2 - n_i**2 * np.sin(theta_i)**2) +
                                       np.sqrt((n**2 - chi**2 - n_i**2 * np.sin(theta_i)**2)**2 + 4 * n**2 * chi**2))
            N = np.sqrt(0.5 * double_N_square)
            chi_prime = np.sqrt(0.5 * double_chi_prime_square)

            r_s = ((n_i**2 * np.cos(theta_i)**2 - double_N_square/2) - double_chi_prime_square/2 + 2j * chi_prime * n_i * np.cos(theta_i)) / ((n_i * np.cos(theta_i) + N)**2 + double_chi_prime_square/2)
            r_p = ((n**2 + chi**2)**2 * np.cos(theta_i)**2 - n_i**2 * (double_N_square/2 + double_chi_prime_square/2) + 2j * n_i * np.cos(theta_i) * ((n**2 - chi**2) * chi_prime - 2 * N * n * chi)) / (((n**2 - chi**2) * np.cos(theta_i) + n_i * N)**2 + (2 * n * chi * np.cos(theta_i) + n_i * chi_prime)**2)

        return r_s, r_p
    
    def calc_ts_tp(self, theta_i):
        """计算振幅折射系数"""
        n_i = self.n_i
        n_t = self.n_t

        theta_t = self._theta_i2t(theta_i)

        t_s = (2 * n_i * np.cos(theta_i)) / (n_i * np.cos(theta_i) + n_t * np.cos(theta_t))
        t_p = (2 * n_i * np.cos(theta_i)) / (n_t * np.cos(theta_i) + n_i * np.cos(theta_t))

        return t_s, t_p
    
    def calc_Ts_Tp(self, theta_i):
        """计算能量折射比"""
        if theta_i == 0.:
            theta_i = 1e-7

        theta_t = self._theta_i2t(theta_i)

        T_s = (cmath.sin(2 * theta_i) * cmath.sin(2 * theta_t)) / (cmath.sin(theta_i + theta_t) ** 2)
        T_p = (cmath.sin(2 * theta_i) * cmath.sin(2 * theta_t)) / (cmath.sin(theta_i + theta_t) ** 2 * cmath.cos(theta_i - theta_t) ** 2)

        return T_s, T_p
    
    def calc_Mr(self, theta_i):
        """计算反射穆勒矩阵"""
        if theta_i == 0.:
            theta_i = 1e-7

        theta_t = self._theta_i2t(theta_i)

        a = theta_i - theta_t
        b = theta_i + theta_t

        M_r = 0.5 * (np.tan(a)/np.sin(b)) ** 2 * np.array(
            [[np.cos(a)**2 + np.cos(b)**2, np.cos(a)**2 - np.cos(b)**2, 0, 0],
             [np.cos(a)**2 - np.cos(b)**2, np.cos(a)**2 + np.cos(b)**2, 0, 0],
             [0, 0, -2*np.cos(a)*np.cos(b), 0],
             [0, 0, 0, -2*np.cos(a)*np.cos(b)]])

        return M_r

    def calc_Mt(self, theta_i):
        """计算折射穆勒矩阵"""
        if theta_i == 0:
            theta_i = 1e-7

        theta_t = self._theta_i2t(theta_i)

        a = theta_i - theta_t
        b = theta_i + theta_t

        M_t = 0.5 * (np.sin(2 * theta_i) * np.sin(2 * theta_t)) / (np.sin(b) * np.cos(a)) ** 2 * np.array(
            [[np.cos(a)**2 + 1, np.cos(a)**2 - 1, 0, 0],
             [np.cos(a)**2 - 1, np.cos(a)**2 + 1, 0, 0],
             [0, 0, -2 * np.cos(a), 0],
             [0, 0, 0, -2 * np.cos(a)]])

        return M_t
    
    def r2delta(self, ndarray_r):
        """将输入的反射系数 ndarray_r 处理为复数形式，并提取相位信息"""
        ndarray_r = ndarray_r.astype(complex)
        ndarray_delta = np.angle(ndarray_r)
        return ndarray_r, ndarray_delta


    def calc_rs_rp_curve(self, save=False):
        """工程光学.图11-9 rs、rp随入射角的变化关系"""
        ndarray_theta_i = self.ndarray_theta_i

        ndarray_r_s = []
        ndarray_r_p = []
        for theta_i in ndarray_theta_i:
            r_s, r_p = self.calc_rs_rp(theta_i)
            ndarray_r_s.append(r_s)
            ndarray_r_p.append(r_p)

        ndarray_r_s = np.array(ndarray_r_s)
        ndarray_r_p = np.array(ndarray_r_p)

        ndarray_r_s, ndarray_delta_s = self.r2delta(ndarray_r_s)
        ndarray_r_p, ndarray_delta_p = self.r2delta(ndarray_r_p)

        if save:
            plt.figure('reflection coefficient curve')
            plt.title('reflection coefficient curve')

            plt.subplot(311)
            plt.plot(np.rad2deg(ndarray_theta_i), self.normalize_angle_to_0_pi(ndarray_delta_s), label='delta_s')
            plt.plot(np.rad2deg(ndarray_theta_i), self.normalize_angle_to_0_pi(ndarray_delta_p), label='delta_p')
            plt.xlim(0, 90)
            plt.xlabel('theta_i (degrees)')
            plt.ylabel('delta (radians)')
            plt.legend()

            plt.subplot(312)
            plt.plot(np.rad2deg(ndarray_theta_i), abs(ndarray_r_s), label='r_s')
            plt.plot(np.rad2deg(ndarray_theta_i), abs(ndarray_r_p), label='r_p')
            plt.xlim(0, 90)
            plt.xlabel('theta_i (degrees)')
            plt.ylabel('r (magnitude)')
            plt.legend()
            
            save_path = os.path.join(self.save_dir, 'rs_rp_curve.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()  # 关闭当前图形，释放内存

        return ndarray_theta_i, ndarray_r_s, ndarray_r_p, ndarray_delta_s, ndarray_delta_p

    def calc_ts_tp_curve(self, save=False):
        """工程光学.图11-9 ts、tp随入射角的变化关系"""
        ndarray_theta_i = self.ndarray_theta_i

        ndarray_t_s = []
        ndarray_t_p = []
        for theta_i in ndarray_theta_i:
            t_s, t_p = self.calc_ts_tp(theta_i)
            ndarray_t_s.append(t_s)
            ndarray_t_p.append(t_p)

        ndarray_t_s = np.array(ndarray_t_s)
        ndarray_t_p = np.array(ndarray_t_p)

        ndarray_t_s, ndarray_delta_s = self.r2delta(ndarray_t_s)
        ndarray_t_p, ndarray_delta_p = self.r2delta(ndarray_t_p)

        if save:
            plt.figure('transmission coefficient curve')
            plt.title('transmission coefficient curve')

            plt.subplot(311)
            plt.plot(np.rad2deg(ndarray_theta_i), self.normalize_angle_to_0_pi(ndarray_delta_s), label='delta_s')
            plt.plot(np.rad2deg(ndarray_theta_i), self.normalize_angle_to_0_pi(ndarray_delta_p), label='delta_p')
            plt.xlim(0, 90)
            plt.xlabel('theta_i')
            plt.ylabel('delta')
            plt.legend()

            plt.subplot(312)
            plt.plot(np.rad2deg(ndarray_theta_i), abs(ndarray_t_s), label='t_s')
            plt.plot(np.rad2deg(ndarray_theta_i), abs(ndarray_t_p), label='t_p')
            plt.xlim(0, 90)
            plt.xlabel('theta_i')
            plt.ylabel('t')
            plt.legend()

            save_path = os.path.join(self.save_dir, 'rs_rp_curve.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()  # 关闭当前图形，释放内存

        return ndarray_theta_i, ndarray_t_s, ndarray_t_p, ndarray_delta_s, ndarray_delta_p

    @staticmethod
    def normalize_angle_to_0_pi(theta):
        """将角度归一化到 [0, pi]"""
        # 角度范围至 [0, 2*pi)
        theta = theta % (2 * np.pi)

        # 如果角度大于等于 pi，则映射到 (0, pi]
        theta = np.where(theta > np.pi, 2*np.pi - theta, theta)

        return np.clip(theta, 0, np.pi)

    def calc_Rs_Rp_curve(self, save=False):
        """偏振光学(p43)分界面上反射时相位和振幅的变化"""
        ndarray_theta_i, ndarray_r_s, ndarray_r_p, ndarray_delta_s, ndarray_delta_p = self.calc_rs_rp_curve()

        ndarray_R_s = abs(ndarray_r_s) ** 2
        ndarray_R_p = abs(ndarray_r_p) ** 2

        if save:
            P = np.sqrt(ndarray_R_s / ndarray_R_p)
            delta = ndarray_delta_p - ndarray_delta_s

            plt.figure('calc reflectance curve')
            plt.title('calc reflectance curve')

            plt.subplot(311)
            plt.plot(np.rad2deg(ndarray_theta_i), self.normalize_angle_to_0_pi(-delta))
            plt.xlim(0, 90)
            plt.xlabel('theta_i')
            plt.ylabel('delta')

            plt.subplot(312)
            plt.plot(np.rad2deg(ndarray_theta_i), P)
            plt.xlim(0, 90)
            plt.xlabel('theta_i')
            plt.ylabel('P')

            plt.subplot(313)
            plt.plot(np.rad2deg(ndarray_theta_i), ndarray_R_s, label='R_s')
            plt.plot(np.rad2deg(ndarray_theta_i), ndarray_R_p, label='R_p')
            plt.xlim(0, 90)
            plt.ylim(0, 1.1)
            plt.xlabel('theta_i')
            plt.ylabel('R')
            plt.legend()
            
            save_path = os.path.join(self.save_dir, 'Rs_Rp_curve.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()  # 关闭当前图形，释放内存

        return ndarray_theta_i, ndarray_R_s, ndarray_R_p


    def calc_Ts_Tp_curve(self, save=False):
        ndarray_theta_i = self.ndarray_theta_i

        ndarray_T_s = []
        ndarray_T_p = []
        for theta_i in ndarray_theta_i:
            T_s, T_p = self.calc_Ts_Tp(theta_i)
            ndarray_T_s.append(T_s)
            ndarray_T_p.append(T_p)

        ndarray_T_s = np.array(ndarray_T_s)
        ndarray_T_p = np.array(ndarray_T_p)

        ndarray_theta_i, ndarray_t_s, ndarray_t_p, ndarray_delta_s, ndarray_delta_p = self.calc_ts_tp_curve(save=save)

        if save:
            P = np.sqrt(ndarray_T_s / ndarray_T_p)
            delta = ndarray_delta_p - ndarray_delta_s

            plt.figure('calc refraction curve')
            plt.title('calc refraction curve')

            plt.subplot(311)
            plt.plot(np.rad2deg(ndarray_theta_i), self.normalize_angle_to_0_pi(-delta))
            plt.xlim(0, 90)
            plt.xlabel('theta_i')
            plt.ylabel('delta')

            plt.subplot(312)
            plt.plot(np.rad2deg(ndarray_theta_i), abs(P))
            plt.xlim(0, 90)
            plt.xlabel('theta_i')
            plt.ylabel('P')

            plt.subplot(313)
            plt.plot(np.rad2deg(ndarray_theta_i), abs(ndarray_T_s), label='T_s')
            plt.plot(np.rad2deg(ndarray_theta_i), abs(ndarray_T_p), label='T_p')
            plt.xlim(0, 90)
            plt.xlabel('theta_i')
            plt.ylabel('T')
            plt.legend()
            
            save_path = os.path.join(self.save_dir, 'Ts_Tp_curve.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()  # 关闭当前图形，释放内存

        return ndarray_theta_i, ndarray_T_s, ndarray_T_p
    
    def calc_reflected_dop_curve(self, inp_stokes=np.array([1, 0, 0, 0]), method='fresnel', save=False):
        """镜面反射偏振度曲线"""
        if method == 'fresnel':
            ndarray_theta_i, ndarray_R_s, ndarray_R_p = self.calc_Rs_Rp_curve(save=save)
            ndarray_dop = abs(ndarray_R_s - ndarray_R_p) / (ndarray_R_s + ndarray_R_p)
        elif method == 'muller':
            ndarray_theta_i = np.linspace(0, np.pi / 2, num=self.num)
            ndarray_dop = []
            for theta_i in ndarray_theta_i:
                M_r = self.calc_Mr(theta_i)
                out_stokes = np.dot(M_r, inp_stokes.reshape(4, 1))
                dop = np.sqrt(out_stokes[1]**2 + out_stokes[2]**2 + out_stokes[3]**2) / out_stokes[0]
                ndarray_dop.append(dop)
            ndarray_dop = np.array(ndarray_dop)[:, 0]
        else:
            raise ValueError

        if save:
            plt.figure('reflected dop')
            plt.title('dop')

            plt.plot(np.rad2deg(ndarray_theta_i), ndarray_dop.real, label='dop')
            plt.xlim(0, 90)
            plt.ylim(0., 1.2)
            plt.xlabel('theta_i')
            plt.ylabel('dop')
            plt.legend()
            
            save_path = os.path.join(self.save_dir, 'reflected_dop.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()  # 关闭当前图形，释放内存

        return ndarray_theta_i, ndarray_dop

    def calc_refracted_dop_curve(self, inp_stokes=np.array([1, 0, 0, 0]), method='muller', save=False):
        if method == 'fresnel':
            ndarray_theta_i, ndarray_T_s, ndarray_T_p = self.calc_Ts_Tp_curve(save=save)
            ndarray_dop = abs(ndarray_T_s - ndarray_T_p) / (ndarray_T_s + ndarray_T_p)
        elif method == 'muller':
            ndarray_theta_i = np.linspace(0, np.pi / 2, num=self.num)
            ndarray_dop = []
            for theta_i in ndarray_theta_i:
                M_t = self.calc_Mt(theta_i)
                out_stokes = np.dot(M_t, inp_stokes.reshape(4, 1))
                dop = np.sqrt(out_stokes[1] ** 2 + out_stokes[2] ** 2 + out_stokes[3] ** 2) / out_stokes[0]
                ndarray_dop.append(dop)
            ndarray_dop = np.array(ndarray_dop)[:, 0]
        else:
            raise ValueError

        if save:
            plt.figure('refracted dop')
            plt.title('dop')
            
            plt.plot(np.rad2deg(ndarray_theta_i), ndarray_dop.real, label='dop')
            plt.xlim(0, 90)
            plt.ylim(0., 1.2)
            plt.xlabel('theta_i')
            plt.ylabel('dop')
            plt.grid()
            plt.legend()
            
            save_path = os.path.join(self.save_dir, 'refracted_dop.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()  # 关闭当前图形，释放内存

        return ndarray_theta_i, ndarray_dop
