import matlab
import matlab.engine
# import matlab.engine导入的时候，前面不要导入其他内置模块

import os
import skimage
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat

__all__ = ['MatlabImageFuser']


def add_recursive_paths(engine, path):
    """
    Recursively adds paths and sub-paths to the MATLAB engine.
    :param engine: MATLAB engine instance.
    :param path: Path to the directory that needs to be added.
    """
    engine.addpath(path)
    for root, dirs, _ in os.walk(path):
        for directory in dirs:
            full_path = os.path.join(root, directory)
            engine.addpath(full_path)


class MatlabImageFuser:
    """多模态图像融合器"""
    def __init__(self):
        project_address = os.path.join(os.path.dirname(os.path.abspath(__file__)), '')
        self.matlab_engine = matlab.engine.start_matlab()
        # Add the search dir of matlab
        add_recursive_paths(self.matlab_engine, os.path.join(project_address, '../matlab_utils/evaluation_methods'))
        add_recursive_paths(self.matlab_engine, os.path.join(project_address, '../matlab_utils/fusion_methods'))

    def __del__(self):
        self.matlab_engine.quit()

    @staticmethod
    def _np_to_mat(img):
        """
        Transfer numpy to matlab style
        :param img: image, np.array
        :return: matlab style
        """
        img_mat = matlab.double(img.tolist())
        return img_mat

    @staticmethod
    def _mat_to_np(img_mat):
        """
        Transfer matlab style to numpy
        :param img: image, matlab style
        :return: np.array
        """
        img_np = np.array(img_mat)
        return img_np

    # ************************************************* NSCT ****************************************************
    def fuse_by_nsct(self, img1, img2):
        """
        https://github.com/Keep-Passion/SESF-Fuse
        Q. Zhang, B. Guo, Multifocus image fusion using the nonsubsampled contourlet
        transform, Signal Process. 89 (7) (2009) 1334–1346
        Liu Y, Liu S, Wang Z. A general framework for image fusion based on multi-scale transform and
        sparse representation[J]. Information Fusion, 2015, 24: 147-164.
        :param img1: last image, np.array
        :param img2: next image, np.array
        """
        fused = np.zeros(img1.shape)
        param_mat = self._np_to_mat(np.array([2, 3, 3, 4]))
        if img1.ndim == 2:
            img1_mat = self._np_to_mat(img1)
            img2_mat = self._np_to_mat(img2)
            fused = self.matlab_engine.nsct_fuse(img1_mat, img2_mat, param_mat)
        else:   # color mode
            for index in range(0, 3):
                img1_mat = self._np_to_mat(img1[:, :, index])
                img2_mat = self._np_to_mat(img2[:, :, index])
                fused[:, :, index] = self.matlab_engine.nsct_fuse(img1_mat, img2_mat, param_mat)
        fused= np.clip(self._mat_to_np(fused), 0, 1)
        return fused

    # ************************************************* CVT *****************************************************
    def fuse_by_cvt(self, img1, img2):
        """
        CVT （Curvelet transform）image fusion
        The codes are copied from http://www.escience.cn/people/liuyu1/Codes.html
        F. Nencini, A. Garzelli, S. Baronti, L. Alparone, Remote sensing image fusion
        using the curvelet transform, Inform. Fusion 8 (2) (2007) 143–156
        Liu Y, Liu S, Wang Z. A general framework for image fusion based on multi-scale transform and
        sparse representation[J]. Information Fusion, 2015, 24: 147-164.
        :param img1: last image, np.array
        :param img2: next image, np.array
        :return: fused result, np.array
        """
        fused = np.zeros(img1.shape)
        level = 5.0
        if img1.ndim == 2:  # gray mode
            img1_mat = self._np_to_mat(img1)
            img2_mat = self._np_to_mat(img2)
            fused = self.matlab_engine.curvelet_fuse(img1_mat, img2_mat, level)
        else:  # color mode
            for index in range(0, 3):
                img1_mat = self._np_to_mat(img1[:, :, index])
                img2_mat = self._np_to_mat(img2[:, :, index])
                fused[:, :, index] = self.matlab_engine.curvelet_fuse(img1_mat, img2_mat, level)
        fused = self._mat_to_np(fused)
        fused = (fused - fused.min()) / (fused.max() - fused.min())
        return fused

    # ************************************************** SR *****************************************************
    def fuse_by_sr(self, img1, img2):
        """
        SR （Sparse representation）image fusion
        The codes are copied from http://www.escience.cn/people/liuyu1/Codes.html
        B. Yang, S. Li, Multifocus image fusion and restoration with sparse representation,
        IEEE Trans. Instrum. Meas. 59 (4) (2010) 884–892.
        Liu Y, Liu S, Wang Z. A general framework for image fusion based on multi-scale transform and
        sparse representation[J]. Information Fusion, 2015, 24: 147-164.
        :param img1: last image, np.array
        :param img2: next image, np.array
        :return: fused result, np.array
        """
        fused = np.zeros(img1.shape)
        file_path = os.path.dirname(os.path.abspath(__file__))
        dict_address = os.path.join(file_path, '', '../matlab_utils/fusion_methods', 'MST_SR_fusion_toolbox',
                                    'sparsefusion', 'Dictionary', 'D_100000_256_8.mat')
        dict_mat = self._np_to_mat(loadmat(dict_address)['D'])
        overlap = 6
        epsilon = 0.001
        if img1.ndim == 2:
            img1_mat = self._np_to_mat(img1)
            img2_mat = self._np_to_mat(img2)
            fused = self.matlab_engine.sparse_fusion(img1_mat, img2_mat, dict_mat, overlap, epsilon)
        else:  # color mode
            for index in range(0, 3):
                img1_mat = self._np_to_mat(img1[:, :, index])
                img2_mat = self._np_to_mat(img2[:, :, index])
                fused[:, :, index] = self.matlab_engine.sparse_fusion(img1_mat, img2_mat, dict_mat, overlap, epsilon)
        fused = np.clip(self._mat_to_np(fused), 0, 1)
        return fused


def _test_nsct_fusion():
    cwd = os.getcwd()
    img_dir = os.path.join(cwd, '../_test_images')
    img1_path = os.path.join(img_dir, 'rho', '00001.png')
    img2_path = os.path.join(img_dir, 'vis', '00001.png')

    img1 = skimage.io.imread(img1_path)
    img2 = skimage.io.imread(img2_path)
    img1 = img1.astype(np.float64) / 255
    img2 = img2.astype(np.float64) / 255

    polar_fuser = ImageFuserByMatlab()
    fused_image = polar_fuser.fuse_by_nsct(img1, img2)

    plt.figure('test nsct fusion')
    plt.title('test nsct fusion')
    plt.imshow(fused_image)
    plt.show(block=True)


if __name__ == '__main__':
    _test_nsct_fusion()
