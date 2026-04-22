import matlab
import matlab.engine
import os
import numpy as np


__all__ =['HighlightRemoverByMatlab']


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


class HighlightRemoverByMatlab:
    def __init__(self):
        project_address = os.path.join(os.path.dirname(os.path.abspath(__file__)), '')
        self.matlab_engine = matlab.engine.start_matlab()
        # Add the search dir of matlab
        add_recursive_paths(self.matlab_engine, os.path.join(project_address, '../matlab_utils', 'hr_methods'))

    def __del__(self):
        self.matlab_engine.quit()

    @staticmethod
    def _np_to_mat(img_np):
        """
        Transfer numpy to matlab style
        :param img_np: image, np.array
        :return: matlab style
        """
        img_mat = matlab.double(img_np.tolist())
        return img_mat

    @staticmethod
    def _mat_to_np(img_mat):
        """
        Transfer matlab style to numpy
        :param img_mat: image, matlab style
        :return: np.array
        """
        img_np = np.array(img_mat)
        return img_np

    def hr_by_yang2010real(self, img):
        """
        This method uses a fast bilateralFilter implementation by Jiawen Chen:
        http://people.csail.mit.edu/jiawen/software/bilateralFilter.m
        This method should have equivalent functionality as
 `      qx_highlight_removal_bf.cpp` formerly distributed by the author.
        See also SIHR, Tan2005.

        :param img: color image, [0,1], float64
        """
        img_mat = self._np_to_mat(img)
        hr_img = self.matlab_engine.yang2010real(img_mat)
        hr_img = self._mat_to_np(hr_img)
        return hr_img

    def hr_by_umeyama2004separation(self, images, stokes):
        """
        :param images: polarization images
        :param stokes: polarization stokes
        :return hr_img1: highlight removal image1
        :return hr_img2: highlight removal image2
        """
        images_mat = self._np_to_mat(images)
        stokes_mat = self._np_to_mat(stokes)

        hr_img = self.matlab_engine.umeyama2004separation(images_mat, stokes_mat)
        hr_img = self._mat_to_np(hr_img)

        return hr_img

    def hr_by_wang2017specularity(self, umeyama2004separation_diffuse, stokes):
        umeyama2004separation_diffuse_mat = self._np_to_mat(umeyama2004separation_diffuse)
        stokes_mat = self._np_to_mat(stokes)

        hr_img = self.matlab_engine.wang2017specularity(umeyama2004separation_diffuse_mat, stokes_mat)
        hr_img = self._mat_to_np(hr_img)

        return hr_img
