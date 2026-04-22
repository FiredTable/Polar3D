import os
import glob
import time
import cv2
import torch
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import skimage.measure
from typing import Union
from datetime import datetime
from natsort import natsorted
from torchvision import transforms
from skimage import transform as sk_transform
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from scipy.ndimage import binary_erosion, binary_fill_holes, binary_opening, binary_closing
from skimage.morphology import remove_small_objects 
from tqdm import tqdm

from utils.image_inpaint.utils.canopy_cluster import Canopy
from utils.common import imsave, im_dtype2uint8
from utils.tensor import img_tensor2np
from utils.polarization_analyser import ImageCropper, ClickToGetPixelPosition
from utils.U2Net import uint8_to_sample, RescaleT, ToTensorLab, U2NET, normPRED

__all__ = [
    'MonocularCalibrator', 
    'BinocularCalibrator',
    'BinocularStereoAnalyser',
    'DisparityFuser']


def cv2_imread(file_path, imformat=cv2.IMREAD_UNCHANGED):
    """允许中英文路径的图像读取"""
    # 读取文件为二进制数据
    with open(file_path, 'rb') as f:
        data = f.read()

    # 将二进制数据转为 numpy 数组
    img_array = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(img_array, imformat)

    return img


class MonocularCalibrator:
    def __init__(self, save_dir='runs'):
        self.save_dir = save_dir
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # calibrate_camera
        self.ret = None  # 重投影误差
        self.mtx = None  # 内参矩阵 [[fx,0,cx],[0,fy,cy],[0,0,1]]
        self.dist = None  # 畸变系数
        self.rvecs = None  # 旋转向量, 用于描述相机坐标系和世界坐标系之间的旋转关系
        self.tvecs = None  # 平移向量, 用于描述相机坐标系和世界坐标系之间的平移关系

    def calibrate(self, calib_dir, checkerboard, square_size, save=False):
        """
        :param calib_dir: 标定文件路径
        :param checkerboard: 棋盘格内角点的数量（行，列）
        :param square_size: 棋盘格的尺寸, 单位mm
        """
        Warning('Matlab的图像标定精度远高于OpenCV, 建议使用Matlab进行标定')
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard[1], 0:checkerboard[0]].T.reshape(-1, 2)

        # Create real world coords. Use your metric.
        objp[:, 0] = objp[:, 0] * square_size[1]
        objp[:, 1] = objp[:, 1] * square_size[0]

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        imgpoints = []  # 2d points in image plane.

        extensions = ['*.bmp', '*.png', '*.jpg']
        img_paths = natsorted([file for ext in extensions for file in glob.glob(os.path.join(calib_dir, ext))])
        assert len(img_paths) != 0

        for img_path in img_paths:
            img = cv2_imread(img_path, cv2.IMREAD_GRAYSCALE)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(img, checkerboard[::-1], None)

            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)

                corners2 = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), self.criteria)
                imgpoints.append(corners2)

                if save:
                    # Draw and save the corners
                    disp_img = cv2.drawChessboardCorners(img, checkerboard[::-1], corners2, ret)
                    file_name = os.path.join(self.save_dir, f'{os.path.basename(img_path)}.png')
                    cv2.imwrite(file_name, disp_img)
            else:
                print(f"Chessboard couldn't detected. Image:{img_path}")

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img.shape[::-1], None, None)
        print(f"Monocular calibration RMS error: {ret}")

        self.ret = ret
        self.mtx = mtx
        self.dist = dist
        self.rvecs = rvecs
        self.tvecs = tvecs

    def save_monocular_calibration_results(self, file_name='monocular_calibration_results.yml'):
        """ Save the camera matrix and the distortion coefficients to given path/file. """
        yml_save_path = os.path.join(file_name)

        cv_file = cv2.FileStorage(yml_save_path, cv2.FILE_STORAGE_WRITE)
        cv_file.writeComment('monocular calibration')

        cv_file.writeComment('arguments')
        cv_file.write('calib_dir', self.calib_dir)
        cv_file.write('checkerboard', self.checkerboard)
        cv_file.write('square_size', self.square_size)

        cv_file.writeComment('calibration results')
        cv_file.write('mtx', self.mtx)
        cv_file.write('dist', self.dist)
        # note you *release* you don't close() a FileStorage object
        cv_file.release()

    @staticmethod
    def load_monocular_calibration_results(file_path):
        cv_file = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)
        
        mtx = cv_file.getNode('mtx').mat()
        dist = cv_file.getNode('dist').mat()
        
        cv_file.release()
        return [mtx, dist]
    

class BinocularCalibrator:
    def __init__(self, save_dir='../runs'):
        self.save_dir = save_dir
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)

        # termination criteria
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        # binocular_calibration
        self.img_shape = None
        self.retval = None
        self.mtx1 = None
        self.dist1 = None
        self.mtx2 = None
        self.dist2 = None
        self.R = None
        self.T = None
        self.E = None
        self.F = None
        self.R1 = None
        self.R2 = None
        self.P1 = None
        self.P2 = None
        self.Q = None
        self.roi_left = None
        self.roi_right = None

    def calibrate(self, left_file, right_file, left_dir, right_dir, checkerboard, square_size):
        # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((checkerboard[0] * checkerboard[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:checkerboard[1], 0:checkerboard[0]].T.reshape(-1, 2)

        # Create real world coords. Use your metric.
        objp[:, 0] = objp[:, 0] * square_size[1]
        objp[:, 1] = objp[:, 1] * square_size[0]

        # Arrays to store object points and image points from all the images.
        objpoints = []  # 3d point in real world space
        left_imgpoints = []  # 2d points in image plane.
        right_imgpoints = []  # 2d points in image plane.

        extensions = ['*.bmp', '*.png', '*.jpg']
        left_img_paths = natsorted([file for ext in extensions for file in glob.glob(os.path.join(left_dir, ext))])
        right_img_paths = natsorted([file for ext in extensions for file in glob.glob(os.path.join(right_dir, ext))])

        # Pairs should be same size. Otherwise we have sync problem.
        assert len(left_img_paths) != 0 and len(left_img_paths) == len(right_img_paths)

        for left_img_path, right_img_path in zip(left_img_paths, right_img_paths):
            left_img = cv2_imread(left_img_path, cv2.IMREAD_GRAYSCALE)
            right_img = cv2_imread(right_img_path, cv2.IMREAD_GRAYSCALE)
            
            assert left_img.shape == right_img.shape
            img_shape = left_img.shape

            # Find the chess board corners
            left_ret, left_corners = cv2.findChessboardCorners(
                left_img, checkerboard[::-1], cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS)
            right_ret, right_corners = cv2.findChessboardCorners(
                right_img, checkerboard[::-1], cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FILTER_QUADS)

            if left_ret and right_ret:
                objpoints.append(objp)
                # Left points
                left_corners2 = cv2.cornerSubPix(left_img, left_corners, (5, 5), (-1, -1), self.criteria)
                left_imgpoints.append(left_corners2)
                # Right points
                right_corners2 = cv2.cornerSubPix(right_img, right_corners, (5, 5), (-1, -1), self.criteria)
                right_imgpoints.append(right_corners2)
            else:
                print("Chessboard couldn't detected. Image pair: ", os.path.basename(left_img_path),
                      " and ", os.path.basename(right_img_path))

        mtx1, dist1 = MonocularCalibrator.load_monocular_calibration_results(left_file)
        mtx2, dist2 = MonocularCalibrator.load_monocular_calibration_results(right_file)
        # 确定每台相机的内部参数（如焦距、主点等）和外部参数（如旋转和平移），以及两台相机之间的相对位置和姿态。
        # retval: 重投影误差，即所有图像点的平均重投影误差。这个值越小，表示标定结果越准确。
        # mtx: 相机矩阵。
        # dist: 畸变系数。
        # R: 两台相机之间的旋转矩阵。
        # T: 两台相机之间的平移向量。
        # E: 本质矩阵（Essential Matrix），描述了两台相机之间的相对位置和姿态。
        # F: 基础矩阵（Fundamental Matrix），描述了两台相机之间的几何关系。
        (retval, mtx1, dist1, mtx2, dist2, R, T, E, F) = cv2.stereoCalibrate(
            objpoints, left_imgpoints, right_imgpoints,
            mtx1, dist1, mtx2, dist2, img_shape[::-1])
        print(f"Stereo calibration RMS error: {retval}")
        
        self.img_shape = img_shape
        self.retval = retval
        self.mtx1 = mtx1
        self.dist1 = dist1
        self.mtx2 = mtx2
        self.dist2 = dist2
        self.R = R
        self.T = T
        self.E = E
        self.F = F

        self._perform_stereo_rectification()

    def _perform_stereo_rectification(self):
        # 立体校正：将两幅立体图像对齐，使得每一对对应的特征点位于同一水平线上。
        # R: 相机的旋转矩阵，用于将原始图像转换到校正后的图像。
        # P: 相机的投影矩阵，包含校正后的内参和外参。
        # Q: 重投影矩阵，用于从视差图计算三维点。
        # roi: 相机的有效像素区域，表示校正后图像中有效部分的边界框。
        R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
            self.mtx1, self.dist1, self.mtx2, self.dist2, self.img_shape[::-1], 
            self.R, self.T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)

        self.R1 = R1
        self.R2 = R2
        self.P1 = P1
        self.P2 = P2
        self.Q = Q
        self.roi_left = roi_left
        self.roi_right = roi_right
    
    def save_binocular_calibration_results(self, file_name='binocular_calibration_results.yml'):
        """ Save the stereo coefficients to given path/file. """
        yml_save_path = os.path.join(file_name)

        cv_file = cv2.FileStorage(yml_save_path, cv2.FILE_STORAGE_WRITE)
        cv_file.writeComment('stereo calibration')
        cv_file.writeComment('arguments')
        cv_file.write('img_shape', self.img_shape)

        cv_file.writeComment('calibration results')
        cv_file.write('retval', self.retval)
        cv_file.write('mtx1', self.mtx1)
        cv_file.write('dist1', self.dist1)
        cv_file.write('mtx2', self.mtx2)
        cv_file.write('dist2', self.dist2)
        cv_file.write('R', self.R)
        cv_file.write('T', self.T)
        cv_file.write('E', self.E)
        cv_file.write('F', self.F)
        cv_file.write('R1', self.R1)
        cv_file.write('R2', self.R2)
        cv_file.write('P1', self.P1)
        cv_file.write('P2', self.P2)
        cv_file.write('Q', self.Q)
        # note you *release* you don't close() a FileStorage object
        cv_file.release()

    def load_binocular_calibration_coefficients(self, file_path):
        """ Loads binocular calibration matrix coefficients. """
        # FILE_STORAGE_READ
        cv_file = cv2.FileStorage(file_path, cv2.FILE_STORAGE_READ)

        # note we also have to specify the type to retrieve otherwise we only get a
        # FileNode object back instead of a matrix
        img_shape = cv_file.getNode("img_shape").mat().astype(int)
        self.img_shape = (img_shape[0][0], img_shape[1][0])

        self.mtx1 = cv_file.getNode("mtx1").mat()
        self.dist1 = cv_file.getNode("dist1").mat()
        self.mtx2 = cv_file.getNode("mtx2").mat()
        self.dist2 = cv_file.getNode("dist2").mat()
        self.R = cv_file.getNode("R").mat()
        self.T = cv_file.getNode("T").mat()
        # self.E = cv_file.getNode("E").mat()
        # self.F = cv_file.getNode("F").mat()
        self.R1 = cv_file.getNode("R1").mat()
        self.R2 = cv_file.getNode("R2").mat()
        self.P1 = cv_file.getNode("P1").mat()
        self.P2 = cv_file.getNode("P2").mat()
        self.Q = cv_file.getNode("Q").mat()

        cv_file.release()

    def load_calib_params_from_mat(self, mat_file):
        data = scipy.io.loadmat(mat_file)
        self.retval = data['reprojection_error'][0][0].astype(np.float64)
        self.mtx1 = data['cameraMatrix1'].astype(np.float64)
        self.mtx2 = data['cameraMatrix2'].astype(np.float64)
        self.dist1 = data['distCoeffs1'].astype(np.float64)
        self.dist2 = data['distCoeffs2'].astype(np.float64)
        self.R = data['R'].astype(np.float64)
        self.T = data['T'].astype(np.float64).reshape(3,1)
        self.img_shape = int(data['imageSize'][0][0]), int(data['imageSize'][0][1]) # (height, width)


class BinocularStereoAnalyser():
    def __init__(self, binocular_calibrator:BinocularCalibrator, 
                 u2net_model_path='u2net.pth', save_dir='runs'):
        self.binocular_calibrator = binocular_calibrator

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))

        try:
            self.u2net = U2NET(3,1)
            self.u2net.load_state_dict(torch.load(u2net_model_path))
            self.u2net.cuda()
            self.u2net.eval()
        except:
            print('Failed to load U2NET model.')

        self.left_map_x = None
        self.left_map_y = None
        self.right_map_x = None
        self.right_map_y = None

        # set_sgbm_params
        self.sgbm_params = None
        self.left_matcher = None
        self.right_matcher = None
        
        # calc_disparity_map
        self.mask_l = None
        self.mask_r = None

    @staticmethod
    def depth_to_pointcloud(Z: np.ndarray, output_dimension=3, xy_scale=2):
        height, width = Z.shape[0:2]
        U, V = np.meshgrid(np.arange(width), np.arange(height))
        points_3d = np.stack([U, V, Z], axis=-1)

        x_min, x_max = points_3d[:, :, 0].min(), points_3d[:, :, 0].max()
        y_min, y_max = points_3d[:, :, 1].min(), points_3d[:, :, 1].max()
        z_min, z_max = points_3d[:, :, 2].min(), points_3d[:, :, 2].max()

        # 归一化 X/Y 到 [z_min, z_max]
        points_3d[:, :, 0] = xy_scale * (points_3d[:, :, 0] - x_min) / (x_max - x_min) * (z_max - z_min) + z_min
        points_3d[:, :, 1] = xy_scale * (points_3d[:, :, 1] - y_min) / (y_max - y_min) * (z_max - z_min) + z_min
        points_3d[:, :, 1] = height / width * points_3d[:, :, 1]

        if output_dimension == 3:
            return points_3d  # (H, W, 3)
        elif output_dimension == 2:
            return points_3d.reshape(-1, 3)  # (H*W, 3)

    def calc_undistort_rectify_map(self):
        img_shape = self.binocular_calibrator.img_shape
        mtx1 = self.binocular_calibrator.mtx1
        dist1 = self.binocular_calibrator.dist2
        mtx2 = self.binocular_calibrator.mtx2
        dist2 = self.binocular_calibrator.dist2
        R1 = self.binocular_calibrator.R1
        R2 = self.binocular_calibrator.R2
        P1 = self.binocular_calibrator.P1
        P2 = self.binocular_calibrator.P2

        self.left_map_x, self.left_map_y = cv2.initUndistortRectifyMap(
            mtx1, dist1, R1, P1, img_shape[::-1], cv2.INTER_NEAREST)
        self.right_map_x, self.right_map_y = cv2.initUndistortRectifyMap(
            mtx2, dist2, R2, P2, img_shape[::-1], cv2.INTER_NEAREST)
  
    @staticmethod
    def overlay_mask(img: np.ndarray, mask: np.ndarray, alpha=0.3):
        if len(img.shape) == 3:
            mask_on_img = img.copy()
        else:
            img = np.stack((img,)*3, axis=-1)
            mask_on_img = img.copy()

        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                if mask[i, j]:
                    mask_on_img[i, j, 0] = alpha * img[i, j, 0] + (1 - alpha) * 168
                    mask_on_img[i, j, 1] = alpha * img[i, j, 1] + (1 - alpha) * 215
                    mask_on_img[i, j, 2] = alpha * img[i, j, 2] + (1 - alpha) * 125
                    
        return mask_on_img
    
    def generate_mask_by_u2net(self, img, u2net_thresh=0.1, vis_thresh=10):
        if len(img.shape) == 3:
            img = np.mean(img, axis=-1)
        uint8_image = im_dtype2uint8(img)
        sample = uint8_to_sample(
            uint8_image, transform=transforms.Compose([RescaleT(320), ToTensorLab(flag=0)]))
        
        with torch.no_grad():
            print("inferencing mask")
            inputs_test = sample['image'][None, ...]

            # 将input_test 的数据类型转化为float
            inputs_test = inputs_test.type(torch.FloatTensor)
            inputs_test = inputs_test.cuda()

            st_time = time.time()
            d1, d2, d3, d4, d5, d6, d7= self.u2net(inputs_test)
            print('U2Net Inference time: ', time.time() - st_time)

            # normalization
            pred = d1[:,0,:,:]
            pred = normPRED(pred)

            np_img = img_tensor2np(pred[0])
            resized_img = sk_transform.resize(
                np_img, (uint8_image.shape[0], uint8_image.shape[1]))
            u2net_mask = resized_img > u2net_thresh

        vis_mask = img > vis_thresh

        return u2net_mask & vis_mask

    def create_sgbm_matchers(self, minDisparity=0, numDisparities=64, blockSize=3, disp12MaxDiff=1,
                             preFilterCap=63, uniquenessRatio=15, speckleWindowSize=50, speckleRange=2,
                             mode=cv2.STEREO_SGBM_MODE_HH):
        """
        :param minDisparity: 最小可能的视差值。
        :param numDisparities: 视差搜索范围内的视差值数量。
        :param blockSize: 匹配窗口大小, 必须是奇数. 默认值通常是3或5。
        :param disp12MaxDiff: 检查左右视差一致性时的最大允许差异。默认值通常是1。将其设置为非正值以禁用检查。
        :param preFilterCap: 预过滤的截断值。默认值通常是31。低对比度图像设置为 15-20.
        :param uniquenessRatio: 唯一性检查比率。默认值通常是15。
        :param speckleWindowSize: 用于滤除小噪声区域的窗口大小。默认值通常是0, 表示不进行这种滤波。
        :param speckleRange: 允许的最大视差变化。默认值通常是2。
        :param mode: sgbm算法选择模式, 以速度由快到慢为:
          STEREO_SGBM_MODE_SGBM_3WAY、STEREO_SGBM_MODE_HH4、STEREO_SGBM_MODE_SGBM、STEREO_SGBM_MODE_HH。精度反之
        """
        self.sgbm_params = {
            'minDisparity': minDisparity,
            'numDisparities': numDisparities,
            'blockSize': blockSize,
            'P1': 8 * blockSize * blockSize,
            'P2': 32 * blockSize * blockSize,
            'disp12MaxDiff': disp12MaxDiff,
            'preFilterCap': preFilterCap,  # 它限制了在预过滤过程中允许的最大亮度变化
            'uniquenessRatio': uniquenessRatio,  # 最佳（最小）计算成本函数值应该“赢”第二个最佳值以考虑找到的匹配正确的百分比保证金。
            'speckleWindowSize': speckleWindowSize,
            'speckleRange': speckleRange,
            'mode': mode
        }

        # 创建左视图匹配器
        self.left_matcher = cv2.StereoSGBM_create(**self.sgbm_params)

        # 创建右视图匹配器
        self.right_matcher = cv2.ximgproc.createRightMatcher(self.left_matcher)

    @staticmethod
    def remove_outliers_z_score(data: np.ndarray, threshold=3):
        mean = np.mean(data)
        std_dev = np.std(data)
        z_scores = [(x - mean) / std_dev for x in data]
        filtered_data = [x for x, z in zip(data, z_scores) if abs(z) < threshold]
        return np.array(filtered_data)
    
    def estimate_disparity_by_sift(self, img_l, img_r, mask_l, mask_r, redundancy=1, save=False):
        sift = cv2.SIFT_create()

        if mask_l.dtype == 'bool':
            mask_l = mask_l.astype('uint8') * 255
        
        if mask_r.dtype == 'bool':
            mask_r = mask_r.astype('uint8') * 255

        # 检测特征点并计算描述符
        kp1, des1 = sift.detectAndCompute(img_l, mask=mask_l)
        kp2, des2 = sift.detectAndCompute(img_r, mask=mask_r)

        # 使用 FLANN 匹配器(Fast Library for Approximate Nearest Neighbors)
        FLANN_INDEX_KDTREE = 1  # FLANN_INDEX_KDTREE 表示使用 k-d 树（k-dimensional tree）作为索引结构
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # 指定在搜索过程中检查的叶子节点数。

        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)  # k=2，意味着对于每个左图像的特征点，找到右图像中最相似的两个特征点。

        # 过滤匹配点
        good_matches = []
        for m, n in matches:
            if m.distance < 0.6 * n.distance:
                pt1 = kp1[m.queryIdx].pt
                pt2 = kp2[m.trainIdx].pt
                if abs(pt1[1] - pt2[1]) < 5:  # 确保寻找到的匹配点在3行以内
                    good_matches.append(m)

        # 提取匹配点的坐标
        pts_left = np.float32([kp1[m.queryIdx].pt for m in good_matches])
        pts_right = np.float32([kp2[m.trainIdx].pt for m in good_matches])

        # 计算视差
        disparities = pts_left[:, 0] - pts_right[:, 0]

        # 排除匹配点中的离群值
        disparities = self.remove_outliers_z_score(disparities)

        # 计算视差的最小值和最大值
        min_disparity = int(np.min(disparities))
        max_disparity = int(np.max(disparities))

        print(f"Estimated min disparity: {min_disparity}")
        print(f"Estimated max disparity: {max_disparity}")

        num_disparities = ((max_disparity - min_disparity) // 16 + 1) * 16
        print(f"Estimated numDisparities: {num_disparities}")

        # 可视化匹配点
        if save:
            img_matches = cv2.drawMatches(
                img_l, kp1, img_r, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
            file_name = os.path.join(self.save_dir, f"disparity_range.png")
            cv2.imwrite(file_name, img_matches)

        try:
            self.sgbm_params['minDisparity'] = min_disparity - 8 * redundancy
            self.sgbm_params['numDisparities'] = num_disparities + 16 * redundancy
            print(f"Setting minDisparity to: {self.sgbm_params['minDisparity']}")
            print(f"Setting numDisparities to: {self.sgbm_params['numDisparities']}")
        except Exception as e:
            print(f"Error setting minDisparity and numDisparities: {e}")

    def plot_horizontal_lines_on_img(self, img: np.ndarray, space=20, line_color: tuple = (0, 255, 0), line_thickness=1):
        height, width = img.shape[:2]
        for y in range(0, height, space):
            cv2.line(img, (0, y), (width, y), line_color, line_thickness)
        return img
    
    def undistort(self, img_l, img_r, save=False):
        """利用双目标定参数对双目图像进行畸变校正和极线校正"""
        remap_img_l = cv2.remap(img_l, self.left_map_x, self.left_map_y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        remap_img_r = cv2.remap(img_r, self.right_map_x, self.right_map_y, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT)
        if save:
            disp_remap_img = np.concatenate((remap_img_l, remap_img_r), axis=1)
            disp_remap_img = self.plot_horizontal_lines_on_img(
                disp_remap_img, space=150, line_color=(255, 0, 255), line_thickness=5)
            
            file_name = os.path.join(self.save_dir, f"plot_horizontal_lines_on_img.png")
            cv2.imwrite(file_name, disp_remap_img)

        return remap_img_l, remap_img_r

    def evaluate_disparity_confidence(self, disp_l, disp_r, img_l, img_r):
        """生成多维度可靠性评估掩码 (0-1范围，1=最可靠)"""
        height, width = disp_l.shape
        confidence = np.zeros((height, width), dtype=np.float32)

        disp_r_mapped = np.full_like(disp_l, np.nan)
        img_r_mapped = np.full_like(img_l, np.nan)
        if False:
            for y in range(height):
                for x in range(width):
                    d =  disp_l[y, x]
                    if d <=0:
                        continue
                    
                    x_r = int(round(x-d))
                    if 0 <= x_r < width:
                        disp_r_mapped[y, x] = -disp_r[y, x_r]
        else:
            # 使用向量化操作消除显示循环
            y_grid, x_grid = np.indices((height, width))
            x_r = np.round(x_grid - disp_l).astype(int)
            valid_mask = (disp_l > 0) & (x_r >= 0) & (x_r < width)
            disp_r_mapped[valid_mask] = -disp_r[y_grid[valid_mask], x_r[valid_mask]]
            img_r_mapped[valid_mask] = img_r[y_grid[valid_mask], x_r[valid_mask]]
        
        # 1. 左右一致性检查
        valid_mask = ~np.isnan(disp_r_mapped)
        lrc_diff = np.full_like(disp_l, np.inf)  # 初始化为无穷大
        lrc_diff[valid_mask] = np.abs(disp_l[valid_mask] - disp_r_mapped[valid_mask])
        lrc_score = np.exp(-0.2 * lrc_diff)  # 超出5个视差以上分数小于0.36

        # 2. 光流残差计算
        valid_mask = ~np.isnan(img_r_mapped)
        residual = np.full_like(disp_l, np.inf)
        residual[valid_mask] = np.abs(img_l[valid_mask] - img_r_mapped[valid_mask])
        flow_score = np.exp(-0.025 * residual)  # 超出40个残差以上分数小于0.36

        # 3. 纹理丰富性检测
        texture_map = cv2.blur(cv2.convertScaleAbs(cv2.Laplacian(img_l, cv2.CV_32F)), (5, 5))
        texture_score = cv2.normalize(texture_map, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # 4. 视差连续性检测 (基于二阶导数)
        laplacian = cv2.Laplacian(disp_l, cv2.CV_32F)
        smoothness_score = np.exp(-0.05 * np.abs(laplacian))

        confidence = 0.6 * lrc_score + 0.2 * flow_score + 0.1 * texture_score + 0.1 * smoothness_score
        confidence = cv2.medianBlur(confidence, 3)

        return confidence, {
            'lrc_score': lrc_score,
            'flow_score': flow_score,
            'texture_score': texture_score,
            'smooth_score': smoothness_score
        }

    def calc_disparity_map_by_sgbm(self, remap_img_l, remap_img_r, 
                                   enhance=False, need_mask=True, 
                                   estimate_disparity=True, conf=0.5, save=False,
                                   down_scale=False):
        st_time = time.time()

        if enhance:
            remap_img_l = self.clahe.apply(remap_img_l)
            remap_img_r = self.clahe.apply(remap_img_r)

        if need_mask:
            # 计算校正后图像的前景掩膜
            self.mask_l = self.generate_mask_by_u2net(remap_img_l, u2net_thresh=0.1, vis_thresh=0.1)
            self.mask_r = self.generate_mask_by_u2net(remap_img_r, u2net_thresh=0.1, vis_thresh=0.1)
            if save:
                overlay_l = self.overlay_mask(remap_img_l, self.mask_l, alpha=0.3)
                overlay_r = self.overlay_mask(remap_img_r, self.mask_r, alpha=0.3)
                imsave(overlay_l, 'overlay_l.png', self.save_dir)
                imsave(overlay_r, 'overlay_r.png', self.save_dir)
        else:
            self.mask_l = np.ones_like(remap_img_l, dtype=bool)
            self.mask_r = np.ones_like(remap_img_r, dtype=bool)

        # 估计视察范围
        st_time_sift = time.time()
        if estimate_disparity:
            self.estimate_disparity_by_sift(remap_img_l, remap_img_r, self.mask_l, self.mask_r, redundancy=1, save=save)
        sift_time = time.time() - st_time_sift

        # 计算左右图视差
        st_time_match = time.time()

        if down_scale == False:
            disparity_l = self.left_matcher.compute(remap_img_l, remap_img_r)
            disparity_r = self.right_matcher.compute(remap_img_r, remap_img_l)
        else:
            left_image_down = cv2.pyrDown(remap_img_l)
            right_image_down = cv2.pyrDown(remap_img_r)
            factor = remap_img_l.shape[1] / left_image_down.shape[1]

            disparity_left_half = self.left_matcher.compute(left_image_down, right_image_down)
            disparity_right_half = self.right_matcher.compute(right_image_down, left_image_down)

            disparity_l = cv2.pyrUp(disparity_left_half) * factor
            disparity_r = cv2.pyrUp(disparity_right_half) * factor

        match_time = time.time() - st_time_match

        # WLS滤波
        st_time_wls = time.time()
        wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=self.left_matcher)
        wls_filter.setLambda(8000)  # 平滑系数
        wls_filter.setSigmaColor(0.5)  # 颜色相似度权重
        disparity_l = wls_filter.filter(disparity_l, remap_img_l, None, disparity_r)
        wls_time = time.time() - st_time_wls

        # 中值滤波
        if np.any(np.isnan(disparity_l)):
            print(f'nan found in disparity_l, replace with 0')
            disparity_l = np.nan_to_num(disparity_l, nan=0.0)
        if np.any(np.isinf(disparity_l)):
            print(f'inf found in disparity_l, replace with 0')
            disparity_l = np.where(np.isinf(disparity_l), 0, disparity_l)
        disparity_l = cv2.medianBlur(disparity_l.astype(np.float32), 5)
        
        # OpenCV 为了提高计算效率，将视差值乘以了一个固定的缩放因子 16，然后以整数形式存储
        disparity_l = disparity_l.astype(np.float32) / 16.
        disparity_r = disparity_r.astype(np.float32) / 16.
        disparity_l[self.mask_l == False] = np.min(disparity_l[self.mask_l])

        # 生成可靠性掩膜
        st_time_conf = time.time()
        confidence_score, scores = self.evaluate_disparity_confidence(disparity_l, disparity_r, remap_img_l, remap_img_r)
        high_confidence_mask = confidence_score > conf
        high_confidence_mask[self.mask_l == False] = 0
        conf_time = time.time() - st_time_conf

        if save:
            imsave(disparity_l, 'bino_disparity.png', self.save_dir, cmap_name='jet', norm=True)
            imsave(scores['lrc_score'], 'lrc_score.png', self.save_dir, cmap_name='jet', norm=True)
            imsave(scores['texture_score'], 'texture_score.png', self.save_dir, cmap_name='jet', norm=True)
            imsave(scores['smooth_score'], 'smooth_score.png', self.save_dir, cmap_name='jet', norm=True)
            imsave(confidence_score, 'confidence_score.png', self.save_dir, cmap_name='jet', norm=True)
            imsave(high_confidence_mask, 'high_confidence_mask.png', self.save_dir, norm=True)
            mask_on_img = self.overlay_mask(remap_img_l, high_confidence_mask)
            imsave(mask_on_img, 'mask_on_img.png', self.save_dir)

        print(f'cost time is: {time.time() - st_time:.2f} s, '
              f'cost time of sift is: {sift_time:.2f} s, '
              f'cost time of match is: {match_time:.2f} s, '
              f'cost time of wls is: {wls_time:.2f} s, '
              f'cost time of conf is: {conf_time:.2f} s')

        return disparity_l, confidence_score

    def disparity2xyz_points(self, disparity_map):
        return cv2.reprojectImageTo3D(disparity_map, self.binocular_calibrator.Q)


class DisparityFuser:
    def __init__(self, disparity_maps, mask):
        self.disparity_maps = disparity_maps
        self.mask = mask
        self.fused_disparity_map = None

    def disparity_fusion(self, threshold):
        disparity_maps = self.disparity_maps
        mask = self.mask

        st_time = time.time()
        idx = np.where(mask != 0)
        fused_disparity_map = np.mean(disparity_maps, axis=2)
        for k in range(len(idx[0])):
            if k % int(0.1 * len(idx[0])) == 0:
                print(f'disparity_fusion processing {k/len(idx[0])*100:.1f}% ,'
                      f' cost time is: {time.time() - st_time:.2f} s')

            disparities = disparity_maps[idx[0][k], idx[1][k], :]
            filtered_disparities = BinocularStereoAnalyser.remove_outliers_z_score(disparities, threshold=threshold)
            fused_disparity_map[idx[0][k], idx[1][k]] = np.mean(filtered_disparities)

        self.fused_disparity_map = fused_disparity_map

    def click_to_show_disparity_points(self):
        """the shape of disparity maps is [H,W,C]"""
        disparity_maps = self.disparity_maps
        fused_disparity_map = self.fused_disparity_map

        clicker = ClickToGetPixelPosition(np.mean(disparity_maps, axis=2))
        values = disparity_maps[clicker.coord_y, clicker.coord_x, :]
        plt.scatter(np.arange(len(values)), values, label='scatter')
        plt.axhline(y=np.mean(values), color='r', linestyle='--', label=f'Mean: {np.mean(values):.2f}')

        if fused_disparity_map is not None:
            fusion = fused_disparity_map[clicker.coord_y, clicker.coord_x]
            plt.axhline(y=fusion, color='g', linestyle='--', label=f'Fusion: {fusion:.2f}')

        plt.legend()
        plt.show(block=True)


class PiecewiseLinearRegression:
    def __init__(self, points):
        self.points = points 
        self.piecewise_model = [] 

        self.generate_piecewise_model()
    
    def generate_piecewise_model(self):
        """根据点构造分段线性连续函数"""
        points = self.points

        for k in range(len(points)-1):
            model = LinearRegression()
            x = np.array((points[k][0], points[k+1][0]))
            y = np.array((points[k][1], points[k+1][1]))
            model.fit(x.reshape(-1,1), y)
            print(f'线段{k}的斜率为: {model.coef_[0]:.2f}，截距为: {model.intercept_:.2f}')
            self.piecewise_model.append(model)

    def predict(self, X: np.ndarray):
        X = np.array(X).reshape(-1, 1)
        y_pred = np.zeros(len(X))

        for k in range(len(self.piecewise_model)):
            if k == 0:
                mask = X < self.points[1][0]
                if mask.any():
                    y_pred[mask.flatten()] = self.piecewise_model[0].predict(X[mask].reshape(-1, 1))
            elif k == len(self.piecewise_model) - 1:
                mask = X >= self.points[-2][0]
                if mask.any():
                    y_pred[mask.flatten()] = self.piecewise_model[-1].predict(X[mask].reshape(-1, 1))
            else:
                mask = (X >= self.points[k][0]) & (X < self.points[k+1][0])
                if mask.any():
                    y_pred[mask.flatten()] = self.piecewise_model[k].predict(X[mask].reshape(-1, 1))
        
        return y_pred
    

class DisparityFuser:
    def __init__(self, binocular_calibrator: BinocularCalibrator, save_dir=None):
        self.binocular_calibrator = binocular_calibrator
        self.save_dir = save_dir
    
    def get_sample_points_from_mask(self, mask, num_samples=2000):
        # 获取掩码区域内所有点的坐标 [N, 2]，格式 [y, x]
        points = np.argwhere(mask > 0)

        # 如果点的数量小于所需的样本数量，则抛出异常
        if len(points) < num_samples:
            raise ValueError("Mask contains fewer points than requested samples")
        
        # 随机选择索引 (无放回)
        indices = np.random.choice(len(points), size=num_samples, replace=False)
        sampled_points = points[indices]

        return sampled_points
    
    def _get_mapping_model(self, bino_disparity:np.ndarray, monocular_depth:np.ndarray, 
                                 label_mask: np.ndarray, iters=2, segments=3):
        """得到视差映射模型"""
        # 随机选取可靠的样本点
        sampled_num = round(np.sum(label_mask > 0) / 32)
        sampled_points = self.get_sample_points_from_mask(label_mask, num_samples=sampled_num)
        sampled_mono = monocular_depth[sampled_points[:, 0], sampled_points[:, 1]]
        sampled_bino = bino_disparity[sampled_points[:, 0], sampled_points[:, 1]]

        # 初始化数据（原始点）
        current_x = sampled_mono.copy()
        current_y = sampled_bino.copy()
        
        best_model = None
        best_score = -np.inf

        for _ in range(iters):
            if segments > 1:
                # 分段线性拟合
                percentiles = np.linspace(0, 100, segments + 1).astype(int)
                thresholds = np.percentile(current_x, percentiles[1:-1]).tolist()

                masks = []
                for k in range(segments):
                    if k == 0:
                        masks.append(current_x < thresholds[0])
                    elif k == segments - 1:
                        masks.append(current_x >= thresholds[-1])
                    else:
                        masks.append((current_x >= thresholds[k-1]) & (current_x < thresholds[k]))
                
                models = []
                for mask in masks:
                    model = LinearRegression()
                    model.fit(current_x[mask].reshape(-1,1), current_y[mask])
                    models.append(model)

                # 得到用于创建分段线性模型的交叉点
                crossing_points = []
                for k in range(len(masks)):
                    if k == 0:
                        x_seg = current_x[masks[k]]
                        x_seg_mean = np.mean(x_seg)
                        y_pred = models[k].predict(x_seg_mean.reshape(-1, 1))
                        crossing_points.append((x_seg_mean.item(), y_pred.item()))
                    elif k == len(masks) - 1:
                        x_seg = current_x[masks[k]]
                        x_seg_mean = np.mean(x_seg)
                        y_pred = models[k].predict(x_seg_mean.reshape(-1, 1))
                        crossing_points.append((x_seg_mean.item(), y_pred.item()))
                    else:
                        x_end = np.array(thresholds[k])
                        y_pred1 = models[k].predict(x_end.reshape(-1, 1))
                        y_pred2 = models[k+1].predict(x_end.reshape(-1, 1))
                        crossing_points.append((x_end.item(), ((y_pred1 + y_pred2) / 2).item()))

                # 创建分段模型
                model = PiecewiseLinearRegression(crossing_points)
            else:
                # 线性拟合
                model = LinearRegression()
                model.fit(current_x.reshape(-1, 1), current_y)

            # 评估分段模型
            y_pred = model.predict(current_x.reshape(-1, 1))
            residuals = current_y - y_pred
            sigma = np.std(residuals)
            r2 = 1 - np.var(residuals) / np.var(current_y)  # 拟合优度，越接近1表示模型解释变异性的能力越强。

            # 更新最佳模型
            if r2 > best_score:
                best_score = r2
                best_model = model
                print(f"迭代更新: 拟合优度R²={r2:.4f}")

            # 移除离群点（3σ原则）
            outlier_mask = np.abs(residuals) > 3 * sigma
            current_x = current_x[~outlier_mask.flatten()]
            current_y = current_y[~outlier_mask.flatten()]

        return best_model, [sampled_mono, sampled_bino, current_x, current_y], best_score

    def _get_global_mapping_model(self, bino_disparity:np.ndarray, monocular_depth:np.ndarray,
                                  label_image:np.ndarray, mask:np.ndarray, iters=2, segments=3, save=False, score_thresh=0.6):
        """得到全局深度映射模型"""
        labels_num = int(np.max(label_image) + 1)
        global_models = {}
        time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        os.makedirs(os.path.join(self.save_dir, 'fitting_results', f'{time_str}'), exist_ok=True)
        for k in range(1, labels_num):
            label_mask = (label_image == k) & mask
            print(f'label{k} 的有效点是 {np.sum(label_mask)}')
            if np.sum(label_mask) < 500:
                print(f'label{k} 有效点数量太少，略过')
                global_models[f'label{k}'] = None
                continue
            best_model, [sampled_mono, sampled_bino, current_x, current_y], best_score = self._get_mapping_model(
                bino_disparity, monocular_depth, label_mask, iters, segments)

            if save:
                # 绘制散点图和拟合曲线
                mono_depth_range = np.linspace(current_x.min(), current_x.max(), 500).reshape(-1,1)
                pred_bino_depth = best_model.predict(mono_depth_range)

                plt.figure(figsize=(10, 7))
                plt.scatter(sampled_mono, sampled_bino, c='green', alpha=0.5, s=10, label='sampled points')
                plt.scatter(current_x, current_y, c='blue', alpha=0.5, s=10, label='cleaned points')
                plt.plot(mono_depth_range, pred_bino_depth, 'r-', linewidth=2, label='fitting curve')
                plt.xlabel('x1', fontsize=12)
                plt.ylabel('x2', fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.text(0.05, 0.95, f'R²={best_score:.4f}', transform=plt.gca().transAxes, fontsize=12, verticalalignment='top')
                plt.savefig(os.path.join(self.save_dir, 'fitting_results', f'{time_str}', f'depth_mapping_{k}.png'), dpi=300)
                plt.close()
            if best_score > score_thresh:
                global_models[f'label{k}'] = best_model
            else:
                global_models[f'label{k}'] = None

        valid_label_mask = np.zeros_like(label_image)
        for k in range(1, labels_num):
            if global_models[f'label{k}'] is not None:
                valid_label_mask[label_image == k] = k

        if save:
            imsave(valid_label_mask, 'valid_label_mask.png', self.save_dir, norm=True, cmap_name='turbo')

        return global_models, valid_label_mask
    
    def _apply_calibration(self, monocular_depth:np.ndarray, label_image: np.ndarray, mapping_model: tuple, save):
        """应用校准模型到整个单目深度图"""
        # 获取原始深度值
        original_shape = monocular_depth.shape
        calibrated_depths = monocular_depth.flatten()

        # 根据每个聚类进行预测
        labels_num = int(np.max(label_image) + 1)
        for k in range(1, labels_num):
            mask = (label_image == k)
            if np.sum(mask) == 0:
                continue
            mono_flat = monocular_depth[mask]
            model: Union[PiecewiseLinearRegression, LinearRegression] = mapping_model[f'label{k}']
            if model is None:
                continue
            calibrated_depths[mask.flatten()] =  model.predict(mono_flat.reshape(-1, 1))
        
        calibrated_depths = calibrated_depths.reshape(original_shape)
        calibrated_depths[label_image == 0] = np.min(calibrated_depths[label_image > 0])
        if save:
            imsave(calibrated_depths, 'mono_depth_calibrated.png', self.save_dir, norm=True, cmap_name='turbo') 

        return calibrated_depths
    
    @staticmethod
    def _canopy_cluster(dataset, canopy_t1, canopy_t2):
        """使用canopy聚类方法对数据进行聚类分析"""
        canopy = Canopy(dataset)
        canopy.setThreshold(canopy_t1, canopy_t2)
        canopy.canopies = canopy.clustering()

        canopy_centers = np.array([sublist[0] for sublist in canopy.canopies])
        return canopy_centers
    
    @staticmethod
    def _kmeans_cluster(dataset, init_centers):
        """使用kmeans聚类方法对数据进行聚类分析"""
        kmeans_model = KMeans(n_clusters=init_centers.shape[0], init=init_centers)
        kmeans_model.fit(dataset)
        return kmeans_model.cluster_centers_, kmeans_model.labels_

    def cluster(self, src:np.ndarray, n_clusters=8, save=False):
        """采用Canopy+Kmeans进行自适应聚类"""
        height, width = src.shape[0:2]
        dataset = src.reshape(height * width, -1)
        if False:
            canopy_centers = self._canopy_cluster(dataset, t1, t2)
            kmeans_centers, labels = self._kmeans_cluster(dataset, canopy_centers)
        else:
            kmeans_model = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
            kmeans_model.fit_predict(dataset)
            labels = kmeans_model.labels_
        cluster_image = labels.reshape(height, width)

        if save:
            plt.figure()
            plt.imshow(cluster_image, cmap='turbo')
            plt.colorbar()
            plt.savefig(os.path.join(self.save_dir, 'cluster_image_.png'), dpi=300)
            plt.close()
        return cluster_image

    @staticmethod
    def show_anns(anns, borders=True):
        if len(anns) == 0:
            return
        sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
        ax = plt.gca()
        ax.set_autoscale_on(False)

        img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
        img[:, :, 3] = 0
        for ann in sorted_anns:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.5]])
            img[m] = color_mask 
            if borders:
                import cv2
                contours, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
                # Try to smooth contours
                contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
                cv2.drawContours(img, contours, -1, (0, 0, 1, 0.4), thickness=1) 

        ax.imshow(img)
        
    def segment_by_sam2(self, img, mask_generator, min_area=1000, save=False):
        """使用SAM2进行图像分割"""
        original_shape = img.shape
        img = cv2.resize(img, None, fx=0.25, fy=0.25)
        masks = mask_generator.generate(img)
        if save:
            plt.figure()
            plt.imshow(img)
            self.show_anns(masks)
            plt.axis('off')
            plt.savefig(os.path.join(self.save_dir, 'sam2.png'), dpi=300)
            plt.close()

        sorted_masks = sorted(masks, key=(lambda x: x['area']), reverse=True)
        label_image = np.full(img.shape[:2], fill_value=0, dtype=np.uint8)
        for k, mask in enumerate(sorted_masks):
            m = mask['segmentation']

            m = binary_erosion(m, iterations=3)
            m = remove_small_objects(m, min_area)
            m = binary_fill_holes(m)
            m = binary_closing(m, iterations=2)

            label_image[m] = k + 1

        label_image = cv2.resize(label_image, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_NEAREST)
        imsave(label_image, 'label_image.png', self.save_dir, cmap_name='turbo', norm=True) if save else None

        return label_image

    def label_finetune(self, label_image, min_area=1000, iterations=1):
        """对分割结果进行后处理"""
        old_label_image = label_image.copy()
        
        for _ in tqdm(range(iterations)):  
            new_labels_count = 0      
            new_label_image = np.zeros_like(label_image, dtype=np.uint8)

            labels_num = np.max(old_label_image)
            for k in range(1, labels_num + 1):
                m = label_image == k

                m = remove_small_objects(m, min_area)
                m = binary_opening(m, iterations=2)
                m = binary_closing(m, iterations=2)
                
                # 将m划分为多个联通区域
                m = skimage.measure.label(m, connectivity=2, background=0)
                for i in range(1, np.max(m) + 1):
                    m_i = m == i
                    new_labels_count += 1
                    new_label_image[m_i] = new_labels_count

            old_label_image = new_label_image.copy()    
        
        if True:
            for k in range(1, new_labels_count + 1):
                imsave(new_label_image==k, f'mask_{k}.png', os.path.join(self.save_dir, 'sam2_masks'))

        return new_label_image

    def calibrate_monocular_disparity(self, monocular_depth, binocular_disparity, 
                                      label_image=None, mask=None, save=False, score_thresh=0.6):
        """基于双目视差图像(可靠区域)和单目深度图像校准单目视差"""
        if label_image is None:
            label_image = np.zeros_like(monocular_depth, dtype=np.uint8)

        if mask is None:
            mask = np.ones_like(monocular_depth, dtype=np.uint8)
        
        # 1. 对每个聚类建立深度映射模型
        mapping_model, valid_label_mask = self._get_global_mapping_model(
            binocular_disparity, monocular_depth, label_image, mask, iters=2, segments=1, save=False, score_thresh=score_thresh)
    
        # 2. 应用校准模型到整个单目深度图
        calibrated_mono_disparity = self._apply_calibration(monocular_depth, valid_label_mask, mapping_model, save=save)

        return calibrated_mono_disparity, valid_label_mask
    
    def calc_calibrated_error(self, monocular_depth, binocular_depth, valid_mask):
        calibrated_err = np.abs(monocular_depth - binocular_depth) * valid_mask
        avg_calibrated_err = np.sum(calibrated_err) / np.sum(valid_mask)
        print(f'平均误差：{avg_calibrated_err}')
        return avg_calibrated_err

    def decompose_image(self, img: np.ndarray, method: str):
        if method == 'avg_pool':
            base_layer = cv2.blur(img, (31, 31))
        else:
            raise ValueError(f'Invalid decomposition method: {method}')

        detail_layer = img - base_layer
        return base_layer, detail_layer
    
    def fuse_disparities(self, mono_disparity, bino_disparity, conf_score=None, conf_thresh=0.6, save=False):
        if conf_score is None:
            conf_score = np.ones_like(mono_disparity, dtype=np.float32)
        
        weight = conf_score.copy()
        weight[conf_score < conf_thresh] = 0
        fused_disparity = (1 - weight) * mono_disparity + weight * bino_disparity

        guide_image = mono_disparity.astype(np.float32)
        fused_disparity = cv2.ximgproc.guidedFilter(
            guide=guide_image,
            src=fused_disparity.astype(np.float32),
            radius=5,
            eps=0.01,
            dDepth=-1)
        fused_disparity = cv2.medianBlur(fused_disparity, 3)
        if save:
            imsave(fused_disparity, 'fused_disparity.png', self.save_dir, norm=True, cmap_name='turbo')
        
        return fused_disparity