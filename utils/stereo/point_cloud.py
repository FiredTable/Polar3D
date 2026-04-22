import open3d as o3d
import numpy as np
import os
from typing import Optional
import matplotlib.pyplot as plt

__all__ = [
    'PointCloud'
]


class PointCloud:
    #TODO: 点云图像预处理：点云采样（体素采样/均匀采样）、
    # 点云离群点移除（基于统计、基于密度、基于聚类的DBSCAN离群点移除算法）、点云滤波（高斯滤波、平均滤波以及拉普拉斯滤波）
    # 点云表面重建功能：支持MLS、泊松算法、贪婪投影三角化算法以及本文提出的改进贪婪投影三角化算法在内的8种表面重建算法
    def __init__(self):
        self.xyz_points = None
        self.mask = None
        self.pcd = None

    def create_from_points(self, xyz_points, mask: Optional[np.ndarray] = None):
        self.xyz_points = xyz_points
        if mask is None:
            self.mask = np.ones_like(xyz_points[:, 0], dtype=bool)
        else:
            self.mask = mask.reshape(-1)

        # create point cloud
        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(xyz_points[self.mask != 0])

    def save_pcd(self, file_name='output.ply', save_dir='.'):
        o3d.io.write_point_cloud(os.path.join(save_dir, file_name), self.pcd)

    def colorize(self, color_img:np.ndarray):
        """
        Colorize the point cloud with the given color image.
        :param color_img: Color image float in range [0, 1].
        """
        if color_img.dtype == np.uint8:
            color_img = color_img.astype(np.float32) / 255.0
        colors = color_img.reshape(-1, 3)
        self.pcd.colors = o3d.utility.Vector3dVector(colors[self.mask != 0])

    # 离群点滤波：
    def remove_statistical_outlier(self, nb_neighbors=20, std_ratio=2.0, visualize=False):
        """
        Apply statistical outlier removal to the point cloud.
        :param nb_neighbors: Number of neighbors around the target point.
        :param std_ratio: Standard deviation ratio to determine the threshold.
        :return: Filtered point cloud.
        """
        cl, ind = self.pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
        print(f'remove_statistical_outlier: {len(self.pcd.points) - len(cl.points)}')

        if visualize:
            o3d.visualization.draw_geometries([cl])

        self.pcd = cl

    def remove_radius_outlier(self, nb_points=16, radius=0.1, visualize=False):
        """半径滤波器通过计算每个点在其邻域内的点数，并去除那些邻域内点数少于某个阈值的点。"""
        cl, ind = self.pcd.remove_radius_outlier(nb_points=nb_points, radius=radius)
        print(f'remove_statistical_outlier: {len(self.pcd.points) - len(cl.points)}')

        if visualize:
            o3d.visualization.draw_geometries([cl])

        self.pcd = cl

    # 降采样：
    def voxel_down_sample(self, voxel_size=0.05):
        """Voxel Grid 滤波器通过将点云划分成体素（voxels），并将每个体素内的点平均化，从而减少点云的密度。"""
        filtered_pcd = self.pcd.voxel_down_sample(voxel_size=voxel_size)
        print(f'voxel_down_sample: {len(filtered_pcd.points)}')
        self.pcd = filtered_pcd
    
    def uniform_down_sample(self, every_k_points=5):
        """均匀采样通过每隔一定数量的点选择一个点，从而减少点云的密度。"""
        filtered_pcd = self.pcd.uniform_down_sample(every_k_points=every_k_points)
        print(f'uniform_down_sample: {len(filtered_pcd.points)}')
        self.pcd = filtered_pcd

    # 点云分割：
    def dbscan_cluster(self, eps=0.2, min_points=10, print_progress=False):
        """DBSCAN（Density-Based Spatial Clustering of Applications with Noise）算法是一种基于密度的聚类算法，
        它将具有足够高密度的区域划分为簇，并标记噪声点。"""
        labels = np.array(self.pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=print_progress))
        print(f'DBScan found {len(set(labels)) - (1 if -1 in labels else 0)} clusters')
        max_label = labels.max()
        colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        self.pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
        o3d.visualization.draw_geometries([self.pcd],
                                  zoom=0.455,
                                  front=[-0.4999, -0.1659, -0.8499],
                                  lookat=[2.1813, 2.0619, 2.0999],
                                  up=[0.1204, -0.9852, 0.1215])
        
    @staticmethod
    def point_to_plane_distance(points, plane_model):
        """点到平面距离计算函数"""
        a, b, c, d = plane_model
        x, y, z = points[:, 0], points[:, 1], points[:, 2]
        numerator = np.abs(a * x + b * y + c * z + d)
        denominator = np.sqrt(a ** 2 + b ** 2 + c ** 2)
        return numerator / denominator

    def ransac_plane_fitting(self, distance_threshold=0.5, ransac_n=100, num_iterations=100, max_planes=100):
        """
        param distance_threshold: 内点的最大允许距离。如果一个点到拟合平面的距离小于这个阈值，则该点被认为是内点。
        param rasac_n: 每次迭代中随机选择的点数。通常为 3，因为三个点可以唯一确定一个平面。
        param num_iterations: RANSAC 算法的迭代次数。更多的迭代次数可以提高拟合的准确性，但会增加计算时间。
        """
        current_pcd = self.pcd
        planes = []  # 用于存储拟合的平面

        for k in range(max_planes):
            if len(current_pcd.points) < ransac_n:
                break

            # 使用 RANSAC 算法进行平面拟合
            plane_model, inliers = current_pcd.segment_plane(
                distance_threshold=distance_threshold, ransac_n=ransac_n, num_iterations=num_iterations)

            if len(inliers) == 0:
                break

            # 输出平面模型的参数
            [a, b, c, d] = plane_model
            print(f"Plane equation {k}: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

            # 寻找所有点云中符合拟合平面的点
            all_points = np.asarray(current_pcd.points)
            distances = PointCloud.point_to_plane_distance(all_points, plane_model)
            all_inliers = np.where(distances < distance_threshold)[0]

            # 更新内点云和外点云
            inlier_cloud = current_pcd.select_by_index(all_inliers)
            outlier_cloud = current_pcd.select_by_index(all_inliers, invert=True)

            # 将拟合的平面和内点存储起来
            planes.append((plane_model, inlier_cloud))

            # 更新剩余点云
            current_pcd = outlier_cloud

        num_planes = len(planes)
        cmap = plt.get_cmap('viridis')  # 选择一个颜色映射
        colors = [cmap(i / num_planes)[:3] for i in range(num_planes)]

        # 可视化结果
        for i, (plane_model, inlier_cloud) in enumerate(planes):
            inlier_cloud.paint_uniform_color(colors[i])

        o3d.visualization.draw_geometries([cloud for _, cloud in planes])
    
    # 后处理:
    def create_from_point_cloud_poisson(self, depth=9):
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(self.pcd, depth=9)[0]
        mesh.fill_holes()
        o3d.visualization.draw_geometries([mesh])

    def estimate_normals(self, radius=0.1, max_nn=30):
        """
        search_param 是一个搜索参数对象，用于指定在估计法线时使用的搜索策略。Open3D 提供了几种不同的搜索参数类型，
        其中 KDTreeSearchParamHybrid 是一种混合搜索策略，结合了半径搜索和最近邻搜索。
        :param radius: 定义了一个球形区域的半径，在这个区域内寻找邻近点。这个值应该根据你的点云密度来适当设置。如果半径太小，可能找不到足够的邻近点；如果半径太大，则可能会引入过多的噪声或不相关的点。
        :param max_nn: 最大邻近点数。即使在指定的半径内有更多可用的邻近点，也只考虑前max_nn个最近的点。
        """
        self.pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=max_nn))
        o3d.visualization.draw_geometries([self.pcd], point_show_normal=True)

    def create_from_point_cloud_ball_pivoting(self, radii=[0.1, 0.2, 0.4, 1.0]):
        pcd = self.pcd
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(pcd, o3d.utility.DoubleVector(radii))
        o3d.visualization.draw_geometries([mesh])

