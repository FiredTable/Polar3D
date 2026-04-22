import os
import skimage.io
import torch
import unittest
from pathlib import Path

import MyTools.polarization_utils as pu
from MyTools.image_fusion import GuideFilterFuserTensor
from MyTools.common import imnorm, list2tensor


class TestGuideFilterTensor(unittest.TestCase):
    @unittest.skip
    def test1_GuideFilterFusionTensor(self):
        datapath = Path(os.path.join('../..', '_test_images', 'Multifocus'))
        image_names = ['sbug00.png', 'sbug01.png', 'sbug02.png', 'sbug03.png', 'sbug04.png', 'sbug05.png', 'sbug06.png',
                       'sbug07.png', 'sbug08.png', 'sbug09.png', 'sbug10.png', 'sbug11.png', 'sbug12.png']
        images = [skimage.io.imread(datapath / name) for name in image_names]
        images = torch.stack([torch.from_numpy(image).float() for image in images], dim=0).permute(0, 3, 1, 2) / 255.0

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        images = images.to(device)

        gf_fuser = GuideFilterFuserTensor()
        fused_image = gf_fuser.fusion(images, visualize=False)

    @unittest.skip
    def test2_GuideFilterFusionTensor(self):
        dataset_path = Path(os.path.join('../..', '_test_images'))
        image_src_path = Path(os.path.join('dot', 'full_polarization', '2024-12-25_16-06-10'))

        polar_dataset = pu.PolarDatasetManager(dataset_type=pu.DTYPE_DOT_FP)
        polar_dataset.import_source(dataset_path=dataset_path, image_src_path=image_src_path)

        polar_analyser = pu.PolarizationAnalyser(polar_dataset=polar_dataset, resize_ratio=0.5)
        polar_analyser.calc_stokes(visualize=False)
        polar_analyser.calc_polar_features(visualize=True)

        images = [polar_analyser.iun, polar_analyser.rho, imnorm(polar_analyser.phi, mode='min-max')]
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        images_tensor = list2tensor(images).to(device)

        fuser = GuideFilterFuserTensor()
        fusion_image = fuser.fusion(images_tensor, visualize=True)

    def test3_GuideFilterFusionTensor(self):
        dataset_path = Path(os.path.join('../..', '_test_images'))
        image_src_path = Path(os.path.join('dof', '2024-12-27_10-24-25.png'))

        polar_dataset = pu.PolarDatasetManager(dataset_type=pu.DTYPE_DOF_LP)
        polar_dataset.import_source(dataset_path=dataset_path, image_src_path=image_src_path)

        polar_analyser = pu.PolarizationAnalyser(polar_dataset=polar_dataset, resize_ratio=0.25)
        polar_analyser.calc_stokes(visualize=False)
        polar_analyser.calc_polar_features(visualize=True)

        images = [polar_analyser.iun, polar_analyser.rho]
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        images_tensor = list2tensor(images).to(device)

        fuser = GuideFilterFuserTensor()
        fusion_image = fuser.fusion(images_tensor, visualize=True)
