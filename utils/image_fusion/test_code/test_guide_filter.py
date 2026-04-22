import os
import unittest
import skimage
import torch
import numpy as np

from pathlib import Path
from MyTools.image_fusion import GuideFilterFuser
from MyTools.common import imnorm, tensor_imshow, list2bhwc


def list2tensor(images: list):
    images_np = np.array(images)
    images_tensor = torch.from_numpy(images_np).unsqueeze(1)
    return images_tensor


class TestGuideFilter(unittest.TestCase):
    def test_guided_filter_fuser(self):
        datapath = Path(os.path.join('../..', '_test_images', 'Multifocus'))
        image_names = ['sbug00.png', 'sbug01.png', 'sbug02.png', 'sbug03.png', 'sbug04.png', 'sbug05.png', 'sbug06.png',
                       'sbug07.png', 'sbug08.png', 'sbug09.png', 'sbug10.png', 'sbug11.png', 'sbug12.png']
        images = [skimage.io.imread(datapath / name) for name in image_names]

        # Image fusion
        # It takes a few seconds
        images = list2bhwc(images)

        fuser = GuideFilterFuser()
        fused_image = fuser.fusion(images, norm_mode=None, visualize=False)

