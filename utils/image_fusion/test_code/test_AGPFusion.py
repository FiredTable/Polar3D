import os
import numpy as np
import unittest
import torch
import skimage.io
import matplotlib.pyplot as plt
from pathlib import Path
from torchvision import models
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam import GradCAMPlusPlus
from ..AGPFusion import AdaptiveLocalVarianceAttentionHead


class TestAGPFusion(unittest.TestCase):
    @unittest.skip("Not implemented yet")
    def test_AdaptiveLocalVarianceAttentionHead(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        img_path = Path(os.path.join('_test_images', 'color', 'ball.png'))
        img = skimage.io.imread(img_path).astype(np.float32) / 255.
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

        head = AdaptiveLocalVarianceAttentionHead(kernel_size=3, threshold_ratio=0.6).to(device)
        attention_scores = head(img_tensor)

    @unittest.skip("Not implemented yet")
    def test_GradCAMAttentionHead(self):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        img_path = Path(os.path.join('_test_images', 'color', 'ball.png'))
        img = skimage.io.imread(img_path).astype(np.float32) / 255.
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT).to(device)

        target_layers = [model.features[-1]]

        input_tensor = torch.concat((img_tensor, img_tensor, img_tensor), dim=0)
        targets = None

        with GradCAMPlusPlus(model=model, target_layers=target_layers) as cam:
            # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
            grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
            # In this example grayscale_cam has only one image in the batch:
            grayscale_cam = grayscale_cam[0, :]
            visualization = show_cam_on_image(img, grayscale_cam, use_rgb=True)
            # You can also get the model outputs without having to redo inference
            model_outputs = cam.outputs

            plt.imshow(visualization)
            plt.show(block=True)
