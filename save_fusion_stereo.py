import sys
sys.path.append('core_rt')
import cv2
import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from core_fusion_stereo.rt_igev_stereo import PFIGEVStereo
from core_stereo_rt.utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import os
import skimage.io
import torch.nn.functional as F
from collections import OrderedDict
from core_stereo_rt.utils.frame_utils import read_gen, readDispMiddlebury


DEVICE = 'cuda'

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)

    if len(img.shape) == 2:
        img = np.tile(img[...,None], (1, 1, 3))
    else:
        img = img[..., :3]

    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def demo(args):
    """用于生成kitti测试数据"""
    model = torch.nn.DataParallel(PFIGEVStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory_disp = Path(os.path.join(args.output_directory, 'disp'))
    output_directory_disp.mkdir(exist_ok=True, parents=True)

    output_directory_fused_left = Path(os.path.join(args.output_directory, 'fused_left'))
    output_directory_fused_left.mkdir(exist_ok=True, parents=True)

    output_directory_fused_right = Path(os.path.join(args.output_directory, 'fused_right'))
    output_directory_fused_right.mkdir(exist_ok=True, parents=True)

    with torch.no_grad():
        left_imgs_vis = sorted(glob.glob(args.left_imgs_vis, recursive=True))
        right_imgs_vis = sorted(glob.glob(args.right_imgs_vis, recursive=True))
        left_imgs_pol = sorted(glob.glob(args.left_imgs_pol, recursive=True))
        right_imgs_pol = sorted(glob.glob(args.right_imgs_pol, recursive=True))
        print(f"Found {len(left_imgs_vis)} images.")

        for (left_imfile_vis, left_imfile_pol, right_imfile_vis, right_imfile_pol) in tqdm(list(zip(left_imgs_vis, left_imgs_pol, right_imgs_vis, right_imgs_pol))):
            left_image_vis = load_image(left_imfile_vis)
            left_image_pol = load_image(left_imfile_pol)
            right_image_vis = load_image(right_imfile_vis)
            right_image_pol = load_image(right_imfile_pol)

            padder = InputPadder(left_image_vis.shape, divis_by=32)
            left_image_vis, left_image_pol, right_image_vis, right_image_pol = \
                padder.pad(left_image_vis, left_image_pol, right_image_vis, right_image_pol)
            
            disp, fused_left, fused_right = model(
                left_image_vis, right_image_vis, left_image_pol, right_image_pol, iters=args.valid_iters, test_mode=True)
            disp = padder.unpad(disp)
            fused_left = padder.unpad(fused_left)
            fused_right = padder.unpad(fused_right)

            disp_file_stem = os.path.join(output_directory_disp, left_imfile_vis.split('/')[-1])
            disp = disp.cpu().numpy().squeeze()

            fused_left_file_stem = os.path.join(output_directory_fused_left, left_imfile_vis.split('/')[-1])
            fused_left = fused_left.cpu().numpy().squeeze().transpose(1, 2, 0)

            fused_right_file_stem = os.path.join(output_directory_fused_right, left_imfile_vis.split('/')[-1])
            fused_right = fused_right.cpu().numpy().squeeze().transpose(1, 2, 0)

            if args.save_png:
                disp_16 = np.round(disp * 256).astype(np.uint16)
                skimage.io.imsave(disp_file_stem, disp_16)

                fused_left_8 = np.round(fused_left).astype(np.uint8)
                skimage.io.imsave(fused_left_file_stem, fused_left_8)

                fused_right_8 = np.round(fused_right).astype(np.uint8)
                skimage.io.imsave(fused_right_file_stem, fused_right_8)

            # plt.imsave(file_stem, disp, cmap='jet')

            if args.save_numpy:
                np.save(disp_file_stem.replace('.png', '.npy'), disp)


def remove_module_prefix(state_dict):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        new_key = key.replace('module.', '', 1)  # 只替换开头的第一个匹配项
        new_state_dict[new_key] = value
    return new_state_dict


def save(args):
    model = torch.nn.DataParallel(PFIGEVStereo(args), device_ids=[0])
    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt))
        print(f'Loaded checkpoint from {args.restore_ckpt}')
    
    if args.restore_ckpt_fusion is not None:
        fusion_checkpoint = torch.load(args.restore_ckpt_fusion)
        model.module.fusion_model.load_state_dict(fusion_checkpoint, strict=True)
        print(f'Loaded fusion checkpoint from {args.restore_ckpt_fusion}')

    if args.restore_ckpt_stereo is not None:
        stereo_checkpoint = torch.load(args.restore_ckpt_stereo)
        stereo_checkpoint = remove_module_prefix(stereo_checkpoint)
        model.module.stereo_model.load_state_dict(stereo_checkpoint, strict=True)
        print(f'Loaded stereo checkpoint from {args.restore_ckpt_stereo}')

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory_disp = Path(os.path.join(args.output_directory, 'disp'))
    output_directory_disp.mkdir(exist_ok=True, parents=True)

    output_directory_disp2 = Path(os.path.join(args.output_directory, 'disp2'))
    output_directory_disp2.mkdir(exist_ok=True, parents=True)
    
    output_directory_fused_left = Path(os.path.join(args.output_directory, 'fused_left'))
    output_directory_fused_left.mkdir(exist_ok=True, parents=True)

    with torch.no_grad():
        left_imgs_vis = sorted(glob.glob(args.left_imgs_vis, recursive=True))
        right_imgs_vis = sorted(glob.glob(args.right_imgs_vis, recursive=True))
        left_imgs_pol = sorted(glob.glob(args.left_imgs_pol, recursive=True))
        right_imgs_pol = sorted(glob.glob(args.right_imgs_pol, recursive=True))
        print(f"Found {len(left_imgs_vis)} images.")

        for (left_imfile_vis, left_imfile_pol, right_imfile_vis, right_imfile_pol) in tqdm(list(zip(left_imgs_vis, left_imgs_pol, right_imgs_vis, right_imgs_pol))):
            left_image_vis = load_image(left_imfile_vis)
            left_image_pol = load_image(left_imfile_pol)
            right_image_vis = load_image(right_imfile_vis)
            right_image_pol = load_image(right_imfile_pol)

            padder = InputPadder(left_image_vis.shape, divis_by=32)
            left_image_vis, left_image_pol, right_image_vis, right_image_pol = \
                padder.pad(left_image_vis, left_image_pol, right_image_vis, right_image_pol)
            
            disp, fused_left, fused_right = model(
                left_image_vis, right_image_vis, left_image_pol, right_image_pol, iters=args.valid_iters, test_mode=True)
            disp = padder.unpad(disp)
            fused_left = padder.unpad(fused_left)
            fused_right = padder.unpad(fused_right)

            disp2 = model.stereo_model(left_image_vis, right_image_vis, iters=args.valid_iters, test_mode=True)
            disp2 = padder.unpad(disp2)

            disp_file_stem = os.path.join(output_directory_disp, left_imfile_vis.split('/')[-1])
            disp2_file_stem = os.path.join(output_directory_disp2, left_imfile_vis.split('/')[-1])
            disp = disp.cpu().numpy().squeeze()
            disp2 = disp2.cpu().numpy().squeeze()

            fused_left_file_stem = os.path.join(output_directory_fused_left, left_imfile_vis.split('/')[-1])
            fused_left = fused_left.cpu().numpy().squeeze().transpose(1, 2, 0)

            if args.save_png:
                disp_np = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                # disp_np = cv2.applyColorMap(disp_np, cv2.COLORMAP_JET)

                disp2_np = cv2.normalize(disp2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                # disp2_np = cv2.applyColorMap(disp2_np, cv2.COLORMAP_JET)
                
                cv2.imwrite(disp_file_stem, disp_np)
                cv2.imwrite(disp2_file_stem, disp2_np)

                # fused_left_8 = np.round(fused_left).astype(np.uint8)
                # cv2.imwrite(fused_left_file_stem, cv2.cvtColor(fused_left_8, cv2.COLOR_BGR2RGB))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", 
                        default='checkpoints_fusion_stereo\vits_cagwc\2025-10-17_16-08_kitti\ca-igev-stereo.pth')
    parser.add_argument('--restore_ckpt_fusion', help="restore checkpoint", default=None)
    parser.add_argument('--restore_ckpt_stereo', help="restore checkpoint", default='../checkpoints_stereo/vits_cagwc/2025-09-26_22-18_sceneflow/rt-igev-stereo.pth')
    parser.add_argument('--save_png', action='store_true', default=True, help='save output as gray images')
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('--left_imgs_vis', default="../datasets/3d_polarization/20251015/rectified_left_s0/*.png")
    parser.add_argument('--right_imgs_vis', default="../datasets/3d_polarization/20251015/rectified_right_s0/*.png")
    parser.add_argument('--left_imgs_pol', default="../datasets/3d_polarization/20251015/rectified_left_dop/*.png")
    parser.add_argument('--right_imgs_pol', default="../datasets/3d_polarization/20251015/rectified_right_dop/*.png")
    parser.add_argument('--output_directory', help="directory to save output", default="../output/pfigevpp/vits/3d_polarization/20251015-Fusion_KITTI-Stereo_SceneFlow")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--precision_dtype', default='float32', choices=['float16', 'bfloat16', 'float32'], help='Choose precision type: float16 or bfloat16 or float32')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg', 'mobilenetv2_100'])
    parser.add_argument('--corr_module', type=str, default='cagwc', choices=['gwc', 'cagwc'])
    parser.add_argument('--hidden_dim', nargs='+', type=int, default=96, help="hidden state and context dimensions")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=320, help="max disp range")
    
    args = parser.parse_args()
    os.makedirs(args.output_directory, exist_ok=True)
    save(args)
