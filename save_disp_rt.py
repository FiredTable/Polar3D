import sys
sys.path.append('core_rt')

import argparse
import glob
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pathlib import Path
from core_stereo_rt.rt_igev_stereo import IGEVStereo
from core_stereo_rt.utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
import os
import skimage.io
import cv2
from core_stereo_rt.utils.frame_utils import read_gen, readDispMiddlebury, readDispKITTI


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
    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory.resolve()}/")

        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            image1 = load_image(imfile1)
            image1 = image1[:, 0:3, :, :]
            image2 = load_image(imfile2)
            image2 = image2[:, 0:3, :, :]
            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)
            disp = model(image1, image2, iters=args.valid_iters, test_mode=True)
            disp = padder.unpad(disp)
            file_stem = os.path.join(output_directory, imfile1.split('/')[-1])
            disp = disp.cpu().numpy().squeeze()
            if args.save_png:
                disp_16 = np.round(disp * 256).astype(np.uint16)
                skimage.io.imsave(file_stem, disp_16)

            if args.save_numpy:
                np.save(file_stem.replace('.png', '.npy'), disp)


def save_for_kitti(args):
    """用于kitti图像并计算EPE"""
    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            image1 = load_image(imfile1)
            image1 = image1[:, 0:3, :, :]
            image2 = load_image(imfile2)
            image2 = image2[:, 0:3, :, :]
            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)
            disp = model(image1, image2, iters=args.valid_iters, test_mode=True)
            disp = padder.unpad(disp)
            file_stem = os.path.join(output_directory, imfile1.split('/')[-1])
            disp = disp.cpu().numpy().squeeze()
            if args.save_png:
                disp_8 = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                disp_color = cv2.applyColorMap(disp_8, cv2.COLORMAP_JET)
                skimage.io.imsave(file_stem, disp_color)
            # plt.imsave(file_stem, disp, cmap='jet')

            if args.save_numpy:
                np.save(file_stem.replace('.png', '.npy'), disp)



def save_for_middlebury(args):
    """用于middlebury图像并计算EPE"""
    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    gt_out_directory = Path(args.output_directory.replace('caigevpp_vits', 'gt_disp'))
    gt_out_directory.mkdir(exist_ok=True, parents=True)

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        if args.gt_files is not None:
            gt_files = sorted(glob.glob(args.gt_files, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            image1 = load_image(imfile1)
            image1 = image1[:, 0:3, :, :]
            image2 = load_image(imfile2)
            image2 = image2[:, 0:3, :, :]

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)
            disp = model(image1, image2, iters=args.valid_iters, test_mode=True)
            disp = padder.unpad(disp).cpu().squeeze(0)

            def get_gt_disp(gt_file):
                disp = readDispMiddlebury(gt_file)
                if isinstance(disp, tuple):
                    disp, valid = disp
                else:
                    valid = disp < 1024

                disp = np.array(disp).astype(np.float32)
                flow = np.stack([disp, np.zeros_like(disp)], axis=-1)
                flow = torch.from_numpy(flow).permute(2, 0, 1).float()
                flow = flow[:1]

                valid = torch.from_numpy(valid)
                return flow, valid.float()

            if args.gt_files is not None:
                disp_gt, valid_gt = get_gt_disp(gt_files[0])
                epe = torch.sum((disp - disp_gt)**2, dim=0).sqrt()
                epe_flattened = epe.flatten()
            
                try:
                    occ_mask = Image.open(imfile1.replace('im0.png', 'mask0nocc.png')).convert('L')
                    occ_mask = np.ascontiguousarray(occ_mask, dtype=np.float32).flatten()
                    val = (valid_gt.reshape(-1) >= 0.5) & (disp_gt[0].reshape(-1) < args.max_disp) & (occ_mask==255)
                except:
                    val = (valid_gt.reshape(-1) >= 0.5) & (disp_gt[0].reshape(-1) < args.max_disp)

                out = (epe_flattened > 2.0)
                image_out = out[val].float().mean().item()
                image_epe = epe_flattened[val].mean().item()
                print(f"{imfile1} Validation EPE: {image_epe} Out: {image_out}")

            file_stem = os.path.join(output_directory, f"iter{args.valid_iters}_{imfile1.split('/')[-1]}")
            
            if args.save_png:
                breakpoint()
                if disp.max().item() > 255:
                    disp_np = disp.squeeze().data.cpu().numpy()
                    disp_np = cv2.normalize(disp_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    disp_np = cv2.applyColorMap(disp_np, cv2.COLORMAP_MAGMA)
                else:
                    disp_np = disp.squeeze().data.cpu().numpy().astype(np.uint8)
                    disp_np = cv2.applyColorMap(disp_np, cv2.COLORMAP_MAGMA)
                cv2.imwrite(file_stem, disp_np)
            
            if args.save_gt:
                disp_gt[0][valid_gt==0] = 0
                disp_gt[disp_gt==torch.tensor(np.inf)]=0
                print(f'disp gt max: {disp_gt.max().item()}')
                if disp_gt.max().item() > 255:
                    disp_gt_np = disp_gt.squeeze().data.cpu().numpy()
                    disp_gt_np = cv2.normalize(disp_gt_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    disp_gt_np = cv2.applyColorMap(disp_gt_np, cv2.COLORMAP_MAGMA)
                else:
                    disp_gt_np = disp_gt.squeeze().data.cpu().numpy().astype(np.uint8)
                    disp_gt_np = cv2.applyColorMap(disp_gt_np, cv2.COLORMAP_MAGMA)
                cv2.imwrite(os.path.join(gt_out_directory, imfile1.split('/')[-1]), disp_gt_np)

            if args.save_numpy:
                np.save(file_stem.replace('.png', '.npy'), disp)

    

def save_for_sceneflow(args):
    """用于保存sceneflow图像并计算EPE"""
    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    if args.save_gt:
        gt_directory = Path(args.gt_directory)
        gt_directory.mkdir(exist_ok=True, parents=True)

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        gt_files = sorted(glob.glob(args.gt_files, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for (imfile1, imfile2, gt_file) in list(zip(left_images, right_images, gt_files)):
            image1 = load_image(imfile1)
            image1 = image1[:, 0:3, :, :]
            image2 = load_image(imfile2)
            image2 = image2[:, 0:3, :, :]
            
            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)
            disp = model(image1, image2, iters=args.valid_iters, test_mode=True)
            disp = padder.unpad(disp).cpu()
            
            def get_gt_disp(gt_file):
                disp_gt = read_gen(gt_file)
                disp_gt = np.array(disp_gt).astype(np.float32)
                flow = np.stack([disp_gt, np.zeros_like(disp_gt)], axis=-1)
                flow = torch.from_numpy(flow).permute(2, 0, 1).float()
                valid = (flow[0].abs() < 1024) & (flow[1].abs() < 1024)
                disp_gt = flow[:1]
                valid_gt = valid.float()
                return disp_gt, valid_gt

            disp_gt, valid_gt = get_gt_disp(gt_file)
            epe = torch.abs(disp - disp_gt.unsqueeze(0))

            epe = epe.flatten()
            val = (valid_gt.flatten() >= 0.5) & (disp_gt.abs().flatten() < args.max_disp)

            epe_val = epe[val].mean().item()
            print(f"{imfile1} Validation EPE: {epe_val}")
            print(f'disp gt max: {disp_gt.max().item()}')

            file_stem = os.path.join(output_directory, imfile1.split('/')[-1])
            
            if args.save_png:
                if disp.max().item() > 255:
                    disp_np = disp.squeeze().data.cpu().numpy()
                    disp_np = cv2.normalize(disp_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    disp_np = cv2.applyColorMap(disp_np, cv2.COLORMAP_MAGMA)
                else:
                    disp_np = disp.squeeze().data.cpu().numpy().astype(np.uint8)
                    disp_np = cv2.applyColorMap(disp_np, cv2.COLORMAP_MAGMA)
                cv2.imwrite(file_stem, disp_np)
            
            if args.save_gt:
                if disp_gt.max().item() > 255:
                    disp_gt_np = disp_gt.squeeze().data.cpu().numpy()
                    disp_gt_np = cv2.normalize(disp_gt_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    disp_gt_np = cv2.applyColorMap(disp_gt_np, cv2.COLORMAP_MAGMA)
                else:
                    disp_gt_np = disp_gt.squeeze().data.cpu().numpy().astype(np.uint8)
                    disp_gt_np = cv2.applyColorMap(disp_gt_np, cv2.COLORMAP_MAGMA)
                cv2.imwrite(os.path.join(gt_directory, imfile1.split('/')[-1]), disp_gt_np)

            if args.save_numpy:
                np.save(file_stem.replace('.png', '.npy'), disp)


def save_for_3d_polarization(args):
    """用于保存3d polarization图像"""
    model = torch.nn.DataParallel(IGEVStereo(args), device_ids=[0])
    model.load_state_dict(torch.load(args.restore_ckpt))

    model = model.module
    model.to(DEVICE)
    model.eval()

    output_directory = Path(args.output_directory)
    output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        for (imfile1, imfile2) in list(zip(left_images, right_images)):
            print(f'Processing {imfile1}')
            image1 = load_image(imfile1)
            image1 = image1[:, 0:3, :, :]
            image2 = load_image(imfile2)
            image2 = image2[:, 0:3, :, :]

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)
            disp = model(image1, image2, iters=args.valid_iters, test_mode=True)
            disp = padder.unpad(disp).cpu()
            disp_np = disp.squeeze().data.cpu().numpy()

            if args.edge_trimming:
                disp_np[:, :args.max_disp] = 0

            file_stem = os.path.join(output_directory, imfile1.split('/')[-1])
            
            if args.save_png:
                disp_np = cv2.normalize(disp_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                disp_np = cv2.applyColorMap(disp_np, cv2.COLORMAP_JET)
                cv2.imwrite(file_stem, cv2.cvtColor(disp_np, cv2.COLOR_BGR2RGB))
            
            if args.save_numpy:
                np.save(file_stem.replace('.png', '.npy'), disp)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", 
                        default=r'checkpoints_stereo\vits_cagwc\2025-09-27_17-38_kitti\ca-igev-stereo.pth')
    parser.add_argument('--save_png', action='store_true', default=True, help='save output as gray images')
    parser.add_argument('--save_gt', action='store_true', default=False, help='save ground truth as gray images')
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('--edge_trimming', default=False)
    parser.add_argument('--left_imgs', default='/root/autodl-tmp/Polar3D/datasets/kitti/2015/testing/image_2/*_10.png')
    parser.add_argument('--right_imgs', default='/root/autodl-tmp/Polar3D/datasets/kitti/2015/testing/image_3/*_10.png')
    parser.add_argument('--gt_files', default='/root/autodl-tmp/Polar3D/datasets/sceneflow/disparity/TRAIN/eating_x2/left/0000.pfm')
    parser.add_argument('--gt_directory', default='/root/autodl-tmp/Polar3D/output/gt_disp/sceneflow/frames_finalpass/TRAIN/eating_x2/left/')
    parser.add_argument('--output_directory', default="/root/autodl-tmp/Polar3D/output/caigev_vits_cagwc/variant-d/kitti/2015/testing")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--precision_dtype', default='float32', choices=['float16', 'bfloat16', 'float32'], help='Choose precision type: float16 or bfloat16 or float32')
    parser.add_argument('--valid_iters', type=int, default=8, help='number of flow-field updates during forward pass')

    # Architecture choices
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg', 'mobilenetv2_100'])
    parser.add_argument('--corr_module', type=str, default='cagwc', choices=['gwc', 'cagwc'])
    parser.add_argument('--hidden_dim', nargs='+', type=int, default=96, help="hidden state and context dimensions")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp range")
    parser.add_argument('--mde_encoder_freeze', type=str, default='all', choices=['all', 'except_last_two_layers'])
    parser.add_argument('--mde_decoder_freeze', type=bool, default=False, help='freeze the decoder layers of the MDE')
    
    args = parser.parse_args()
    os.makedirs(args.output_directory, exist_ok=True)
    demo(args)
