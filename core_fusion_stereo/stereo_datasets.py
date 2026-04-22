import os
import torch
import copy
import numpy as np
import logging
import random
import os.path as osp
import torch.nn.functional as F
import torch.utils.data as data
from glob import glob
from core_stereo_rt.utils import frame_utils
from core_fusion_stereo.utils.augmentor import FlowAugmentor, SparseFlowAugmentor


class StereoDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, reader=None):
        self.augmentor = None
        self.sparse = sparse
        self.img_pad = aug_params.pop("img_pad", None) if aug_params is not None else None
        if aug_params is not None and "crop_size" in aug_params:
            if sparse:
                self.augmentor = SparseFlowAugmentor(**aug_params)
            else:
                self.augmentor = FlowAugmentor(**aug_params)

        if reader is None:
            self.disparity_reader = frame_utils.read_gen
        else:
            self.disparity_reader = reader        

        self.is_test = False
        self.init_seed = False
        self.flow_list = []
        self.disparity_list = []
        self.image_list = []
        self.extra_info = []

    def __getitem__(self, index):
        if self.is_test:
            left_img1 = frame_utils.read_gen(self.image_list[index][0])
            right_img1 = frame_utils.read_gen(self.image_list[index][1])
            left_img2 = frame_utils.read_gen(self.image_list[index][2])
            right_img2 = frame_utils.read_gen(self.image_list[index][3])

            left_img1 = np.array(left_img1).astype(np.uint8)[..., :3]
            right_img1 = np.array(right_img1).astype(np.uint8)[..., :3]
            left_img2 = np.array(left_img2).astype(np.uint8)[..., :3]
            right_img2 = np.array(right_img2).astype(np.uint8)[..., :3]

            left_img1 = torch.from_numpy(left_img1).permute(2, 0, 1).float()
            right_img1 = torch.from_numpy(right_img1).permute(2, 0, 1).float()
            left_img2 = torch.from_numpy(left_img2).permute(2, 0, 1).float()
            right_img2 = torch.from_numpy(right_img2).permute(2, 0, 1).float()

            return left_img1, right_img1, left_img2, right_img2, self.extra_info[index]

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)
        disp = self.disparity_reader(self.disparity_list[index])
        
        if isinstance(disp, tuple):
            disp, valid = disp
        else:
            valid = disp < 1024

        left_img1 = frame_utils.read_gen(self.image_list[index][0])
        right_img1 = frame_utils.read_gen(self.image_list[index][1])
        left_img2 = frame_utils.read_gen(self.image_list[index][2])
        right_img2 = frame_utils.read_gen(self.image_list[index][3])

        left_img1 = np.array(left_img1).astype(np.uint8)
        right_img1 = np.array(right_img1).astype(np.uint8)
        left_img2 = np.array(left_img2).astype(np.uint8)
        right_img2 = np.array(right_img2).astype(np.uint8)

        disp = np.array(disp).astype(np.float32)
        flow = np.stack([disp, np.zeros_like(disp)], axis=-1)

        # grayscale images
        if len(left_img1.shape) == 2:
            left_img1 = np.tile(left_img1[...,None], (1, 1, 3))
            right_img1 = np.tile(right_img1[...,None], (1, 1, 3))
        else:
            left_img1 = left_img1[..., :3]
            right_img1 = right_img1[..., :3]
        
        left_img2 = np.tile(left_img2[...,None], (1, 1, 3))
        right_img2 = np.tile(right_img2[...,None], (1, 1, 3))

        if self.augmentor is not None:
            if self.sparse:
                left_img1, right_img1, left_img2, right_img2, flow, valid = self.augmentor(left_img1, right_img1, left_img2, right_img2, flow, valid)
            else:
                left_img1, right_img1, left_img2, right_img2, flow = self.augmentor(left_img1, right_img1, left_img2, right_img2, flow)
        
        left_img1 = torch.from_numpy(left_img1).permute(2, 0, 1).float()
        right_img1 = torch.from_numpy(right_img1).permute(2, 0, 1).float()
        left_img2 = torch.from_numpy(left_img2).permute(2, 0, 1).float()
        right_img2 = torch.from_numpy(right_img2).permute(2, 0, 1).float()
        flow = torch.from_numpy(flow).permute(2, 0, 1).float()

        if self.sparse:
            valid = torch.from_numpy(valid)
        else:
            valid = (flow[0].abs() < 1024) & (flow[1].abs() < 1024)

        if self.img_pad is not None:

            padH, padW = self.img_pad
            left_img1 = F.pad(left_img1, [padW]*2 + [padH]*2)
            right_img1 = F.pad(right_img1, [padW]*2 + [padH]*2)
            left_img2 = F.pad(left_img2, [padW]*2 + [padH]*2)
            right_img2 = F.pad(right_img2, [padW]*2 + [padH]*2)

        flow = flow[:1]
        return self.image_list[index] + [self.disparity_list[index]], left_img1, right_img1, left_img2, right_img2, flow, valid.float()


    def __mul__(self, v):
        copy_of_self = copy.deepcopy(self)
        copy_of_self.flow_list = v * copy_of_self.flow_list
        copy_of_self.image_list = v * copy_of_self.image_list
        copy_of_self.disparity_list = v * copy_of_self.disparity_list
        copy_of_self.extra_info = v * copy_of_self.extra_info
        return copy_of_self
        
    def __len__(self):
        return len(self.image_list)
    

class KITTI(StereoDataset):
    def __init__(self, aug_params=None, root1='../datasets/kitti', root2='../datasets/kitti_pix2pix', image_set='training', year=2015):
        super(KITTI, self).__init__(aug_params, sparse=True, reader=frame_utils.readDispKITTI)
        assert os.path.exists(root1)
        assert os.path.exists(root2)

        if year == 2012:
            root1_12 = os.path.join(root1, '2012')
            root2_12 = os.path.join(root2, '2012')
            left_image1_list = sorted(glob(os.path.join(root1_12, image_set, 'colored_0/*_10.png')))
            right_image1_list = sorted(glob(os.path.join(root1_12, image_set, 'colored_1/*_10.png')))
            left_image2_list = sorted(glob(os.path.join(root2_12, image_set, 'colored_0/*_10.png')))
            right_image2_list = sorted(glob(os.path.join(root2_12, image_set, 'colored_1/*_10.png')))
            disp_list = sorted(glob(os.path.join(root1_12, 'training', 'disp_occ/*_10.png'))) if image_set == 'training' else [osp.join(root1, 'training/disp_occ/000085_10.png')]*len(left_image1_list)

        if year == 2015:
            root1_15 = os.path.join(root1, '2015')
            root2_15 = os.path.join(root2, '2015')

            left_image1_list = sorted(glob(os.path.join(root1_15, image_set, 'image_2/*_10.png')))
            right_image1_list = sorted(glob(os.path.join(root1_15, image_set, 'image_3/*_10.png')))
            left_image2_list = sorted(glob(os.path.join(root2_15, image_set, 'image_2/*_10.png')))
            right_image2_list = sorted(glob(os.path.join(root2_15, image_set, 'image_3/*_10.png')))
            disp_list = sorted(glob(os.path.join(root1_15, 'training', 'disp_occ_0/*_10.png'))) if image_set == 'training' else [osp.join(root1, 'training/disp_occ_0/000085_10.png')]*len(left_image1_list)

        for idx, (left_img1, right_img1, left_img2, right_img2, disp) in enumerate(zip(left_image1_list, right_image1_list, left_image2_list, right_image2_list, disp_list)):
            self.image_list += [ [left_img1, right_img1, left_img2, right_img2] ]
            self.disparity_list += [ disp ]



def fetch_dataloader(args):
    """ Create the data loader for the corresponding trainign set """

    aug_params = {'crop_size': args.image_size, 'min_scale': args.spatial_scale[0], 'max_scale': args.spatial_scale[1], 'do_flip': False, 'yjitter': not args.noyjitter}
    if hasattr(args, "saturation_range") and args.saturation_range is not None:
        aug_params["saturation_range"] = args.saturation_range
    if hasattr(args, "img_gamma") and args.img_gamma is not None:
        aug_params["gamma"] = args.img_gamma
    if hasattr(args, "do_flip") and args.do_flip is not None:
        aug_params["do_flip"] = args.do_flip

    train_dataset = None
    # for dataset_name in args.train_datasets:
    if args.train_datasets == 'kitti':
        kitti12 = KITTI(aug_params, year=2012)
        logging.info(f"Adding {len(kitti12)} samples from KITTI 2012")
        kitti15 = KITTI(aug_params, year=2015)
        logging.info(f"Adding {len(kitti15)} samples from KITTI 2015")
        new_dataset = kitti12 + kitti15
        logging.info(f"Adding {len(new_dataset)} samples from KITTI")
    
    
    train_dataset = new_dataset if train_dataset is None else train_dataset + new_dataset
    train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, 
            pin_memory=True, shuffle=True, num_workers=8, drop_last=True)

    logging.info('Training with %d image pairs' % len(train_dataset))
    return train_loader
