import torch
import torch.nn as nn
import torch.nn.functional as F
from core_stereo_rt.extractor import Feature
from core_stereo_rt.update import BasicUpdateBlock
from core_stereo_rt.geometry import Geo_Encoding_Volume
from core_stereo_rt.submodule import *
from depth_anything_v2.dpt import DepthAnythingV2, DepthAnythingV2_decoder

try:
    autocast = torch.cuda.amp.autocast
except:
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class hourglass(nn.Module):
    def __init__(self, in_channels):
        super(hourglass, self).__init__()

        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))
                                    
        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1))                             

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   BasicConv(in_channels*6, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=1, dilation=1)) 


        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, 8, deconv=True, is_3d=True, bn=False,
                                  relu=False, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*4, in_channels*4, is_3d=True, kernel_size=3, padding=1, stride=1),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1),
                                   BasicConv(in_channels*2, in_channels*2, is_3d=True, kernel_size=3, padding=1, stride=1))

        self.feature_att_8 = FeatureAtt(in_channels*2, 64)
        self.feature_att_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_32 = FeatureAtt(in_channels*6, 160)
        self.feature_att_up_16 = FeatureAtt(in_channels*4, 192)
        self.feature_att_up_8 = FeatureAtt(in_channels*2, 64)

    def forward(self, x, features):
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, features[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, features[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, features[3])

        conv3_up = self.conv3_up(conv3)
        conv2 = torch.cat((conv3_up, conv2), dim=1)
        conv2 = self.agg_0(conv2)
        conv2 = self.feature_att_up_16(conv2, features[2])

        conv2_up = self.conv2_up(conv2)
        conv1 = torch.cat((conv2_up, conv1), dim=1)
        conv1 = self.agg_1(conv1)
        conv1 = self.feature_att_up_8(conv1, features[1])

        conv = self.conv1_up(conv1)

        return conv
    

class Feat_transfer(nn.Module):
    def __init__(self, dim_list):
        super(Feat_transfer, self).__init__()
        self.conv4x = nn.Sequential(
            nn.Conv2d(in_channels=int(48+dim_list[0]), out_channels=48, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(48), nn.ReLU()
            )
        self.conv8x = nn.Sequential(
            nn.Conv2d(in_channels=int(64+dim_list[0]), out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(64), nn.ReLU()
            )
        self.conv16x = nn.Sequential(
            nn.Conv2d(in_channels=int(192+dim_list[0]), out_channels=192, kernel_size=5, stride=1, padding=2),
            nn.InstanceNorm2d(192), nn.ReLU()
            )
        self.conv32x = nn.Sequential(
            nn.Conv2d(in_channels=dim_list[0], out_channels=160, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(160), nn.ReLU()
            )
        self.conv_up_32x = nn.ConvTranspose2d(160,
                                192,
                                kernel_size=3,
                                padding=1,
                                output_padding=1,
                                stride=2,
                                bias=False)
        self.conv_up_16x = nn.ConvTranspose2d(192,
                                64,
                                kernel_size=3,
                                padding=1,
                                output_padding=1,
                                stride=2,
                                bias=False)
        self.conv_up_8x = nn.ConvTranspose2d(64,
                                48,
                                kernel_size=3,
                                padding=1,
                                output_padding=1,
                                stride=2,
                                bias=False)
        
        self.res_16x = nn.Conv2d(dim_list[0], 192, kernel_size=1, padding=0, stride=1)
        self.res_8x = nn.Conv2d(dim_list[0], 64, kernel_size=1, padding=0, stride=1)
        self.res_4x = nn.Conv2d(dim_list[0], 48, kernel_size=1, padding=0, stride=1)

    def forward(self, features):
        features_mono_list = []
        feat_32x = self.conv32x(features[3])
        feat_32x_up = self.conv_up_32x(feat_32x)
        feat_16x = self.conv16x(torch.cat((features[2], feat_32x_up), 1)) + self.res_16x(features[2])
        feat_16x_up = self.conv_up_16x(feat_16x)
        feat_8x = self.conv8x(torch.cat((features[1], feat_16x_up), 1)) + self.res_8x(features[1])
        feat_8x_up = self.conv_up_8x(feat_8x)
        feat_4x = self.conv4x(torch.cat((features[0], feat_8x_up), 1)) + self.res_4x(features[0])
        features_mono_list.append(feat_4x)
        features_mono_list.append(feat_8x)
        features_mono_list.append(feat_16x)
        features_mono_list.append(feat_32x)
        return features_mono_list
    

class IGEVStereo(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args       
        context_dim = args.hidden_dim
        self.update_block = BasicUpdateBlock(self.args, hidden_dim=args.hidden_dim)
        self.hnet = nn.Sequential(BasicConv(96, args.hidden_dim, kernel_size=3, stride=1, padding=1),
                                     nn.Conv2d(args.hidden_dim, args.hidden_dim, 3, 1, 1, bias=False))

        self.cnet = BasicConv(96, context_dim, kernel_size=3, stride=1, padding=1)
        self.context_zqr_conv = nn.Conv2d(context_dim, context_dim*3, 3, padding=3//2)
        # self.feature = Feature()

        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU()
            )
        self.stem_4 = nn.Sequential(
            BasicConv_IN(32, 48, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(48, 48, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(48), nn.ReLU()
            )

        self.spx = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)
        self.spx_2 = Conv2x_IN(24, 32, True)
        self.spx_4 = nn.Sequential(
            BasicConv_IN(96, 24, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(24, 24, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(24), nn.ReLU()
            )

        self.spx_2_gru = Conv2x(32, 32, True)
        self.spx_gru = nn.Sequential(nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),)

        self.conv = BasicConv_IN(96, 96, kernel_size=3, padding=1, stride=1)
        self.desc = nn.Conv2d(96, 96, kernel_size=1, padding=0, stride=1)

        self.patch0 = nn.Conv3d(8, 8, kernel_size=(2, 1, 1), stride=(2, 1, 1), bias=False)
        self.patch1 = nn.Conv3d(8, 8, kernel_size=(4, 1, 1), stride=(4, 1, 1), bias=False)

        self.cost_agg = hourglass(8)
        self.classifier = nn.Conv3d(8, 1, 3, 1, 1, bias=False)
        
        use_dinov2 = self.use_dinov2 = False
        use_mobilenetv2_100 = self.use_mobilenetv2_100 = False

        if args.encoder in ['vits', 'vitb', 'vitl', 'vitg']:
            use_dinov2 = self.use_dinov2 = True
        elif args.encoder == 'mobilenetv2_100':
            use_mobilenetv2_100 = self.use_mobilenetv2_100 = True

        if use_dinov2:
            self.corr_stem = BasicConv(8, 8, is_3d=True, kernel_size=3, stride=1, padding=1)
            self.corr_feature_att = FeatureAtt(8, 96)
        
            self.intermediate_layer_idx = {
                'vits': [2, 5, 8, 11],
                'vitb': [2, 5, 8, 11], 
                'vitl': [4, 11, 17, 23], 
                'vitg': [9, 19, 29, 39]
            }
            mono_model_configs = {
                'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
                'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
                'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
                'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
            }
            depth_anything = DepthAnythingV2(**mono_model_configs[args.encoder])
            depth_anything_decoder = DepthAnythingV2_decoder(**mono_model_configs[args.encoder])
            state_dict_dpt = torch.load(f'./depth_anything_v2_{args.encoder}.pth', map_location='cpu')
            depth_anything.load_state_dict(state_dict_dpt, strict=True)
            depth_anything_decoder.load_state_dict(state_dict_dpt, strict=False)
            self.mono_encoder = depth_anything.pretrained
            self.feat_decoder = depth_anything_decoder.depth_head
            if self.args.mde_encoder_freeze == 'full':
                self.mono_encoder.requires_grad_(False)
            elif self.args.mde_encoder_freeze == 'except_last_two_layers':
                self.mono_encoder.requires_grad_(False)
                layer_indices = self.intermediate_layer_idx[args.encoder]
                unfreeze_start_idx = layer_indices[-2]
                for i in range(unfreeze_start_idx, len(self.mono_encoder.blocks)):
                    self.mono_encoder.blocks[i].requires_grad_(True)
            
            if self.args.mde_decoder_freeze:
                depth_anything_decoder.requires_grad_(False)

            del depth_anything, state_dict_dpt, depth_anything_decoder

            dim_list_ = mono_model_configs[self.args.encoder]['features']
            dim_list = []
            dim_list.append(dim_list_)
            self.feat_transfer = Feat_transfer(dim_list)
        elif use_mobilenetv2_100:
            self.feature = Feature()

        if self.args.corr_module == 'cagwc':
            self.corr_module = ContextAwareGWCorrelation(in_channels=96, num_groups=8)
        elif self.args.corr_module == 'gwc':
            self.corr_module = GWCorrelation(num_groups=8)

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def upsample_disp(self, disp, mask_feat_4, stem_2x):

        with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
            xspx = self.spx_2_gru(mask_feat_4, stem_2x)# [B, 64, H/2, W/2]
            spx_pred = self.spx_gru(xspx) # [B, 9, H, W]
            spx_pred = F.softmax(spx_pred, 1)
            up_disp = context_upsample(disp*4., spx_pred)
        return up_disp

    def feature_extraction(self, image1, image2):
        resize_image1 = F.interpolate(image1, scale_factor=14 / 16, mode='bilinear', align_corners=True)
        resize_image2 = F.interpolate(image2, scale_factor=14 / 16, mode='bilinear', align_corners=True)

        patch_h, patch_w = resize_image1.shape[-2] // 14, resize_image1.shape[-1] // 14
        features_left_encoder = self.mono_encoder.get_intermediate_layers(
            resize_image1, self.intermediate_layer_idx[self.args.encoder], return_class_token=True)
        features_right_encoder = self.mono_encoder.get_intermediate_layers(
            resize_image2, self.intermediate_layer_idx[self.args.encoder], return_class_token=True)
        
        # [B, 256, H/4, W/4], [B, 256, H/8, W/8], [B, 256, H/16, W/16], [B, 256, H/32, W/32]
        features_left_4x, features_left_8x, features_left_16x, features_left_32x = self.feat_decoder(features_left_encoder, patch_h, patch_w)
        features_right_4x, features_right_8x, features_right_16x, features_right_32x = self.feat_decoder(features_right_encoder, patch_h, patch_w)
        
        return [features_left_4x, features_left_8x, features_left_16x, features_left_32x], [features_right_4x, features_right_8x, features_right_16x, features_right_32x]

    def build_volume(self, refimg_fea, targetimg_fea, maxdisp, num_groups):
        B, C, H, W = refimg_fea.shape
        volume = refimg_fea.new_zeros([B, num_groups, maxdisp, H, W])
        corr_module = self.corr_module

        for i in range(maxdisp):
            if i > 0:
                volume[:, :, i, :, i:] = corr_module(
                    refimg_fea[:, :, :, i:], targetimg_fea[:, :, :, :-i]
                )
            else:
                volume[:, :, i, :, :] = corr_module(refimg_fea, targetimg_fea)
                
        return volume.contiguous()

    def forward(self, image1, image2, iters=12, flow_init=None, test_mode=False):
        """ Estimate disparity between pair of frames """
    
        image1 = (2 * (image1 / 255.0) - 1.0).contiguous()
        image2 = (2 * (image2 / 255.0) - 1.0).contiguous()
        with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
            # feature extraction
            if self.use_dinov2:
                features_mono_left, features_mono_right = self.feature_extraction(image1, image2)

                # [B, 48, H/4, W/4], [B, 64, H/8, W/8], [B, 192, H/16, W/16], [B, 160, H/32, W/32]
                features_left = self.feat_transfer(features_mono_left)
                features_right = self.feat_transfer(features_mono_right)
            elif self.use_mobilenetv2_100:
                features_left = self.feature(image1)
                features_right = self.feature(image2)
                
            stem_2x = self.stem_2(image1)  # [B, 32, H/2, W/2]
            stem_4x = self.stem_4(stem_2x)  # [B, 48, H/4, W/4]
            stem_2y = self.stem_2(image2)
            stem_4y = self.stem_4(stem_2y)
            features_left[0] = torch.cat((features_left[0], stem_4x), 1) # [B, 96, H/4, W/4]
            features_right[0] = torch.cat((features_right[0], stem_4y), 1)

            match_left = self.desc(self.conv(features_left[0]))  # [B, 96, H/4, W/4]
            match_right = self.desc(self.conv(features_right[0]))
            gwc_volume = self.build_volume(match_left, match_right, self.args.max_disp//4, 8)  # [B, 8, max_disp/4, H/4, W/4]

            if self.use_dinov2:
                gwc_volume = self.corr_stem(gwc_volume) 
                gwc_volume = self.corr_feature_att(gwc_volume, features_left[0])
        
            geo_encoding_volume = self.cost_agg(gwc_volume, features_left)

            # Init disp from geometry encoding volume
            prob = F.softmax(self.classifier(geo_encoding_volume).squeeze(1), dim=1)
            init_disp = disparity_regression(prob, self.args.max_disp//4, 1)

            del prob, gwc_volume

            if not test_mode:
                xspx = self.spx_4(features_left[0])
                xspx = self.spx_2(xspx, stem_2x)
                spx_pred = self.spx(xspx)
                spx_pred = F.softmax(spx_pred, 1)

            hidden = self.hnet(features_left[0])  # [B, 96, H/4, W/4]
            net = torch.tanh(hidden)
            context = self.cnet(features_left[0])
            context = list(self.context_zqr_conv(context).split(split_size=self.args.hidden_dim, dim=1))

        geo_block = Geo_Encoding_Volume
        geo_fn = geo_block(geo_encoding_volume.float(), radius=self.args.corr_radius, num_levels=self.args.corr_levels)
        b, c, h, w = match_left.shape
        disp = init_disp
        disp_preds = []

        # GRUs iterations to update disparity
        for itr in range(iters):
            disp = disp.detach()
            geo_feat = geo_fn(disp)
            with autocast(enabled=self.args.mixed_precision, dtype=getattr(torch, self.args.precision_dtype, torch.float16)):
                net, mask_feat_4, delta_disp = self.update_block(net, context, geo_feat, disp)
            disp = disp + delta_disp
            if test_mode and itr < iters-1:
                continue

            # upsample predictions
            disp_up = self.upsample_disp(disp, mask_feat_4, stem_2x)
            disp_preds.append(disp_up)

        if test_mode:
            return disp_up

        init_disp = context_upsample(init_disp*4., spx_pred.float())
        return init_disp, disp_preds
