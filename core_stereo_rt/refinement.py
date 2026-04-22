import torch
import torch.nn as nn
import torch.nn.functional as F
from core_stereo_rt.warp import disp_warp

def conv2d(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1):
    return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                                   stride=stride, padding=dilation, dilation=dilation,
                                   bias=False, groups=groups),
                         nn.BatchNorm2d(out_channels),
                         nn.LeakyReLU(0.2, inplace=True))


class BasicConv_now(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, **kwargs):
        super(BasicConv_now, self).__init__()
        self.relu = relu
        self.use_bn = bn
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x


class Conv2x_now(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, bn=True, relu=True,
                 mdconv=False):
        super(Conv2x_now, self).__init__()
        self.concat = concat

        if deconv and is_3d:
            kernel = (3, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3
        self.conv1 = BasicConv_now(in_channels, out_channels, deconv, is_3d, bn=True, relu=True, kernel_size=kernel,
                               stride=2, padding=1)

        if self.concat:
            # if mdconv:
            #     self.conv2 = DeformConv2d(out_channels * 2, out_channels, kernel_size=3, stride=1)
            # else:
            self.conv2 = BasicConv_now(out_channels * 2, out_channels, False, is_3d, bn, relu, kernel_size=3,
                                       stride=1, padding=1)
        else:
            self.conv2 = BasicConv_now(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1,
                                   padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        # print('x.size()', x.size())
        # print('rem.size()', rem.size())
        assert (x.size() == rem.size())

        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x


class Simple_UNet(nn.Module):
    def __init__(self, in_channels):
        super(Simple_UNet, self).__init__()
        self.conv1a = BasicConv_now(in_channels, 48, kernel_size=3, stride=2, padding=1)
        self.conv2a = BasicConv_now(48, 64, kernel_size=3, stride=2, padding=1)
        self.conv3a = BasicConv_now(64, 96, kernel_size=3, stride=2, dilation=2, padding=2)
        self.conv4a = BasicConv_now(96, 128, kernel_size=3, stride=2, dilation=2, padding=2)

        self.deconv4a = Conv2x_now(128, 96, deconv=True)
        self.deconv3a = Conv2x_now(96, 64, deconv=True)
        self.deconv2a = Conv2x_now(64, 48, deconv=True)
        self.deconv1a = Conv2x_now(48, 32, deconv=True)

        self.conv1b = Conv2x_now(32, 48)
        self.conv2b = Conv2x_now(48, 64)
        self.conv3b = Conv2x_now(64, 96)
        self.conv4b = Conv2x_now(96, 128)

        self.deconv4b = Conv2x_now(128, 96, deconv=True)
        self.deconv3b = Conv2x_now(96, 64, deconv=True)
        self.deconv2b = Conv2x_now(64, 48, deconv=True)
        self.deconv1b = Conv2x_now(48, in_channels, deconv=True)

    def forward(self, x):
        rem0 = x
        x = self.conv1a(x)
        rem1 = x
        x = self.conv2a(x)
        rem2 = x
        x = self.conv3a(x)
        rem3 = x
        x = self.conv4a(x)
        rem4 = x

        x = self.deconv4a(x, rem3)
        rem3 = x

        x = self.deconv3a(x, rem2)
        rem2 = x
        x = self.deconv2a(x, rem1)
        rem1 = x
        x = self.deconv1a(x, rem0)
        rem0 = x

        x = self.conv1b(x, rem1)
        rem1 = x
        x = self.conv2b(x, rem2)
        rem2 = x
        x = self.conv3b(x, rem3)
        rem3 = x
        x = self.conv4b(x, rem4)

        x = self.deconv4b(x, rem3)
        x = self.deconv3b(x, rem2)
        x = self.deconv2b(x, rem1)
        x = self.deconv1b(x, rem0)  # [B, 32, H, W]

        return x
    

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class REMP(nn.Module):
    """Height and width need to be divided by 16"""

    def __init__(self):
        super(REMP, self).__init__()

        # Left and warped flaw
        in_channels = 6
        channel =32
        self.conv1_stereo = conv2d(in_channels, 16)
        self.conv2_stereo = conv2d(1, 16)  # on low disparity

        self.conv_start = BasicConv_now(32, channel, kernel_size=3, padding=2, dilation=2)

        self.RefinementBlock = Simple_UNet(in_channels=channel)#, in_channels

        self.AP = nn.AdaptiveAvgPool2d(1)
        self.LFE = nn.Sequential(
            nn.Conv2d(channel, channel * 2, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel * 2, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.LMC = nn.Sequential(
            default_conv(channel, channel, 3),
            default_conv(channel, channel * 2, 3),
            nn.ReLU(inplace=True),
            default_conv(channel * 2, channel, 3),
            nn.Sigmoid()
        )

        self.final_conv = nn.Conv2d(32, 1, 3, 1, 1)

    def forward(self, disp_stereo, left_img, right_img):
        assert disp_stereo.dim() == 4

        warped_right_stereo = disp_warp(right_img, disp_stereo)[0]  # [B, 3, H, W]
        flaw_stereo = warped_right_stereo - left_img  # [B, 3, H, W]

        ref_flaw_stereo = torch.cat((flaw_stereo, left_img), dim=1)  # [B, 6, H, W]
        ref_flaw_stereo = self.conv1_stereo(ref_flaw_stereo)  # [B, 16, H, W]

        disp_fea_stereo = self.conv2_stereo(disp_stereo)  # [B, 16, H, W]

        x = torch.cat((ref_flaw_stereo, disp_fea_stereo), dim=1)  # [B, 32, H, W]
        x = self.conv_start(x)  # [B, 32, H, W]
        x = self.RefinementBlock(x) # [B, 32, H, W]

        low = self.LFE(self.AP(x))
        motif = self.LMC(x)
        x = torch.mul((1 - motif), low) + torch.mul(motif, x)
        x = self.final_conv(x)  # [B, 1, H, W]
        disp_stereo = nn.LeakyReLU()(disp_stereo + x)  # [B, 1, H, W]

        return disp_stereo

