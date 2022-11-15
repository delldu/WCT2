"""Create model."""
# coding=utf-8
#
# /************************************************************************************
# ***
# ***    Copyright 2022 Dell(18588220928@163.com), All Rights Reserved.
# ***
# ***    File Author: Dell, 2022年 09月 08日 星期四 01:39:22 CST
# ***
# ************************************************************************************/
#

import torch
import torch.nn as nn
from typing import List, Dict
import torch.nn.functional as F

import numpy as np
import pdb


def svd(feat):
    size = feat.size()
    mean = torch.mean(feat, 1)
    mean = mean.unsqueeze(1).expand_as(feat)
    _feat = feat.clone()
    _feat -= mean
    if size[1] > 1:
        conv = torch.mm(_feat, _feat.t()).div(size[1] - 1)
    else:
        conv = torch.mm(_feat, _feat.t())
    conv += torch.eye(size[0]).to(feat.device)
    u, e, v = torch.svd(conv, some=False)
    return u, e, v


def get_squeeze_feat(feat):
    _feat = feat.squeeze(0)
    size = _feat.size(0)
    return _feat.view(size, -1).clone()


def get_rank(singular_values, dim: int, eps: float = 0.00001) -> int:
    r = dim
    for i in range(dim - 1, -1, -1):
        if singular_values[i] >= eps:
            r = i + 1
            break
    return r


def wct_core(cont_feat, styl_feat):
    cont_feat = get_squeeze_feat(cont_feat)
    cont_min = cont_feat.min()
    cont_max = cont_feat.max()
    cont_mean = torch.mean(cont_feat, 1).unsqueeze(1).expand_as(cont_feat)
    cont_feat -= cont_mean

    _, c_e, c_v = svd(cont_feat)
    styl_feat = get_squeeze_feat(styl_feat)
    s_mean = torch.mean(styl_feat, 1)
    _, s_e, s_v = svd(styl_feat)

    k_s = get_rank(s_e, styl_feat.size()[0])
    s_d = (s_e[0:k_s]).pow(0.5)
    EDE = torch.mm(torch.mm(s_v[:, 0:k_s], torch.diag(s_d)), (s_v[:, 0:k_s].t()))

    k_c = get_rank(c_e, cont_feat.size()[0])
    c_d = (c_e[0:k_c]).pow(-0.5)

    # TODO could be more fast
    step1 = torch.mm(c_v[:, 0:k_c], torch.diag(c_d))
    step2 = torch.mm(step1, (c_v[:, 0:k_c].t()))
    whiten_cF = torch.mm(step2, cont_feat)

    targetFeature = torch.mm(EDE, whiten_cF)
    targetFeature = targetFeature + s_mean.unsqueeze(1).expand_as(targetFeature)
    targetFeature.clamp_(cont_min, cont_max)

    return targetFeature


def feature_wct(content_feat, style_feat):
    target_feature = wct_core(content_feat, style_feat)
    target_feature = target_feature.view_as(content_feat)
    # alpha = 1.0
    # target_feature = alpha * target_feature + (1 - alpha) * content_feat
    return target_feature


def get_wav(in_channels, pool=True):
    """wavelet decomposition using conv2d"""
    harr_wav_L = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H = 1 / np.sqrt(2) * np.ones((1, 2))
    harr_wav_H[0, 0] = -1 * harr_wav_H[0, 0]

    harr_wav_LL = np.transpose(harr_wav_L) * harr_wav_L
    harr_wav_LH = np.transpose(harr_wav_L) * harr_wav_H
    harr_wav_HL = np.transpose(harr_wav_H) * harr_wav_L
    harr_wav_HH = np.transpose(harr_wav_H) * harr_wav_H

    filter_LL = torch.from_numpy(harr_wav_LL).unsqueeze(0)
    filter_LH = torch.from_numpy(harr_wav_LH).unsqueeze(0)
    filter_HL = torch.from_numpy(harr_wav_HL).unsqueeze(0)
    filter_HH = torch.from_numpy(harr_wav_HH).unsqueeze(0)

    if pool:
        net = nn.Conv2d
    else:
        net = nn.ConvTranspose2d

    LL = net(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)
    LH = net(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)
    HL = net(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)
    HH = net(in_channels, in_channels, kernel_size=2, stride=2, padding=0, bias=False, groups=in_channels)

    LL.weight.requires_grad = False
    LH.weight.requires_grad = False
    HL.weight.requires_grad = False
    HH.weight.requires_grad = False

    LL.weight.data = filter_LL.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    LH.weight.data = filter_LH.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    HL.weight.data = filter_HL.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()
    HH.weight.data = filter_HH.float().unsqueeze(0).expand(in_channels, -1, -1, -1).clone()

    return LL, LH, HL, HH


class WavePool(nn.Module):
    def __init__(self, in_channels):
        super(WavePool, self).__init__()
        self.LL, self.LH, self.HL, self.HH = get_wav(in_channels, pool=True)

    def forward(self, x) -> List[torch.Tensor]:
        return self.LL(x), self.LH(x), self.HL(x), self.HH(x)


class WaveUnpool(nn.Module):
    def __init__(self, in_channels):
        super(WaveUnpool, self).__init__()
        self.in_channels = in_channels
        self.LL, self.LH, self.HL, self.HH = get_wav(self.in_channels, pool=False)

    def forward(self, LL, LH, HL, HH, original):
        return torch.cat([self.LL(LL), self.LH(LH), self.HL(HL), self.HH(HH), original], dim=1)


class WaveEncoder(nn.Module):
    """cat5"""

    def __init__(self):
        super(WaveEncoder, self).__init__()
        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)

        self.conv0 = nn.Conv2d(3, 3, 1, 1, 0)
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 0)
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 0)
        self.pool1 = WavePool(64)

        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 0)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 0)
        self.pool2 = WavePool(128)

        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 0)
        self.pool3 = WavePool(256)

        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 0)

    def forward(self, x):
        skips = {}
        for level in [1, 2, 3, 4]:
            x = self.encode(x, skips, level)

        return x

    def encode(self, x, skips: Dict[str, torch.Tensor], level: int):
        assert level in [1, 2, 3, 4]
        if level == 1:
            out = self.conv0(x)
            out = self.relu(self.conv1_1(self.pad(out)))
            return out

        elif level == 2:
            out = self.relu(self.conv1_2(self.pad(x)))
            skips["conv1_2"] = out
            LL, LH, HL, HH = self.pool1(out)
            # skips['pool1'] = [LH, HL, HH] # LL not in this pool
            skips["pool1_LH"] = LH
            skips["pool1_HL"] = HL
            skips["pool1_HH"] = HH
            out = self.relu(self.conv2_1(self.pad(LL)))
            return out

        elif level == 3:
            out = self.relu(self.conv2_2(self.pad(x)))
            skips["conv2_2"] = out
            LL, LH, HL, HH = self.pool2(out)
            # skips['pool2'] = [LH, HL, HH] # LL not in this pool
            skips["pool2_LH"] = LH
            skips["pool2_HL"] = HL
            skips["pool2_HH"] = HH
            out = self.relu(self.conv3_1(self.pad(LL)))
            return out

        else:
            out = self.relu(self.conv3_2(self.pad(x)))
            out = self.relu(self.conv3_3(self.pad(out)))
            out = self.relu(self.conv3_4(self.pad(out)))
            skips["conv3_4"] = out
            LL, LH, HL, HH = self.pool3(out)
            # skips['pool3'] = [LH, HL, HH] # LL not in this pool
            skips["pool3_LH"] = LH
            skips["pool3_HL"] = HL
            skips["pool3_HH"] = HH
            out = self.relu(self.conv4_1(self.pad(LL)))
            return out


class WaveDecoder(nn.Module):
    """cat5"""

    def __init__(self):
        super(WaveDecoder, self).__init__()

        multiply_in = 5
        self.pad = nn.ReflectionPad2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.conv4_1 = nn.Conv2d(512, 256, 3, 1, 0)

        self.recon_block3 = WaveUnpool(256)
        self.conv3_4_2 = nn.Conv2d(256 * multiply_in, 256, 3, 1, 0)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 0)
        self.conv3_1 = nn.Conv2d(256, 128, 3, 1, 0)

        self.recon_block2 = WaveUnpool(128)
        self.conv2_2_2 = nn.Conv2d(128 * multiply_in, 128, 3, 1, 0)
        self.conv2_1 = nn.Conv2d(128, 64, 3, 1, 0)

        self.recon_block1 = WaveUnpool(64)
        self.conv1_2_2 = nn.Conv2d(64 * multiply_in, 64, 3, 1, 0)
        self.conv1_1 = nn.Conv2d(64, 3, 3, 1, 0)

    def forward(self, x, skips: Dict[str, torch.Tensor]):
        for level in [4, 3, 2, 1]:
            x = self.decode(x, skips, level)
        return x

    def decode(self, x, skips: Dict[str, torch.Tensor], level: int):
        assert level in [4, 3, 2, 1]
        if level == 4:
            out = self.relu(self.conv4_1(self.pad(x)))
            # LH, HL, HH = skips['pool3']
            LH = skips["pool3_LH"]
            HL = skips["pool3_HL"]
            HH = skips["pool3_HH"]
            # original = skips['conv3_4'] if 'conv3_4' in skips.keys() else None
            # out = self.recon_block3(out, LH, HL, HH, original)
            out = self.recon_block3(out, LH, HL, HH, skips["conv3_4"])
            out = self.relu(self.conv3_4_2(self.pad(out)))
            out = self.relu(self.conv3_3(self.pad(out)))
            return self.relu(self.conv3_2(self.pad(out)))
        elif level == 3:
            out = self.relu(self.conv3_1(self.pad(x)))
            # LH, HL, HH = skips['pool2']
            LH = skips["pool2_LH"]
            HL = skips["pool2_HL"]
            HH = skips["pool2_HH"]
            # original = skips['conv2_2'] if 'conv2_2' in skips.keys() else None
            # out = self.recon_block2(out, LH, HL, HH, original)
            out = self.recon_block2(out, LH, HL, HH, skips["conv2_2"])
            return self.relu(self.conv2_2_2(self.pad(out)))
        elif level == 2:
            out = self.relu(self.conv2_1(self.pad(x)))
            # LH, HL, HH = skips['pool1']
            LH = skips["pool1_LH"]
            HL = skips["pool1_HL"]
            HH = skips["pool1_HH"]
            # original = skips['conv1_2'] if 'conv1_2' in skips.keys() else None
            # out = self.recon_block1(out, LH, HL, HH, original)
            out = self.recon_block1(out, LH, HL, HH, skips["conv1_2"])
            return self.relu(self.conv1_2_2(self.pad(out)))
        else:
            return self.conv1_1(self.pad(x))


class WCT2(nn.Module):
    def __init__(self):
        super(WCT2, self).__init__()
        # Define max GPU/CPU memory -- GPU 9G, 460ms
        self.MAX_H = 1024
        self.MAX_W = 1024
        self.MAX_TIMES = 8

        self.encoder = WaveEncoder()
        self.decoder = WaveDecoder()

    def get_style_features(self, x, skips: Dict[str, torch.Tensor]) -> Dict[int, torch.Tensor]:
        feats: Dict[int, torch.Tensor] = {}
        for level in [1, 2, 3, 4]:
            x = self.encoder.encode(x, skips, level)
        feats[4] = x
        for level in [4, 3, 2]:
            x = self.decoder.decode(x, skips, level)
            feats[level - 1] = x

        # feats.keys() -- dict_keys([4, 3, 2, 1])
        # skips.keys() -- dict_keys(['conv1_2', 'pool1_LH|HL|HH',
        #    'conv2_2', 'pool2_LH|HL|HH', 'conv3_4', 'pool3_LH|HL|HH'])

        return feats

    def forward_x(self, content, style):
        H, W = style.size(2), style.size(3)
        if H % 8 != 0 or W % 8 != 0:
            H = 8 * (H // 8)
            W = 8 * (W // 8)
            style = style[:, :, 0:H, 0:W]

        content_feat = content
        content_skips: Dict[str, torch.Tensor] = {}

        style_skips: Dict[str, torch.Tensor] = {}
        style_feats = self.get_style_features(style, style_skips)
        # ==>style_skips.keys()--['conv1_2',
        # 'pool1_LH|HL|HH', 'conv2_2', 'pool2_LH|HL|HH', 'conv3_4', 'pool3_LH|HL|HH']

        for level in [1, 2, 3, 4]:
            content_feat = self.encoder.encode(content_feat, content_skips, level)

        # ==> content_skips.keys()--['conv1_2',
        # 'pool1_LH|HL|HH', 'conv2_2', 'pool2_LH|HL|HH', 'conv3_4', 'pool3_LH|HL|HH']

        for level in [4, 3, 2, 1]:
            content_feat = feature_wct(content_feat, style_feats[level])
            content_feat = self.decoder.decode(content_feat, content_skips, level)

        return content_feat.clamp(0.0, 1.0)

    def forward(self, x1, x2):
        # Need Resize ?
        B1, C1, H1, W1 = x1.size()
        if H1 > self.MAX_H or W1 > self.MAX_W:
            s = min(self.MAX_H / H1, self.MAX_W / W1)
            SH, SW = int(s * H1), int(s * W1)
            resize_x1 = F.interpolate(x1, size=(SH, SW), mode="bilinear", align_corners=False)
        else:
            resize_x1 = x1

        B2, C2, H2, W2 = x2.size()
        if H2 > self.MAX_H or W2 > self.MAX_W:
            s = min(self.MAX_H / H2, self.MAX_W / W2)
            SH, SW = int(s * H2), int(s * W2)
            resize_x2 = F.interpolate(x2, size=(SH, SW), mode="bilinear", align_corners=False)
        else:
            resize_x2 = x2

        # Need Pad ?
        PH1, PW1 = resize_x1.size(2), resize_x1.size(3)
        if PH1 % self.MAX_TIMES != 0 or PW1 % self.MAX_TIMES != 0:
            r_pad = self.MAX_TIMES - (PW1 % self.MAX_TIMES)
            b_pad = self.MAX_TIMES - (PH1 % self.MAX_TIMES)
            resize_pad_x1 = F.pad(resize_x1, (0, r_pad, 0, b_pad), mode="replicate")
        else:
            resize_pad_x1 = resize_x1

        PH2, PW2 = resize_x2.size(2), resize_x2.size(3)
        if PH2 % self.MAX_TIMES != 0 or PW2 % self.MAX_TIMES != 0:
            r_pad = self.MAX_TIMES - (PW2 % self.MAX_TIMES)
            b_pad = self.MAX_TIMES - (PH2 % self.MAX_TIMES)
            resize_pad_x2 = F.pad(resize_x2, (0, r_pad, 0, b_pad), mode="replicate")
        else:
            resize_pad_x2 = resize_x2

        y = self.forward_x(resize_pad_x1, resize_pad_x2)
        del resize_pad_x1, resize_x1, resize_pad_x2, resize_x2  # Release memory !!!

        y = y[:, :, 0:PH1, 0:PW1]  # Remove Pads
        y = F.interpolate(y, size=(H1, W1), mode="bilinear", align_corners=False)  # Remove Resize

        return y
