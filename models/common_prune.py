# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Common modules
"""

import json
import math
import platform
import warnings
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
from PIL import Image
from torch.cuda import amp

from utils.datasets import exif_transpose, letterbox
from utils.general import (LOGGER, check_requirements, check_suffix, check_version, colorstr, increment_path,
                           make_divisible, non_max_suppression, scale_coords, xywh2xyxy, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import copy_attr, time_sync
from models.common import Conv
import torch.nn.functional as F


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        out = F.relu6(x + 3., self.inplace) / 6.
        return out * x



class BottleneckPruned(nn.Module):
    # Pruned bottleneck
    def __init__(self, cv1in, cv1out, cv2out, shortcut=True, g=1):  # ch_in, ch_out, shortcut, groups, expansion
        super(BottleneckPruned, self).__init__()
        self.cv1 = Conv(cv1in, cv1out, 1, 1)
        self.cv2 = Conv(cv1out, cv2out, 3, 1, g=g)
        self.add = shortcut and cv1in == cv2out

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3Pruned(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, cv1in, cv1out, cv2out, cv3out, bottle_args, n=1, shortcut=True, g=1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3Pruned, self).__init__()
        cv3in = bottle_args[-1][-1]
        self.cv1 = Conv(cv1in, cv1out, 1, 1)
        self.cv2 = Conv(cv1in, cv2out, 1, 1)
        self.cv3 = Conv(cv3in+cv2out, cv3out, 1)
        self.m = nn.Sequential(*[BottleneckPruned(*bottle_args[k], shortcut, g) for k in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class SPPFPruned(nn.Module):
    # Spatial pyramid pooling layer used in YOLOv3-SPP
    def __init__(self, cv1in, cv1out, cv2out, k=5):
        super(SPPFPruned, self).__init__()
        self.cv1 = Conv(cv1in, cv1out, 1, 1)
        self.cv2 = Conv(cv1out * 4, cv2out, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class C3CAPruned(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, cv1in, cv1out, cv2out, cv3out, bottle_args, n=1, shortcut=True, g=1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3CAPruned, self).__init__()
        cv3in = bottle_args[-1][-1]
        self.cv1 = Conv(cv1in, cv1out, 1, 1)
        self.cv2 = Conv(cv1in, cv2out, 1, 1)
        self.cv3 = Conv(cv3in+cv2out, cv3out, 1)
        self.m = nn.Sequential(*[CABottleneckpruned(*bottle_args[k], shortcut, g) for k in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class CABottleneckpruned(nn.Module):
    # Standard bottleneck
    def __init__(self, cv1in, cv1out, cv2out, shortcut=True, g=1, e=0.5, ratio=32):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        # c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(cv1in, cv1out, 1, 1)
        self.cv2 = Conv(cv1out, cv2out, 3, 1, g=g)
        self.add = shortcut and cv1in == cv2out
        # self.ca=CoordAtt(c1,c2,ratio)
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, cv1in // ratio)
        self.conv1 = nn.Conv2d(cv1in, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, cv2out, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, cv2out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # print('x:',x.shape)
        # print('self.cv1(x)',self.cv1(x).shape)
        x1 = self.cv2(self.cv1(x))
        # print('x1:',x1.shape)
        n, c, h, w = x.size()
        # c*1*W
        x_h = self.pool_h(x1)
        # c*H*1
        # C*1*h
        x_w = self.pool_w(x1).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        # C*1*(h+w)
        # print('y',y.shape)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        # print('y', y.shape)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        # print('x_h', x_h.shape)
        # print('x_w', x_w.shape)
        x_w = x_w.permute(0, 1, 3, 2)
        # print('x_w',x_w.shape)
        a_h = self.conv_h(x_h).sigmoid()
        # print('a_h',a_h.shape)
        a_w = self.conv_w(x_w).sigmoid()
        # print('a_w', a_w.shape)
        out = x1 * a_w * a_h

        # out=self.ca(x1)*x1
        return x + out if self.add else out


class ECABottleneckprune(nn.Module):
    # Standard bottleneck
    def __init__(self, cv1in, cv1out, cv2out, shortcut=True, g=1, e=0.5, ratio=16, k_size=3):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        # c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(cv1in, cv1out, 1, 1)
        self.cv2 = Conv(cv1out, cv2out, 3, 1, g=g)
        self.add = shortcut and cv1in == cv2out
        # self.eca=ECA(c1,c2)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.cv2(self.cv1(x))
        # out=self.eca(x1)*x1
        y = self.avg_pool(x1)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        out = x1 * y.expand_as(x1)

        return x + out if self.add else out

class C3ECAPruned(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, cv1in, cv1out, cv2out, cv3out, bottle_args, n=1, shortcut=True, g=1):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(C3ECAPruned, self).__init__()
        cv3in = bottle_args[-1][-1]
        self.cv1 = Conv(cv1in, cv1out, 1, 1)
        self.cv2 = Conv(cv1in, cv2out, 1, 1)
        self.cv3 = Conv(cv3in+cv2out, cv3out, 1)
        self.m = nn.Sequential(*[ECABottleneckprune(*bottle_args[k], shortcut, g) for k in range(n)])

    def forward(self, x):
        # print('torch.cat((self.m(self.cv1(x)), self.cv2(x)',torch.cat((self.m(self.cv1(x)), self.cv2(x)),dim=1).shape)
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


