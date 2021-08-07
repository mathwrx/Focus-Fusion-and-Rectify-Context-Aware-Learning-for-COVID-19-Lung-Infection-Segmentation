# -*- coding: utf-8 -*-
# @File    : FFRNet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from .FeaFusion import FeaFusion
from .base import *
from .Autofocus import *


__all__ = ['FFRNetHead', 'FFRNet', 'get_model']

class FFRNetHead(nn.Module):
    def __init__(self, in_channels, out_channels, input_size, num_layers=2, has_trans=False, trans_dim=None):
        super(FFRNetHead, self).__init__()
        self.FF4 = FeaFusion(input_size, in_channels, in_channels, (3, 3), num_layers, return_all_layers=False)
        self.FF3 = FeaFusion(input_size, in_channels, in_channels, (3, 3), num_layers, return_all_layers=False)
        self.up_conv3 = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
                                      nn.BatchNorm2d(in_channels),
                                      nn.ReLU(True))
        self.FF2 = FeaFusion(input_size, in_channels, in_channels, (3, 3), num_layers, return_all_layers=False)
        self.up_conv2 = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
                                      nn.BatchNorm2d(in_channels),
                                      nn.ReLU(True))

        self.FF1 = FeaFusion((input_size[0] * 2, input_size[1] * 2), in_channels, in_channels, (3, 3), num_layers,
                            return_all_layers=False)
        self.up_conv1 = nn.Sequential(nn.Conv2d(in_channels * 2, in_channels, 1, bias=False),
                                      nn.BatchNorm2d(in_channels),
                                      nn.ReLU(True))
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, 1, 1),
                                   nn.BatchNorm2d(in_channels),
                                   nn.ReLU(True))
        self.conv_s1 = nn.Conv2d(in_channels, out_channels, 1)


    def forward(self, x1, x2, x3, x4):
        d4 = self.FF4(x4, x1, x2, x3, cross_fusion=True, mid_level=True)
        d3 = self.FF3(x3, x1, x2, x4, cross_fusion=True, mid_level=True)
        d2 = self.FF2(x2, x1, x3, x4, cross_fusion=True, mid_level=True)
        d1 = self.FF1(x1, x2, x3, x4, cross_fusion=True, mid_level=True)
        d3 = self.up_conv3(torch.cat((d3, d4), dim=1))
        d2 = self.up_conv2(torch.cat((d2, d3), dim=1))
        up_d2 = F.interpolate(d2, scale_factor=2.0, mode='bilinear', align_corners=True)
        d1 = self.up_conv1(torch.cat((d1, up_d2), dim=1))
        out = self.conv1(d1)
        out = self.conv_s1(out)

        return out


class FFRNet(BaseNet):
    def __init__(self, n_class, backbone, batchnorm, trans_rates=None, is_train=True,
                 test_size=[256, 256], trans_out_dim=256, reduce_dim=128, pooling='max', num_layers=2, **kwargs):
        super(FFRNet, self).__init__(n_class, backbone, batchnorm=batchnorm, pooling=pooling, **kwargs)
        self.trans_rate = trans_rates
        self.is_train = is_train
        self.test_size = test_size
        self.af_1 = AutoFocus(256, trans_out_dim, trans_rates, batchnorm)
        self.af_2 = AutoFocus(512, trans_out_dim, trans_rates, batchnorm)
        self.af_3 = AutoFocus(1024, trans_out_dim, trans_rates, batchnorm)
        self.af_4 = AutoFocus(2048, trans_out_dim, trans_rates, batchnorm)
        # self.reduce_conv4 = nn.Conv2d(256, reduce_dim, 1, 1, bias=False)
        # self.reduce_conv3 = nn.Conv2d(256, reduce_dim, 1, 1, bias=False)
        # self.reduce_conv2 = nn.Conv2d(256, reduce_dim, 1, 1, bias=False)
        # self.reduce_conv1 = nn.Conv2d(256, reduce_dim, 1, 1, bias=False)
        self.head = FFRNetHead(reduce_dim, 2, input_size=(
            kwargs['img_size'][0] // kwargs['output_stride'], kwargs['img_size'][1] // kwargs['output_stride']),
                               num_layers=num_layers, has_trans=True, trans_dim=trans_out_dim)

    def forward(self, x):
        imsize = x.size()[2:]
        c1, c2, c3, c4 = self.base_forward(x)
        c1 = self.af_1(c1)
        c2 = self.af_2(c2)
        c3 = self.af_3(c3)
        c4 = self.af_4(c4)
        l1 = c1
        l2 = c2
        l3 = c3
        l4 = c4
        # l1 = self.reduce_conv1(c1)
        # l2 = self.reduce_conv2(c2)
        # l3 = self.reduce_conv3(c3)
        # l4 = self.reduce_conv4(c4)
        out = self.head(l1, l2, l3, l4)

        if self.is_train:
            outputs = F.interpolate(out, imsize, mode='bilinear', align_corners=True)
            return outputs
        else:
            outputs = F.interpolate(out, self.test_size, mode='bilinear', align_corners=True)
            return outputs



def get_model(model_name='FFRNet', dataset='covid_19_seg', backbone='resnet50', root='./pretrain_models', **kwargs):
    model_dict = {'ffrnet': FFRNet}
    from encoding.dataset import datasets
    model = model_dict[model_name.lower()](datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)

    return model
