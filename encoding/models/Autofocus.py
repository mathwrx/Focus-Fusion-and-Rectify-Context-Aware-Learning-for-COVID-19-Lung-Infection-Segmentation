# -*- coding: utf-8 -*-
# @File    : Autofocus.py


import torch
import torch.nn as nn
from .FeaFusion import FeaFusion

class _BasicConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, atrous_rate, norm_layer=nn.BatchNorm2d):
        super(_BasicConv, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=atrous_rate,
                      bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.block(x)


class AutoFocus(nn.Module):
    def __init__(self, in_channels=2048, out_channels=256, atrous_rates=[6, 12, 18], norm_layer=nn.BatchNorm2d):
        super(AutoFocus, self).__init__()
        self.b0 = _BasicConv(in_channels, out_channels, kernel_size=1, padding=0, atrous_rate=1, norm_layer=norm_layer)
        self.b1 = _BasicConv(in_channels, out_channels, kernel_size=3, padding=atrous_rates[0],
                            atrous_rate=atrous_rates[0],
                            norm_layer=norm_layer)
        self.b2 = _BasicConv(in_channels, out_channels, kernel_size=3, padding=atrous_rates[1],
                            atrous_rate=atrous_rates[1],
                            norm_layer=norm_layer)
        self.b3 = _BasicConv(in_channels, out_channels, kernel_size=3, padding=atrous_rates[2],
                            atrous_rate=atrous_rates[2],
                            norm_layer=norm_layer)

        self.project = nn.Sequential(nn.Conv2d(4 * out_channels, out_channels, 1, bias=False),
                                     norm_layer(out_channels),
                                     nn.ReLU(True))
        self.LFF_foreward_head = FeaFusion((64, 64), out_channels, out_channels, (3, 3), 1, return_all_layers=False)
        self.LFF_backward_head = FeaFusion((64, 64), out_channels, out_channels, (3, 3), 1, return_all_layers=False)
        self.LFF_foreward = FeaFusion((32, 32), out_channels, out_channels, (3, 3), 1, return_all_layers=False)
        self.LFF_backward = FeaFusion((32, 32), out_channels, out_channels, (3, 3), 1, return_all_layers=False)
        self.combine = nn.Sequential(nn.Conv2d(2 * out_channels, out_channels, 1, bias=False),
                                     norm_layer(out_channels),
                                     nn.ReLU(True))

    def forward(self, x):
        feat_size = x.size()[2:]
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)

        if feat_size[0] == 64:
            x_f = self.LFF_foreward_head(feat0, feat1, feat2, feat3)
            x_b = self.LFF_backward_head(feat3, feat2, feat1, feat0)
        else:
            x_f = self.LFF_foreward(feat0, feat1, feat2, feat3)
            x_b = self.LFF_backward(feat3, feat2, feat1, feat0)

        x = torch.cat([x_f, x_b], dim=1)
        x = self.combine(x)

        return x


