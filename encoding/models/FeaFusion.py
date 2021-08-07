# -*- coding: utf-8 -*-
# @File    : FeaFusion.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class channel_attention(nn.Module):
    def __init__(self, in_channels, out_channals):
        super(channel_attention, self).__init__()
        # self.conv1x1 = nn.Conv2d(in_channels, out_channals, 1, 1, bias=False)
        self.global_avg = nn.AdaptiveAvgPool2d(1)
        # self.conv_down = nn.Conv2d(out_channals, out_channals // 16, kernel_size=1, stride=1, bias=False)
        # self.relu = nn.ReLU(True)
        # self.conv_up = nn.Conv2d(out_channals // 16, out_channals, kernel_size=1, stride=1, bias=False)
        self.conv_up = nn.Conv2d(out_channals, out_channals, kernel_size=1, stride=1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x = self.conv1x1(x)
        x = self.global_avg(x)
        #x = self.conv_down(x)
        #x = self.relu(x)
        x = self.conv_up(x)
        attention = self.sigmoid(x)

        return attention

class FeaFusion(nn.Module):
    def __init__(self, input_size, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(FeaFusion, self).__init__()
        self.input_size = input_size
        self.sigmoid = nn.Sigmoid()
        self.mConv_1 = nn.Conv2d(input_dim, input_dim, 1, bias=False)
        self.mConv_2 = nn.Conv2d(input_dim, input_dim, 1, bias=False)
        self.mConv_3 = nn.Conv2d(input_dim, input_dim, 1, bias=False)
        self.fConv_1 = nn.Conv2d(input_dim, input_dim, 1, bias=False)
        self.fConv_2 = nn.Conv2d(input_dim, input_dim, 1, bias=False)
        self.fusion_2 = nn.Conv2d(input_dim * 2, input_dim, 1, bias=False)
        self.fusion_3 = nn.Conv2d(input_dim * 2, input_dim, 1, bias=False)
        self.fusion_4 = nn.Conv2d(input_dim * 2, input_dim, 1, bias=False)
        self.g_conv = nn.Conv2d(4*input_dim, input_dim, 1, bias=False)
        self.att2 = nn.Conv2d(2 * input_dim, input_dim, 1, bias=False)
        self.att3 = nn.Conv2d(2 * input_dim, input_dim, 1, bias=False)
        self.att4 = nn.Conv2d(2 * input_dim, input_dim, 1, bias=False)
        self.channel_att2 = channel_attention(input_dim, input_dim)
        self.channel_att3 = channel_attention(input_dim, input_dim)
        self.channel_att4 = channel_attention(input_dim, input_dim)

    def forward(self, s1, s2, s3, s4, cross_fusion=True, mid_level=False):
        if s1.size()[2:] != self.input_size:
            s1 = F.interpolate(s1, self.input_size, mode='bilinear', align_corners=True)
        if s2.size()[2:] != self.input_size:
            s2 = F.interpolate(s2, self.input_size, mode='bilinear', align_corners=True)
        if s3.size()[2:] != self.input_size:
            s3 = F.interpolate(s3, self.input_size, mode='bilinear', align_corners=True)
        if s4.size()[2:] != self.input_size:
            s4 = F.interpolate(s4, self.input_size, mode='bilinear', align_corners=True)
        if mid_level:
            global_fea = self.g_conv(torch.cat([s1, s2, s3, s4], dim=1))

            ggl_2 = self.att2(torch.cat([global_fea, s2], dim=1))
            ggl_3 = self.att3(torch.cat([global_fea, s3], dim=1))
            ggl_4 = self.att4(torch.cat([global_fea, s4], dim=1))

            channel_attention_2 = self.channel_att2(ggl_2)
            channel_attention_3 = self.channel_att3(ggl_3)
            channel_attention_4 = self.channel_att4(ggl_4)

            ch_2 = channel_attention_2 * ggl_2
            ch_3 = channel_attention_3 * ggl_3
            ch_4 = channel_attention_4 * ggl_4
            out = s1 + (1 - self.sigmoid(s1)) * (ch_2 + ch_3 + ch_4)

        else:
            feature_1 = s1
            memory_1 = (1 - self.sigmoid(self.mConv_1(feature_1))) * feature_1
            feature_2 = s2 + (1 - self.sigmoid(s2)) * self.fusion_2(torch.cat([memory_1, feature_1], dim=1))
            memory_2 = (1 - self.sigmoid(self.mConv_2(feature_2))) * feature_2 + self.sigmoid(self.fConv_1(memory_1)) * memory_1
            feature_3 = s3 + (1 - self.sigmoid(s3)) * self.fusion_3(torch.cat([memory_2, feature_2], dim=1))
            memory_3 = (1 - self.sigmoid(self.mConv_3(feature_3))) * feature_3 + self.sigmoid(self.fConv_2(memory_2)) * memory_2
            out = s4 + (1 - self.sigmoid(s4)) * self.fusion_4(torch.cat([memory_3, feature_3], dim=1))

        return out