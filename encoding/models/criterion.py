# -*- coding: utf-8 -*-
# @File    : criterion.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import ndimage


class SegmentationLoss(nn.CrossEntropyLoss):
    def __init__(self, ce_weight=1.0, dice_weight=1.0, weight=None, size_average=None, ignore_index=-1):
        super(SegmentationLoss, self).__init__(weight, size_average, ignore_index)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.DiceLoss = DiceLoss_with_softmax()
        self.KLdivLoss = nn.KLDivLoss()
    def forward(self, input, target):
        ce_loss = super(SegmentationLoss, self).forward(input, target)
        dice_loss = self.DiceLoss(input, target.float())
        predict = input.softmax(1)
        _, predict = predict.max(dim=1)
        input_t = torch.from_numpy(ndimage.distance_transform_edt(predict.cpu()))
        target_t = torch.from_numpy(ndimage.distance_transform_edt(target.cpu()))
        input_t = input_t.cuda()
        target_t = target_t.cuda()
        structure_loss = self.KLdivLoss(input_t.float(), target_t.float())  # + self.KLdivLoss(target_t.float(), input_t.float())
        loss = self.ce_weight * ce_loss + self.dice_weight * dice_loss + structure_loss
        return loss


class FocalLoss(nn.Module):
    r"""
        This criterion is a implemenation of Focal Loss, which is proposed in
        Focal Loss for Dense Object Detection.

            Loss(x, class) = - \alpha (1-softmax(x)[class])^gamma \log(softmax(x)[class])

        The losses are averaged across observations for each minibatch.

        Args:
            alpha(1D Tensor, Variable) : the scalar factor for this criterion
            gamma(float, double) : gamma > 0; reduces the relative loss for well-classiﬁed examples (p > .5),
                                   putting more focus on hard, misclassiﬁed examples
            size_average(bool): By default, the losses are averaged over observations for each minibatch.
                                However, if the field size_average is set to False, the losses are
                                instead summed for each minibatch.


    """

    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = torch.autograd.Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, torch.autograd.Variable):
                self.alpha = alpha
            else:
                self.alpha = torch.autograd.Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = F.softmax(inputs, 1)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = torch.autograd.Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)
        # print(class_mask)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P * class_mask).sum(1).view(-1, 1)

        log_p = probs.log()
        # print('probs size= {}'.format(probs.size()))
        # print(probs)

        batch_loss = -alpha * (torch.pow((1 - probs), self.gamma)) * log_p
        # print('-----bacth_loss------')
        # print(batch_loss)

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss

class DiceLoss_with_sigmoid(nn.Module):
    def __init__(self):
        super(DiceLoss_with_sigmoid, self).__init__()

    def forward(self, predict, target):
        predict = F.sigmoid(predict)
        predict[predict >= 0.5] = 1
        predict[predict != 1] = 0
        N = target.size(0)
        smooth = 1

        predict_flat = predict.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = predict_flat * target_flat

        loss = 2. * (intersection.sum(1) + smooth) / (predict_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss


class DiceLoss_with_softmax(nn.Module):
    def __init__(self):
        super(DiceLoss_with_softmax, self).__init__()

    def forward(self, predict, target):
        predict = F.softmax(predict, dim=1)
        # predict = predict.max(dim=1)[1].float()
        predict = predict[:, 1]
        N = target.size(0)
        smooth = 1

        predict_flat = predict.view(N, -1)
        target_flat = target.view(N, -1)

        intersection = predict_flat * target_flat

        loss = 2. * (intersection.sum(1) + smooth) / (predict_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N

        return loss
