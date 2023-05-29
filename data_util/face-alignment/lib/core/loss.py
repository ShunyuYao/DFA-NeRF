# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy as np
import cv2
from scipy import ndimage
import math


# class AdaptiveWingLoss(nn.Module):
#     def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1,
#                  use_target_weight=False, use_weighted_loss=False):
#         super(AdaptiveWingLoss, self).__init__()
#         self.omega = omega
#         self.theta = theta
#         self.epsilon = epsilon
#         self.alpha = alpha
#         self.use_target_weight = use_target_weight
#         self.use_weighted_loss = use_weighted_loss
#
#     def forward(self, output, target, target_weight=None, weight_mask=None):
#         # output, target (N, num_joints, H, W)
#         # target_weight: (N, num_joints, 1)
#         # weight_mask: (N, num_joints, H, W)
#         N, num_joints, H, W = output.shape
#         heatmaps_pred = output
#         heatmaps_gt = target
#
#         if self.use_target_weight:
#             heatmaps_pred = heatmaps_pred.view((N, num_joints, -1))  # .split(1, 1)
#             heatmaps_gt = heatmaps_gt.view((N, num_joints, -1))  # .split(1, 1)
#
#             heatmaps_pred = heatmaps_pred.mul(target_weight)
#             heatmaps_gt = heatmaps_gt.mul(target_weight)
#
#         loss_pos = torch.Tensor([0])
#         loss_neg = torch.Tensor([0])
#
#         y = heatmaps_gt.view(-1)
#         y_hat = heatmaps_pred.view(-1)
#         total_num = y_hat.size(0)
#         if self.use_weighted_loss:
#             weight_mask = weight_mask.view(-1)
#         theta_div_eps = self.theta / self.epsilon
#         alpha_m_y = self.alpha - y
#
#         y_abs_diff = torch.abs(y - y_hat)
#         postive_pos = y_abs_diff < self.theta
#         negtive_pos = y_abs_diff >= self.theta
#
#         alpha_m_y_neg = alpha_m_y[negtive_pos]
#
#         A = self.omega * (1 / (1 + theta_div_eps**alpha_m_y_neg)) \
#                 * alpha_m_y_neg * (theta_div_eps**(alpha_m_y_neg - 1)) * 1 / self.epsilon
#         C = self.theta * A - self.omega * torch.log1p(theta_div_eps**alpha_m_y_neg)
#         if postive_pos.sum() > 0:
#             y_diff_pos = y_abs_diff[postive_pos]
#             alpha_m_y_pos = alpha_m_y[postive_pos]
#             loss_pos = self.omega * torch.log1p((y_diff_pos / self.epsilon)**alpha_m_y_pos)
#             if self.use_weighted_loss:
#                 weight_mask_pos = weight_mask[postive_pos]
#                 loss_pos = loss_pos * weight_mask_pos
#
#         if negtive_pos.sum() > 0:
#             y_diff_neg = y_abs_diff[negtive_pos]
#             loss_neg = A * y_diff_neg - C
#             if self.use_weighted_loss:
#                 weight_mask_neg = weight_mask[negtive_pos]
#                 loss_neg = loss_neg * weight_mask_neg
#
#         loss = (loss_pos.sum() + loss_neg.sum()) / total_num
#         # torch.mean(loss_pos.mean() + loss_neg.mean()) / 2
#
#         return loss


# torch.log  and math.log is e based
class WingLoss(nn.Module):
    def __init__(self, omega=10, epsilon=2):
        super(WingLoss, self).__init__()
        self.omega = omega
        self.epsilon = epsilon

    def forward(self, pred, target):
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.omega]
        delta_y2 = delta_y[delta_y >= self.omega]
        loss1 = self.omega * torch.log(1 + delta_y1 / self.epsilon)
        C = self.omega - self.omega * math.log(1 + self.omega / self.epsilon)
        loss2 = delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))


class AWingLoss(nn.Module):

    def __init__(self, alpha=2.1, omega=14, epsilon=1, theta=0.5):
        super().__init__()
        self.alpha   = float(alpha)
        self.omega   = float(omega)
        self.epsilon = float(epsilon)
        self.theta   = float(theta)

    def forward(self, y_pred, y):
        lossMat = torch.zeros_like(y_pred)
        A = self.omega * (1/(1+(self.theta/self.epsilon)**(self.alpha-y)))*(self.alpha-y)*((self.theta/self.epsilon)**(self.alpha-y-1))/self.epsilon
        C = self.theta*A - self.omega*torch.log(1+(self.theta/self.epsilon)**(self.alpha-y))
        case1_ind = torch.abs(y-y_pred) < self.theta
        case2_ind = torch.abs(y-y_pred) >= self.theta
        lossMat[case1_ind] = self.omega*torch.log(1+torch.abs((y[case1_ind]-y_pred[case1_ind])/self.epsilon)**(self.alpha-y[case1_ind]))
        lossMat[case2_ind] = A[case2_ind]*torch.abs(y[case2_ind]-y_pred[case2_ind]) - C[case2_ind]
        return lossMat.mean()


class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1,
                 use_target_weight=False, use_weighted_loss=False):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha
        self.use_target_weight = use_target_weight
        self.use_weighted_loss = use_weighted_loss

    def forward(self, output, target, target_weight=None, weight_mask=None):
        # output, target (N, num_joints, H, W)
        # target_weight: (N, num_joints, 1)
        # weight_mask: (N, num_joints, H, W)
        N, num_joints, H, W = output.shape
        heatmaps_pred = output
        heatmaps_gt = target

        if self.use_target_weight:
            heatmaps_pred = heatmaps_pred.view((N, num_joints, -1))  # .split(1, 1)
            heatmaps_gt = heatmaps_gt.view((N, num_joints, -1))  # .split(1, 1)

            heatmaps_pred = heatmaps_pred.mul(target_weight)
            heatmaps_gt = heatmaps_gt.mul(target_weight)

        loss_pos = torch.Tensor([0])
        loss_neg = torch.Tensor([0])

        y = heatmaps_gt.view(-1)
        y_hat = heatmaps_pred.view(-1)
        total_num = y_hat.size(0)
        if self.use_weighted_loss:
            weight_mask = weight_mask.view(-1)
        theta_div_eps = self.theta / self.epsilon
        # alpha_m_y = self.alpha - y

        y_abs_diff = torch.abs(y - y_hat)
        postive_pos = y_abs_diff < self.theta
        negtive_pos = y_abs_diff >= self.theta
        y_pos = y[postive_pos]
        y_neg = y[negtive_pos]
        y_hat_pos = y_hat[postive_pos]
        y_hat_neg = y_hat[negtive_pos]

        alpha_m_y_pos = self.alpha - y_pos
        alpha_m_y_neg = self.alpha - y_neg

        A = self.omega * (1 / (1 + theta_div_eps**alpha_m_y_neg)) \
                * alpha_m_y_neg * (theta_div_eps**(alpha_m_y_neg - 1)) * 1 / self.epsilon
        C = self.theta * A - self.omega * torch.log1p(theta_div_eps**alpha_m_y_neg)
        if postive_pos.sum() > 0:
            y_diff_pos = torch.abs(y_pos - y_hat_pos)
            loss_pos = self.omega * torch.log1p((y_diff_pos / self.epsilon)**alpha_m_y_pos)
            if self.use_weighted_loss:
                weight_mask_pos = weight_mask[postive_pos]
                loss_pos = loss_pos * weight_mask_pos

        if negtive_pos.sum() > 0:
            y_diff_neg = torch.abs(y_neg - y_hat_neg)
            loss_neg = A * y_diff_neg - C
            if self.use_weighted_loss:
                weight_mask_neg = weight_mask[negtive_pos]
                loss_neg = loss_neg * weight_mask_neg

        loss = (loss_pos.sum() + loss_neg.sum()) / total_num
        # torch.mean(loss_pos.mean() + loss_neg.mean()) / 2

        return loss


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='mean')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        # print('output, target, target_weight shape: ', output.shape, target.shape, target_weight.shape)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class JointsOHKMMSELoss(nn.Module):
    def __init__(self, use_target_weight, topk=8):
        super(JointsOHKMMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='none')
        self.use_target_weight = use_target_weight
        self.topk = topk

    def ohkm(self, loss):
        ohkm_loss = 0.
        for i in range(loss.size()[0]):
            sub_loss = loss[i]
            topk_val, topk_idx = torch.topk(
                sub_loss, k=self.topk, dim=0, sorted=False
            )
            tmp_loss = torch.gather(sub_loss, 0, topk_idx)
            ohkm_loss += torch.sum(tmp_loss) / self.topk
        ohkm_loss /= loss.size()[0]
        return ohkm_loss

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = []
        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss.append(0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                ))
            else:
                loss.append(
                    0.5 * self.criterion(heatmap_pred, heatmap_gt)
                )

        loss = [l.mean(dim=1).unsqueeze(dim=1) for l in loss]
        loss = torch.cat(loss, dim=1)

        return self.ohkm(loss)

def generate_weighted_mask(heatmap, w=10, thresh=0.2, kernel_size=3):
    mask, heatmap = generate_dilation_hm_mask(heatmap, thresh, kernel_size)
    mask = mask * w + 1
    return mask

# def generate_dilation_hm_mask(heatmap, thresh=0.2, kernel_size=3, iterations=1):
#     num_joints = heatmap.shape[0]
#     heatmap = heatmap.copy()
#     dilate_kernel = np.ones((kernel_size, kernel_size), np.uint8)
#     for i in range(num_joints):
#         heatmap[i] = cv2.dilate(heatmap[i], dilate_kernel, iterations=iterations)
#     mask = (heatmap > thresh).astype(np.float32)
#     return mask, heatmap


def generate_dilation_hm_mask(heatmap, thresh=0.2, kernel_size=3):
    num_joints = heatmap.shape[0]
    weight_map = np.zeros_like(heatmap)
    heatmap = heatmap.copy()
    for i in range(num_joints):
        dilate = ndimage.grey_dilation(heatmap[i], size=(kernel_size, kernel_size))
        weight_map[i][np.where(dilate>thresh)] = 1
        heatmap[i] = dilate

    return weight_map, heatmap

if __name__ == '__main__':
    t1 = torch.zeros((4, 17, 64, 64))
    t1[:, :, :7, :7] = 0.5
    t2 = torch.zeros((4, 17, 64, 64))
    adw_loss = AdaptiveWingLoss(use_target_weight=False)
    print(t1, t2)
    print(adw_loss(t1, t2, None))
