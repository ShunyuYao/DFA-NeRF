import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.utils import to_dense_batch
from nnutils import make_conv_2d, make_upscale_2d, make_downscale_2d, ResBlock2d, Identity
from warp_utils import get_occu_mask_bidirection, get_ssv_weights
import os
import numpy as np
import math
from math import ceil
import pdb
from raft import RAFT_ALL
from models.point_render_func import Render
import time
import cv2
from PIL import Image
from point_render_utils import rb_colormap
import multiprocessing


class NeuralNRT(nn.Module):
    def __init__(self, opt, path=None, device="cuda:0"):
        super(NeuralNRT, self).__init__()
        self.opt = opt
        self.CorresPred = RAFT_ALL(opt)
        self.LossClass = LossClass(opt)
        #init
        if self.opt.isTrain:
            if path is not None:
                data = torch.load(path,map_location='cpu')
                if 'state_dict' in data.keys():
                    self.CorresPred.load_state_dict(data['state_dict'])
                    print("load done")
                else:
                    self.CorresPred.load_state_dict({k.replace('module.', ''):v for k,v in data.items()})
                    print("load done")

        self.Nc = self.opt.num_corres
        self.lambda_2d = self.opt.lambda_2d
        self.lambda_depth = self.opt.lambda_depth
        self.lambda_reg = self.opt.lambda_reg
        self.num_adja = self.opt.num_adja
        
        #(N*2*H*W)
        # mesh grid 
        self.opt.width = self.opt.width
        self.opt.height = self.opt.height

    def construct_corres(self,src_input,tar_input,src_crop_im,tar_crop_im,src_mask,tar_mask, Crop_param):

        N=src_input.shape[0]
        src_points = src_input[:, 3:, :, :]
        tar_points = tar_input[:, 3:, :, :]

        flow_fw_crop, feature_fw_crop = self.CorresPred(src_crop_im, tar_crop_im, iters=self.opt.iters)
        flow_bw_crop, feature_bw_crop = self.CorresPred(tar_crop_im, src_crop_im, iters=self.opt.iters)

        xx = torch.arange(self.opt.width, device=src_input.device).view(1,-1).repeat(self.opt.height,1)
        yy = torch.arange(self.opt.height, device=src_input.device).view(-1,1).repeat(1,self.opt.width)
        xx = xx.view(1,1,self.opt.height,self.opt.width).repeat(N,1,1,1)
        yy = yy.view(1,1,self.opt.height,self.opt.width).repeat(N,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        scale_value = torch.ones((1, 2, 1, 1), device=src_input.device)
        scale_value[:, 0, :, :] *= (2.0 / max(self.opt.width-1,1))
        scale_value[:, 1, :, :] *= (2.0 / max(self.opt.height-1,1))

        grid_crop = grid[:, :, :self.opt.crop_height, :self.opt.crop_width]

        leftup1 = torch.cat((Crop_param[:, 0:1, 0], Crop_param[:, 2:3, 0]), 1)[:, :, None, None]
        leftup2 = torch.cat((Crop_param[:, 4:5, 0], Crop_param[:, 6:7, 0]), 1)[:, :, None, None]

        scale1 = torch.cat(((Crop_param[:, 1:2, 0]-Crop_param[:, 0:1, 0]).float() / self.opt.crop_width, (Crop_param[:, 3:4, 0]-Crop_param[:, 2:3, 0]).float() / self.opt.crop_height), 1)[:, :, None, None]
        scale2 = torch.cat(((Crop_param[:, 5:6, 0]-Crop_param[:, 4:5, 0]).float() / self.opt.crop_width, (Crop_param[:, 7:8, 0]-Crop_param[:, 6:7, 0]).float() / self.opt.crop_height), 1)[:, :, None, None]

        flow_fw_lists = []
        flow_bw_lists = []

        for j in range(len(flow_bw_crop)):
            flow_fwj = torch.zeros(grid.shape, device=grid.device)
            flow_bwj = torch.zeros(grid.shape, device=grid.device)
            flow_fw_cropj = (scale2 - scale1) * grid_crop + scale2 * flow_fw_crop[j]
            flow_bw_cropj = (scale1 - scale2) * grid_crop + scale1 * flow_bw_crop[j]

            for i in range(N):
                flow_fw_cropi = F.interpolate(flow_fw_cropj[i:(i+1)], ((Crop_param[i, 3, 0]-Crop_param[i, 2, 0]).item(), (Crop_param[i, 1, 0]-Crop_param[i, 0, 0]).item()), mode='bilinear')
                flow_fw_cropi = flow_fw_cropi + (leftup2 - leftup1)[i:(i+1), :, :, :]                
                flow_bw_cropi = F.interpolate(flow_bw_cropj[i:(i+1)], ((Crop_param[i, 7, 0]-Crop_param[i, 6, 0]).item(), (Crop_param[i, 5, 0]-Crop_param[i, 4, 0]).item()), mode='bilinear')
                flow_bw_cropi = flow_bw_cropi + (leftup1 - leftup2)[i:(i+1), :, :, :]
                flow_fwj[i, :, Crop_param[i, 2, 0]:Crop_param[i, 3, 0], Crop_param[i, 0, 0]:Crop_param[i, 1, 0]] = flow_fw_cropi[0]
                flow_bwj[i, :, Crop_param[i, 6, 0]:Crop_param[i, 7, 0], Crop_param[i, 4, 0]:Crop_param[i, 5, 0]] = flow_bw_cropi[0]
            flow_fw_lists.append(flow_fwj)
            flow_bw_lists.append(flow_bwj)

        flow_fw = flow_fw_lists[-1]
        flow_bw = flow_bw_lists[-1]
        corres = grid + flow_fw
        src_mask = src_mask.view(N, -1)
        corres_bw = grid + flow_bw
        corres_bw = corres_bw.view(N, 2, -1)
        outrange_mask_bw = torch.isnan(corres_bw[:, 0, :]) | torch.isnan(corres_bw[:, 1, :]) | \
        (corres_bw[:, 0, :]<=0) | (corres_bw[:, 0, :]>=(self.opt.width-1)) | (corres_bw[:, 1, :]<=0) | \
        (corres_bw[:, 1, :]>=(self.opt.height-1))
        outrange_mask_bw = outrange_mask_bw | (~tar_mask.view(N, -1))
        corres_bw[outrange_mask_bw.unsqueeze(1).repeat(1, 2, 1)] = 1.0
        src_mask_list_1 = (corres_bw[:, 1, :].floor() * self.opt.width + corres_bw[:, 0, :].floor()).long()
        src_mask_list_2 = (corres_bw[:, 1, :].floor() * self.opt.width + corres_bw[:, 0, :].ceil()).long()
        src_mask_list_3 = (corres_bw[:, 1, :].ceil() * self.opt.width + corres_bw[:, 0, :].floor()).long()
        src_mask_list_4 = (corres_bw[:, 1, :].ceil() * self.opt.width + corres_bw[:, 0, :].ceil()).long()
        bw_outrange_mask = outrange_mask_bw | (~torch.gather(src_mask, 1, src_mask_list_1)) | \
        (~torch.gather(src_mask, 1, src_mask_list_2)) | (~torch.gather(src_mask, 1, src_mask_list_3)) | \
        (~torch.gather(src_mask, 1, src_mask_list_4))

        tar_mask = tar_mask.view(N, -1)
        corres = corres.view(N, 2, -1)

        outrange_mask_fw = torch.isnan(corres[:, 0, :]) | torch.isnan(corres[:, 1, :]) | (corres[:, 0, :]<=0) | (corres[:, 0, :]>=(self.opt.width-1)) | (corres[:, 1, :]<=0) | (corres[:, 1, :]>=(self.opt.height-1))
        outrange_mask = outrange_mask_fw | (~src_mask)
        corres[outrange_mask.unsqueeze(1).repeat(1, 2, 1)] = 1.0
        tar_mask_list_1 = (corres[:, 1, :].floor() * self.opt.width + corres[:, 0, :].floor()).long()
        tar_mask_list_2 = (corres[:, 1, :].floor() * self.opt.width + corres[:, 0, :].ceil()).long()
        tar_mask_list_3 = (corres[:, 1, :].ceil() * self.opt.width + corres[:, 0, :].floor()).long()
        tar_mask_list_4 = (corres[:, 1, :].ceil() * self.opt.width + corres[:, 0, :].ceil()).long()
        fw_outrange_mask = (outrange_mask) | (~torch.gather(tar_mask, 1, tar_mask_list_1)) | (~torch.gather(tar_mask, 1, tar_mask_list_2)) | \
            (~torch.gather(tar_mask, 1, tar_mask_list_3)) | (~torch.gather(tar_mask, 1, tar_mask_list_4))

        return flow_fw_lists, flow_bw_lists, fw_outrange_mask, bw_outrange_mask

    def forward(self, src_input, tar_input, src_crop_im, tar_crop_im, src_mask, tar_mask, Crop_param):
        flow_lists, flow_bw_lists, fw_outrange_mask, bw_outrange_mask=\
            self.construct_corres(src_input,tar_input,src_crop_im,tar_crop_im,src_mask,tar_mask, Crop_param) 

        loss_ph, loss_smooth = self.LossClass(flow_lists, flow_bw_lists, src_input, tar_input, src_mask, fw_outrange_mask, bw_outrange_mask, tar_mask)
        return loss_ph.unsqueeze(0), loss_smooth.unsqueeze(0)

########################################################################
#######Loss
########################################################################

def euler2rot(euler_angle):
    batch_size = euler_angle.shape[0]
    one = torch.ones((batch_size, 1, 1), device = (euler_angle.device))
    zero = torch.zeros((batch_size, 1, 1), device = (euler_angle.device))
    theta = euler_angle[:, 0].reshape(-1, 1, 1)
    phi = euler_angle[:, 1].reshape(-1, 1, 1)
    psi = euler_angle[:, 2].reshape(-1, 1, 1)
    rot_x = torch.cat((
        torch.cat((one, zero, zero), 1),
        torch.cat((zero, theta.cos(), theta.sin()), 1),
        torch.cat((zero, -theta.sin(), theta.cos()), 1),
    ), 2)
    rot_y = torch.cat((
        torch.cat((phi.cos(), zero, -phi.sin()), 1),
        torch.cat((zero, one, zero), 1),
        torch.cat((phi.sin(), zero, phi.cos()), 1),
    ), 2)
    rot_z = torch.cat((
        torch.cat((psi.cos(), -psi.sin(), zero), 1),
        torch.cat((psi.sin(), psi.cos(), zero), 1),
        torch.cat((zero, zero, one), 1)
    ), 2)
    return torch.bmm(rot_z, torch.bmm(rot_y, rot_x))

class LossClass(nn.Module):
    def __init__(self, opt):
        super(LossClass, self).__init__()
        self.opt = opt

    # Crecit: https://github.com/simonmeister/UnFlow/blob/master/src/e2eflow/core/losses.py
    def charbonnier_loss(self, x, mask=None, gamma_exp=0.45, beta=1.0, epsilon=0.001):
        error = torch.pow(torch.square(x * beta) + epsilon**2, gamma_exp)
        if mask is not None:
            error = torch.mul(mask, error)
            # print("mask_sum:", mask.sum())
            # return error.sum()/(mask.sum().float())
            return error.sum()/(mask.sum().float())
        return error.mean()

    def TernaryLoss(self, im, im_warp, max_distance=1):
        patch_size = 2 * max_distance + 1

        def _rgb_to_grayscale(image):
            grayscale = image[:, 0, :, :] * 0.2989 + \
                        image[:, 1, :, :] * 0.5870 + \
                        image[:, 2, :, :] * 0.1140
            return grayscale.unsqueeze(1)

        def _ternary_transform(image):
            intensities = _rgb_to_grayscale(image) * 255
            out_channels = patch_size * patch_size
            w = torch.eye(out_channels).view((out_channels, 1, patch_size, patch_size))
            weights = w.type_as(im)
            patches = F.conv2d(intensities, weights, padding=max_distance)
            transf = patches - intensities
            transf_norm = transf / torch.sqrt(0.81 + torch.pow(transf, 2))
            return transf_norm

        def _hamming_distance(t1, t2):
            dist = torch.pow(t1 - t2, 2)
            dist_norm = dist / (0.1 + dist)
            dist_mean = torch.mean(dist_norm, 1, keepdim=True)  # instead of sum
            return dist_mean

        def _valid_mask(t, padding):
            n, _, h, w = t.size()
            inner = torch.ones(n, 1, h - 2 * padding, w - 2 * padding).type_as(t)
            mask = F.pad(inner, [padding] * 4)
            return mask

        t1 = _ternary_transform(im)
        t2 = _ternary_transform(im_warp)
        dist = _hamming_distance(t1, t2)
        mask = _valid_mask(im, max_distance)
        return dist, mask


    def SSIM(self, x, y, md=1):
        patch_size = 2 * md + 1
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        mu_x = nn.AvgPool2d(patch_size, 1, 0)(x)
        mu_y = nn.AvgPool2d(patch_size, 1, 0)(y)
        mu_x_mu_y = mu_x * mu_y
        mu_x_sq = mu_x.pow(2)
        mu_y_sq = mu_y.pow(2)

        sigma_x = nn.AvgPool2d(patch_size, 1, 0)(x * x) - mu_x_sq
        sigma_y = nn.AvgPool2d(patch_size, 1, 0)(y * y) - mu_y_sq
        sigma_xy = nn.AvgPool2d(patch_size, 1, 0)(x * y) - mu_x_mu_y

        SSIM_n = (2 * mu_x_mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x_sq + mu_y_sq + C1) * (sigma_x + sigma_y + C2)
        SSIM = SSIM_n / SSIM_d
        dist = torch.clamp((1 - SSIM) / 2, 0, 1)
        return dist


    def gradient(self, data):
        D_dy = data[:, :, 1:] - data[:, :, :-1]
        D_dx = data[:, :, :, 1:] - data[:, :, :, :-1]
        return D_dx, D_dy


    def smooth_grad_1st(self, flo, image, alpha):
        img_dx, img_dy = self.gradient(image)
        weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
        weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)

        dx, dy = self.gradient(flo)

        loss_x = weights_x * dx.abs() / 2.
        loss_y = weights_y * dy.abs() / 2

        return loss_x.mean() / 2. + loss_y.mean() / 2.


    def smooth_grad_2nd(self, flo, image, alpha):
        img_dx, img_dy = self.gradient(image)
        weights_x = torch.exp(-torch.mean(torch.abs(img_dx), 1, keepdim=True) * alpha)
        weights_y = torch.exp(-torch.mean(torch.abs(img_dy), 1, keepdim=True) * alpha)

        dx, dy = self.gradient(flo)
        dx2, dxdy = self.gradient(dx)
        dydx, dy2 = self.gradient(dy)

        loss_x = weights_x[:, :, :, 1:] * dx2.abs()
        loss_y = weights_y[:, :, 1:, :] * dy2.abs()

        return loss_x.mean() / 2. + loss_y.mean() / 2.

    def loss_photomatric(self, im1_scaled, im1_recons, mask1, opt):
        loss = []
        if opt .w_l1 > 0:
            loss += [opt.w_l1 * self.charbonnier_loss((im1_scaled - im1_recons).abs(), mask1, opt.gamma_exp, opt.beta, opt.epsilon)]

        if opt .w_ssim > 0:
            loss += [opt.w_ssim * self.charbonnier_loss(self.SSIM(im1_recons * mask1, im1_scaled * mask1), None, opt.gamma_exp, opt.beta, opt.epsilon)]

        if opt .w_ternary > 0:
            dist, mask = self.TernaryLoss(torch.mul(im1_recons.contiguous(), mask1.contiguous()), torch.mul(im1_scaled.contiguous(), mask1.contiguous()))
            loss += [opt.w_ternary * self.charbonnier_loss(dist, torch.mul(mask, mask1), opt.gamma_exp, opt.beta, opt.epsilon)]

        loss = sum(loss) / len(loss)
        #loss = sum([l.mean() for l in loss]) / mask1.mean()
        # print("ph:", loss)
        return loss

    def ph_loss(self, src_color, corres_color, object_mask, out_range_mask=None):
        # mask = torch.mul(object_mask.float(), weights)
        if out_range_mask is not None:
            mask = (object_mask & (~out_range_mask)).float()
        else:
            mask = object_mask.float()
        # loss = torch.sum(torch.mul(mask, (src_color-corres_color)**2)) / mask.sum()
        loss = self.loss_photomatric(src_color, corres_color, mask, self.opt)
        # print("mask_sum:", object_mask.sum(), mask.sum())
        return loss

    def parsing_loss(self, src_parsing_mask, corres_parsing_mask, object_mask, out_range_mask):
        # mask = torch.mul(object_mask.float(), weights)
        mask = (object_mask & (~out_range_mask)).float()
        loss = 0.0
        weights = [5.0]*19
        weights[0] = 0
        weights[18] = 1
        weights[1] = 1
        weights[13] = 1
        weights[14] = 1
        weights[16] = 1
        weights[17] = 1

        for i in range(19):
            loss += weights[i]*torch.mul(mask, ((src_parsing_mask==i).float() - (corres_parsing_mask==i).float()).abs()).sum()
        return loss / 19.0 / mask.sum()

    # to avoid the trivial solution where all pixels become occluded
    def occlusion_loss(self, occu_mask, mask):
        # return (occu_mask * mask.float()).sum() / mask.sum()
        return occu_mask.float().mean()

    def inmask_loss(self, out_mask, mask):
        # return (out_mask.float() * mask.view(mask.shape[0], -1).float()).sum() / mask.sum()
        return out_mask.float().mean()

    def smooth_loss(self, flow, src_color):
        if self.opt.smooth_2nd:
            func_smooth = self.smooth_grad_2nd
        else:
            func_smooth = self.smooth_grad_1st
        loss = func_smooth(flow, src_color, self.opt.alpha).mean()
        # print("smooth:", loss)
        return loss

    def weights_loss(self, weights, bw_outrange_mask, tar_mask):
        # mask = (bw_outrange_mask & tar_mask).float()
        mask = (bw_outrange_mask).float()
        loss = (torch.mul(mask, weights).abs()).sum() / mask.sum()
        return loss

    def get_neighbor_index(self, vertices: "(B, vertice_num, 3)", mask: "(B, vertice_num)", neighbor_num: int):
        """
        Return: (B, vertice_num, neighbor_num)
        """
        bs, v, _ = vertices.size()
        device = vertices.device
        inner = torch.bmm(vertices, vertices.transpose(1, 2)) #(bs, v, v)
        quadratic = torch.sum(vertices**2, dim= 2) #(bs, v)
        distance = inner * (-2) + quadratic.unsqueeze(1) + quadratic.unsqueeze(2)
        distance[(mask[:, :, None]==0).repeat(1, 1, v)] = 1e9
        distance[(mask[:, None, :]==0).repeat(1, v, 1)] = 1e9
        neighbor_index = torch.topk(distance, k= neighbor_num + 1, dim= -1, largest= False)[1]
        neighbor_index = neighbor_index[:, :, 1:]
        return neighbor_index

    def indexing_neighbor(self, tensor: "(bs, vertice_num, dim)", index: "(bs, vertice_num, neighbor_num)" ):
        """
        Return: (bs, vertice_num, neighbor_num, dim)
        """
        bs, v, n = index.size()
        id_0 = torch.arange(bs).view(-1, 1, 1)
        tensor_indexed = tensor[id_0, index]
        return tensor_indexed

    def ComputeDFF(self, view_id, ori_c_img, tgt_c_img, ori_d_img, tgt_d_img, proj_tar_pts_x, proj_tar_pts_y, tgt_pts_bool, mask, mean_img):
        N = ori_c_img.shape[0]
        proj_tar_pts_x[~tgt_pts_bool[:, :, None]] = 10000
        proj_tar_pts_y[~tgt_pts_bool[:, :, None]] = 10000

        xmin = torch.min(proj_tar_pts_x, 1)[0][:, 0]
        ymin = torch.min(proj_tar_pts_y, 1)[0][:, 0]

        proj_tar_pts_x[~tgt_pts_bool[:, :, None]] = -10000
        proj_tar_pts_y[~tgt_pts_bool[:, :, None]] = -10000
        xmax = torch.max(proj_tar_pts_x, 1)[0][:, 0]
        ymax = torch.max(proj_tar_pts_y, 1)[0][:, 0]

        b_length_ = (1.1*torch.sqrt((xmax-xmin)**2 + (ymax-ymin)**2)).int()
        x_min_ = ((xmin+xmax)/2.0 - b_length_/2.0 + 0.5).int()
        y_min_ = ((ymin+ymax)/2.0 - b_length_/2.0+b_length_/15.0 + 0.5).int()

        H = ori_c_img.shape[1]
        W = ori_c_img.shape[2]
        oxmin = torch.clamp(x_min_, 0, W-1)
        oymin = torch.clamp(y_min_, 0, H-1)
        oxmax = torch.clamp(x_min_+b_length_-1, 0, W-1)
        oymax = torch.clamp(y_min_+b_length_-1, 0, H-1)

        txmin = (oxmin-x_min_)
        tymin = (oymin-y_min_)
        txmax = (oxmax-x_min_)
        tymax = (oymax-y_min_)
        temp_oris = torch.zeros((N, 3, 224, 224), device=ori_c_img.device)
        temp_tgts = torch.zeros((N, 3, 224, 224), device=ori_c_img.device)
        feature_masks = torch.zeros((N, 224, 224), device=ori_c_img.device)

        output_depth_tmp = 0.0
        output_color_tmp = 0.0

        for i in range(N):
            temp_ori = torch.zeros((1, b_length_[i], b_length_[i], 3), device=ori_c_img.device)
            temp_tgt = torch.zeros((1, b_length_[i], b_length_[i], 3), device=ori_c_img.device)
            feature_mask = torch.zeros((1, b_length_[i], b_length_[i], 1), device=ori_c_img.device)
            temp_ori[0, tymin[i]:(tymax[i]+1), txmin[i]:(txmax[i]+1), :] = ori_c_img[i, oymin[i]:(oymax[i]+1), oxmin[i]:(oxmax[i]+1),:]
            temp_tgt[0, tymin[i]:(tymax[i]+1), txmin[i]:(txmax[i]+1), :] = tgt_c_img[i, oymin[i]:(oymax[i]+1), oxmin[i]:(oxmax[i]+1),:]
            feature_mask[0, tymin[i]:(tymax[i]+1), txmin[i]:(txmax[i]+1), :] = mask[i, oymin[i]:(oymax[i]+1), oxmin[i]:(oxmax[i]+1), :]

            mask_temp = mask[i, oymin[i]:(oymax[i]+1), oxmin[i]:(oxmax[i]+1), :]
            output_depth_tmp += torch.sum(torch.mul(mask_temp, torch.abs(ori_d_img[i, oymin[i]:(oymax[i]+1), oxmin[i]:(oxmax[i]+1),:]-tgt_d_img[i, oymin[i]:(oymax[i]+1), oxmin[i]:(oxmax[i]+1),:])))/mask_temp.sum()
            # output_color_tmp += torch.sum(torch.mul(mask_temp, torch.abs(ori_c_img[i, oymin[i]:(oymax[i]+1), oxmin[i]:(oxmax[i]+1),:]-tgt_c_img[i, oymin[i]:(oymax[i]+1), oxmin[i]:(oxmax[i]+1),:])))/mask_temp.sum()
            output_color_tmp += self.ph_loss(ori_c_img[i:(i+1), oymin[i]:(oymax[i]+1), oxmin[i]:(oxmax[i]+1),:].permute(0, 3, 1 ,2), tgt_c_img[i:(i+1), oymin[i]:(oymax[i]+1), oxmin[i]:(oxmax[i]+1),:].permute(0, 3, 1 ,2), mask[i:(i+1), oymin[i]:(oymax[i]+1), oxmin[i]:(oxmax[i]+1), 0][:, None, :, :])
            temp_ori = torch.nn.functional.interpolate(temp_ori.permute(0, 3, 1, 2), (224, 224), mode='bilinear')
            temp_tgt = torch.nn.functional.interpolate(temp_tgt.permute(0, 3, 1, 2), (224, 224), mode='bilinear')
            feature_mask = torch.nn.functional.interpolate(feature_mask.permute(0, 3, 1, 2), (224, 224), mode='bilinear')[:, 0, :, :]
            temp_oris[i] = temp_ori[0]
            temp_tgts[i] = temp_tgt[0]
            feature_masks[i] = feature_mask[0]
        
        # global print_num
        # cv2.imwrite("./debug/"+str(view_id)+"_ori.png", temp_ori[0].permute(1, 2, 0).detach().cpu().numpy()*255)
        # cv2.imwrite("./debug/"+str(view_id)+"_tgt.png", temp_tgt[0].permute(1, 2, 0).detach().cpu().numpy()*255)
        # cv2.imwrite("./debug/"+str(view_id)+"_mask.png", feature_masks[0].detach().cpu().numpy()*255)

        # print_num += 1

        temp_oris = (temp_oris - mean_img)[:, [2, 1, 0], :, :]
        temp_tgts = (temp_tgts - mean_img)[:, [2, 1, 0], :, :]

        features = self.model_DFF(torch.cat((temp_oris, temp_tgts), 0))
        
        feature_ori = torch.nn.functional.normalize(features[:N, :, :, :], dim=1)
        feature_tgt = torch.nn.functional.normalize(features[N:, :, :, :], dim=1)

        return feature_ori, feature_tgt, feature_mask, output_depth_tmp, output_color_tmp

    def arap_loss(self, P_Fill, Q_Fill, src_bool, neighbour_indexes):
        bs = Q_Fill.size()[0]
        deformation_neibour_points_ = self.indexing_neighbor(Q_Fill, neighbour_indexes)
        source_neibour_points_ = self.indexing_neighbor(P_Fill, neighbour_indexes)
        deformation_neibour_dis_ = deformation_neibour_points_ - Q_Fill.unsqueeze(2)
        source_neibour_dis_ = source_neibour_points_ - P_Fill.unsqueeze(2)

        deformation_neibour_dis_ = torch.sqrt(torch.mul(deformation_neibour_dis_, deformation_neibour_dis_).sum(dim =-1)+0.00001)
        source_neibour_dis_ = torch.sqrt(torch.mul(source_neibour_dis_, source_neibour_dis_).sum(dim =-1)+0.00001)
        difference = torch.mul((deformation_neibour_dis_ - source_neibour_dis_), src_bool.float()[:, :, None].repeat(1, 1, self.opt.neighbour_num))
        squ_difference = torch.sum(torch.mul(difference, difference)) / src_bool.sum()
        return squ_difference

    def normalization(self, p, mask):
        center = (p.mul(mask[:, :, None].float())).sum(1, keepdim=True) / mask.sum(1, keepdim=True)[:, :, None]
        p_new = p - center
        N = p.shape[0]
        max_ = (p_new.mul(mask[:, :, None].float())).abs().view(N, -1).max(1, keepdim=True)[0]
        p_new = p_new / max_[:, :, None]    
        return p_new, center, max_

    def normalization_with_know_center_scale(self, p, mask, center, scale):
        p_new = p - center
        p_new = p_new / scale[:, :, None]
        return p_new

    def txt2obj(self, path):
        fin = open(path, "r")
        fout = open(path[:-4]+".obj", "w")
        for line in fin.readlines():
            fout.write("v "+line)
        fout.close()
        fin.close()

    def loss_on_lfd(self, deformation_p, orig_color, p1, pts, tgt_color, weights, orig_bool, tgt_bool, tgt_pts_bool, mean_img):
        thisbatchsize = deformation_p.size()[0]
        output_depth = 0
        output_color = 0
        output_DFF = 0
        all_rot_matrix = self.all_rot_matrix_for_lfd.to(p1.device)

        deformation_p, deformation_p_center, dp_maxmin = self.normalization(deformation_p, orig_bool)
        deformation_p_views = torch.bmm(all_rot_matrix.view(1, -1, 3).expand(thisbatchsize, -1, -1), deformation_p.transpose(1,2)).transpose(1,2)

        p1, p1_center, p1_maxmin = self.normalization(p1, tgt_bool)
        p1_views = torch.bmm(all_rot_matrix.view(1, -1, 3).expand(thisbatchsize, -1, -1), p1.transpose(1,2) ).transpose(1,2)
        
        pts = self.normalization_with_know_center_scale(pts, tgt_pts_bool, p1_center, p1_maxmin)
        pts_views = torch.bmm(all_rot_matrix.view(1, -1, 3).expand(thisbatchsize, -1, -1), pts.transpose(1,2) ).transpose(1,2)

        for view_id in range(self.view_divide**2):
            proj_ori_vertex = deformation_p_views[:,:,3*view_id:3*view_id+3]
            proj_ori_vertex_x = (proj_ori_vertex[..., :1] + 1.) * (256-1) / 2 # (B, v_n, 1)
            proj_ori_vertex_y = (proj_ori_vertex[..., 1:2] + 1.) * (256-1) / 2 # (B, v_n, 1)
            proj_ori_vertex = torch.cat((torch.cat((proj_ori_vertex_x, proj_ori_vertex_y), -1), proj_ori_vertex[..., 2:]-1), -1) #(B, v_n, 3)

            proj_tar_vertex = p1_views[:,:,3*view_id:3*view_id+3]
            proj_tar_pts = pts_views[:,:,3*view_id:3*view_id+3]

            proj_tar_vertex_x = (proj_tar_vertex[..., :1] + 1.) * (256-1) / 2 # (B, v_n, 1)
            proj_tar_vertex_y = (proj_tar_vertex[..., 1:2] + 1.) * (256-1) / 2 # (B, v_n, 1)

            proj_tar_pts_x = (proj_tar_pts[..., :1] + 1.) * (256-1) / 2 # (B, v_n, 1)
            proj_tar_pts_y = (proj_tar_pts[..., 1:2] + 1.) * (256-1) / 2 # (B, v_n, 1)

            proj_tar_vertex = torch.cat((torch.cat((proj_tar_vertex_x, proj_tar_vertex_y), -1), proj_tar_vertex[..., 2:]-1), -1) #(B, v_n, 3)  
            weights_orig = torch.ones(proj_ori_vertex[:, :, 0:1].shape, device=proj_ori_vertex.device)
            ori_depth_img, ori_color_img, _, ori_weight_img, _ = self.point_renderer_(proj_ori_vertex.contiguous(), orig_color*255.0, weights_orig, orig_bool, self.threshold_)
            ori_d_img = ori_depth_img / ori_weight_img#(B, h, w, 1)
            ori_c_img = ori_color_img / ori_weight_img / 255.0#(B, h, w, 3)

            # cv2.imwrite("./test.png", ori_c_img[0].detach().cpu().numpy()[:, :, ::-1]*255)
            # exit(0)
            tgt_depth_img, tgt_color_img, Imweights_img, tgt_weight_img, _ = self.point_renderer_(proj_tar_vertex.contiguous(), tgt_color*255.0, weights, tgt_bool, self.threshold_)
            tgt_d_img = tgt_depth_img / tgt_weight_img
            tgt_c_img = tgt_color_img / tgt_weight_img / 255.0
            imweights_img = Imweights_img / tgt_weight_img

            # img1 = tgt_c_img[0].detach().cpu().numpy()*255
            # for i in range(proj_tar_pts_x.shape[1]):
            #     cv2.circle(img1, (int(proj_tar_pts_x[0, i, 0].item()), int(proj_tar_pts_y[0, i, 0].item())), 1, (0, 0, 255))
            # cv2.imwrite("./debug/"+str(view_id)+"_land.png", img1)

    ################# If to use the intersection mask #########################
            mask = torch.sign(ori_d_img*tgt_d_img).detach()
            feature_ori, feature_tgt, feature_mask, output_depth_tmp, output_color_tmp = self.ComputeDFF(view_id, ori_c_img, tgt_c_img, ori_d_img, tgt_d_img, proj_tar_pts_x, proj_tar_pts_y, tgt_pts_bool, mask.abs(), mean_img)
            output_depth += output_depth_tmp/(self.view_divide**2)
            output_color += output_color_tmp/(self.view_divide**2)
            output_DFF += torch.sum((torch.mul(feature_mask, 1 - torch.mul(feature_ori, feature_tgt).sum(1))).abs())/(self.view_divide**2)/feature_mask.sum()
        # exit(0)
        # return (10*output_depth + output_color + output_DFF)/thisbatchsize
        # return (output_depth + output_color + output_DFF)
        return (output_depth + output_color + self.opt.lambda_dff * output_DFF)

    def multiview_loss(self, Q_Fill, src_Fill_im, src_bool, weights_bw_Fill, tarPs_Fill, tarPts_Fill, tar_Fill_im, tar_bool, tar_pts_bool, mean_img):
        loss_multiview = self.loss_on_lfd(Q_Fill, src_Fill_im, tarPs_Fill, tarPts_Fill, tar_Fill_im, weights_bw_Fill, src_bool, tar_bool, tar_pts_bool, mean_img)
        return loss_multiview

    def forward(self, flow_lists, flow_bw_lists, src_input, tar_input, src_mask, fw_outrange_mask, bw_outrange_mask, tar_mask):
        N = src_input.shape[0]
        bw_outrange_mask = bw_outrange_mask.view(-1, 1, self.opt.height, self.opt.width)
        fw_outrange_mask = fw_outrange_mask.view(-1, 1, self.opt.height, self.opt.width)

        xx = torch.arange(self.opt.width, device=src_input.device).view(1,-1).repeat(self.opt.height,1)
        yy = torch.arange(self.opt.height, device=src_input.device).view(-1,1).repeat(1,self.opt.width)
        xx = xx.view(1,1,self.opt.height,self.opt.width).repeat(N,1,1,1)
        yy = yy.view(1,1,self.opt.height,self.opt.width).repeat(N,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        scale_value = torch.ones((1, 2, 1, 1), device=src_input.device)
        scale_value[:, 0, :, :] *= (2.0 / max(self.opt.width-1,1))
        scale_value[:, 1, :, :] *= (2.0 / max(self.opt.height-1,1))

        loss_ph = 0.0
        loss_smooth = 0.0
        n_predictions = len(flow_lists)
        for i in range(n_predictions):
            i_weight = 0.8**(n_predictions - i - 1)
            corres = grid + flow_lists[i] 
            vgrid = (corres.mul(scale_value) - 1.0).permute(0,2,3,1)
            new_tar_im = nn.functional.grid_sample(tar_input[:, :3, :, :], vgrid, padding_mode='border')

            corres_bw = grid + flow_bw_lists[i] 
            vgrid_bw = (corres_bw.mul(scale_value) - 1.0).permute(0,2,3,1)
            new_src_im = nn.functional.grid_sample(src_input[:, :3, :, :], vgrid_bw, padding_mode='border')

            loss_ph_fw = self.ph_loss(src_input[:, :3, :, :], new_tar_im, src_mask, fw_outrange_mask)
            loss_ph_bw = self.ph_loss(tar_input[:, :3, :, :], new_src_im, tar_mask, bw_outrange_mask)
            loss_ph += i_weight * (loss_ph_fw + loss_ph_bw) / 2.0

            loss_smooth_fw = self.smooth_loss(flow_lists[i], src_input[:, :3, :, :])
            loss_smooth_bw = self.smooth_loss(flow_bw_lists[i], tar_input[:, :3, :, :])
            loss_smooth += i_weight * (loss_smooth_fw + loss_smooth_bw) / 2.0

        return loss_ph, loss_smooth
