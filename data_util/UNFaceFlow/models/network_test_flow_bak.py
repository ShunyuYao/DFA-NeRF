import torch
import torch.nn as nn
import torch.nn.functional as F
from raft import RAFT
import cv2
import numpy as np
from nnutils import make_conv_2d, make_upscale_2d, make_downscale_2d, ResBlock2d, Identity
# class NeuralNRT(nn.Module):
#     def __init__(self, opt, path=None, device="cuda:0"):
#         super(NeuralNRT, self).__init__()
#         self.opt = opt
#         self.CorresPred = RAFT(opt)

#     def forward(self, src_im,tar_im):
#         N=src_im.shape[0]
#         src_im = src_im*255.0
#         tar_im = tar_im*255.0
#         flow_fw, _ = self.CorresPred(src_im, tar_im, iters=self.opt.iters)
#         return flow_fw

class ImportanceWeights(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()

        if opt.small:
            in_dim = 128
        else:
            in_dim = 256
        fn_0 = 16
        self.input_fn = fn_0 + 3 * 2
        fn_1 = 16
        self.conv1 = torch.nn.Conv2d(in_channels=in_dim, out_channels=fn_0, kernel_size=3, stride=1, padding=1)

        if opt.use_batch_norm:
            custom_batch_norm = torch.nn.BatchNorm2d
        else:
            custom_batch_norm = Identity

        self.model = nn.Sequential(
            make_conv_2d(self.input_fn, fn_1, n_blocks=1, normalization=custom_batch_norm),
            ResBlock2d(fn_1, normalization=custom_batch_norm),
            ResBlock2d(fn_1, normalization=custom_batch_norm),
            ResBlock2d(fn_1, normalization=custom_batch_norm),
            nn.Conv2d(fn_1, 1, kernel_size=3, padding=1)
            # torch.nn.Sigmoid()
        )

    def forward(self, x, features):
        # Reduce number of channels and upscale to highest resolution
        features = self.conv1(features)
        x = torch.cat([features, x], 1)
        assert x.shape[1] == self.input_fn
        x = self.model(x)
        print(x)
        print(x.max(), x.min(), x.mean())

        return torch.nn.Sigmoid()(x)

class NeuralNRT(nn.Module):
    def __init__(self, opt, path=None, device="cuda:0"):
        super(NeuralNRT, self).__init__()
        self.opt = opt
        self.CorresPred = RAFT(opt)
        self.ImportanceW = ImportanceWeights(opt)
        if path is not None:
            data = torch.load(path,map_location='cpu')
            if 'state_dict' in data.keys():
                self.CorresPred.load_state_dict(data['state_dict'])
                print("load done")
            else:
                self.CorresPred.load_state_dict({k.replace('module.', ''):v for k,v in data.items()})
                print("load done")
    def forward(self, src_im,tar_im, src_im_raw, tar_im_raw, Crop_param):
        N=src_im.shape[0]
        src_im = src_im*255.0
        tar_im = tar_im*255.0
        # cv2.imwrite("src_im.png", src_im[0].permute(1, 2, 0).cpu().numpy())
        # cv2.imwrite("tar_im.png", tar_im[0].permute(1, 2, 0).cpu().numpy())
        # print(src_im.shape, src_im_raw.shape)
        flow_fw_crop, feature_fw_crop = self.CorresPred(src_im, tar_im, iters=self.opt.iters)
        # print(flow_fw_crop.mean())
        # warp = np.zeros((320, 240, 3))
        # for i in range(320):
        #     for j in range(240):
        #         if int(i+flow_fw_crop[0, 1, i, j])>=0 and int(i+flow_fw_crop[0, 1, i, j])<320 and int(j+flow_fw_crop[0, 0, i, j])>=0 and int(j+flow_fw_crop[0, 0, i, j])<240:
        #             warp[int(i+flow_fw_crop[0, 1, i, j]), int(j+flow_fw_crop[0, 0, i, j]), :] = src_im[0, :, i, j].cpu().numpy()
        # cv2.imwrite("warp_im.png", warp)

        new_size = (8 * feature_fw_crop.shape[2], 8 * feature_fw_crop.shape[3])
        feature_fw_crop = F.interpolate(feature_fw_crop, size=new_size, mode='bilinear', align_corners=True)

        flow_fw = torch.zeros((N, 2, self.opt.height, self.opt.width), device=src_im.device)
        feature_fw = torch.zeros((N, feature_fw_crop.shape[1], self.opt.height, self.opt.width), device=src_im.device)

        xx = torch.arange(0, self.opt.width).view(1,-1).repeat(self.opt.height,1)
        yy = torch.arange(0, self.opt.height).view(-1,1).repeat(1,self.opt.width)
        xx = xx.view(1,1,self.opt.height,self.opt.width).repeat(N,1,1,1)
        yy = yy.view(1,1,self.opt.height,self.opt.width).repeat(N,1,1,1)
        grid = torch.cat((xx,yy),1).float()
        grid = grid.to(src_im.device)

        grid_crop = grid[:, :, :self.opt.crop_height, :self.opt.crop_width]
        scale_crop_value = torch.ones((1, 2, 1, 1), device=src_im.device)
        scale_crop_value[:, 0, :, :] *= (2.0 / max(self.opt.crop_width-1,1))
        scale_crop_value[:, 1, :, :] *= (2.0 / max(self.opt.crop_height-1,1))

        corres_crop = grid_crop + flow_fw_crop[-1]
        vgrid_crop = (corres_crop.mul(scale_crop_value) - 1.0).permute(0,2,3,1)
        new_tar_im_crop = nn.functional.grid_sample(tar_im, vgrid_crop, mode='bilinear', padding_mode='border')
        cat_im_crop = torch.cat((src_im, new_tar_im_crop), 1)
        weights_fw_crop = self.ImportanceW(cat_im_crop, feature_fw_crop)

        weights = torch.zeros((N, weights_fw_crop.shape[1], self.opt.height, self.opt.width), device=grid.device)

        leftup1 = torch.cat((Crop_param[:, 0:1, 0], Crop_param[:, 2:3, 0]), 1)[:, :, None, None]
        leftup2 = torch.cat((Crop_param[:, 4:5, 0], Crop_param[:, 6:7, 0]), 1)[:, :, None, None]

        scale1 = torch.cat(((Crop_param[:, 1:2, 0]-Crop_param[:, 0:1, 0]).float() / self.opt.crop_width, (Crop_param[:, 3:4, 0]-Crop_param[:, 2:3, 0]).float() / self.opt.crop_height), 1)[:, :, None, None]
        scale2 = torch.cat(((Crop_param[:, 5:6, 0]-Crop_param[:, 4:5, 0]).float() / self.opt.crop_width, (Crop_param[:, 7:8, 0]-Crop_param[:, 6:7, 0]).float() / self.opt.crop_height), 1)[:, :, None, None]
 
        flow_fw_crop = (scale2 - scale1) * grid_crop + scale2 * flow_fw_crop
        # print(feature_fw_crop)
        for i in range(N):
            flow_fw_cropi = F.interpolate(flow_fw_crop[i:(i+1)], ((Crop_param[i, 3, 0]-Crop_param[i, 2, 0]).item(), (Crop_param[i, 1, 0]-Crop_param[i, 0, 0]).item()), mode='bilinear', align_corners=True)
            flow_fw_cropi  =flow_fw_cropi + (leftup2 - leftup1)[i:(i+1), :, :, :]
            
            weights_fw_cropi = F.interpolate(weights_fw_crop[i:(i+1)], ((Crop_param[i, 3, 0]-Crop_param[i, 2, 0]).item(), (Crop_param[i, 1, 0]-Crop_param[i, 0, 0]).item()), mode='bilinear')
            weights[i, :, Crop_param[i, 2, 0]:Crop_param[i, 3, 0], Crop_param[i, 0, 0]:Crop_param[i, 1, 0]] = weights_fw_cropi[0]
            flow_fw[i, :, Crop_param[i, 2, 0]:Crop_param[i, 3, 0], Crop_param[i, 0, 0]:Crop_param[i, 1, 0]] = flow_fw_cropi[0]
        return flow_fw, weights