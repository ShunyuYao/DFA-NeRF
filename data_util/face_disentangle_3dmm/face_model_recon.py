import torch
import torch.nn as nn
import numpy as np
from scipy.io import loadmat
import os

# BFM 3D face model
class BFM():
	def __init__(self, model_path = './BFM/BFM_model_front.mat'):
		model = loadmat(model_path)
		self.meanshape = model['meanshape'] # mean face shape. [3*N,1]
		self.idBase = model['idBase'] # identity basis. [3*N,80]
		self.exBase = model['exBase'].astype(np.float32) # expression basis. [3*N,64]
		self.meantex = model['meantex'] # mean face texture. [3*N,1] (0-255)
		self.texBase = model['texBase'] # texture basis. [3*N,80]
		self.point_buf = model['point_buf'].astype(np.int32) # face indices for each vertex that lies in. starts from 1. [N,8]
		self.face_buf = model['tri'].astype(np.int32) # vertex indices for each face. starts from 1. [F,3]
		self.front_mask_render = model['frontmask2_idx'].squeeze().astype(np.int32) # vertex indices for small face region to compute photometric error. starts from 1.
		self.mask_face_buf = model['tri_mask2'].squeeze().astype(np.int32) # vertex indices for each face from small face region. starts from 1. [f,3]
		self.skin_mask = model['skinmask'].squeeze().astype(np.int32) # vertex indices for pre-defined skin region to compute reflectance loss
		self.keypoints = model['keypoints'].squeeze().astype(np.int32)  # vertex indices for 68 landmarks. starts from 1. [68,1]

class Face_3DMM_Recon(nn.Module):
    def __init__(self, modelpath, point_num):
        super(Face_3DMM_Recon, self).__init__()
        # id_dim = 100
        # exp_dim = 79
        # tex_dim = 100
        self.point_num = point_num
        facemodel = BFM(modelpath)
        self.facemodel = facemodel
        base_id = self.facemodel.idBase
        base_exp = self.facemodel.exBase
        mu = self.facemodel.meanshape
        self.base_id = torch.as_tensor(base_id).cuda()
        self.base_exp = torch.as_tensor(base_exp).cuda()
        self.mu = torch.as_tensor(mu).cuda()

        base_tex = self.facemodel.texBase
        mu_tex = self.facemodel.meantex
        self.base_tex = torch.as_tensor(base_tex).cuda()
        self.mu_tex = torch.as_tensor(mu_tex).cuda()

    def forward_geo_sub(self, id_para, exp_para, sub_index):
        
        id_tmp = torch.einsum('ij,aj->ai', self.base_id, id_para)
        exp_tmp = torch.einsum('ij,aj->ai', self.base_exp, exp_para)
        # print("id_tmp: ", id_tmp)
        geometry = id_tmp + \
            exp_tmp + self.mu
        
        sel_index = sub_index
        sel_index = sel_index - 1
        # reshape face shape to [batchsize,N,3]
        bs = geometry.shape[0]
        geometry = geometry.reshape(bs, -1, 3)
        # re-centering the face shape with mean shape
        geometry = geometry - self.mu.view(-1, 3).mean(dim=0).view(1, 1, 3)
        geometry = geometry[:, sel_index, :]
        return geometry

    def forward_geo(self, id_para, exp_para):

        id_tmp = torch.einsum('ij,aj->ai', self.base_id, id_para)
        exp_tmp = torch.einsum('ij,aj->ai', self.base_exp, exp_para)
        geometry = id_tmp + \
            exp_tmp + self.mu
        bs = geometry.shape[0]
        # reshape face shape to [batchsize,N,3]
        geometry = geometry.reshape(bs, -1, 3)
        # re-centering the face shape with mean shape
        geometry = geometry - self.mu.view(-1, 3).mean(dim=0).view(1, 1, 3)
        return geometry

    def forward_tex(self, tex_para):
        tex_para = tex_para*self.sig_tex
        texture = torch.mm(tex_para, self.base_tex) + self.mu_tex
        return texture.reshape(-1, self.point_num, 3)
