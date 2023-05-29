import torch
import torch.nn as nn
import numpy as np
import os


class Face_3DMM(nn.Module):
    def __init__(self, modelpath, id_dim, exp_dim, tex_dim, point_num):
        super(Face_3DMM, self).__init__()
        # id_dim = 100
        # exp_dim = 79
        # tex_dim = 100
        self.point_num = point_num
        base_id = np.fromfile(os.path.join(modelpath, 'sub_b_shape.bin'),
                              np.float32).reshape(-1, 3*self.point_num)[:id_dim, :]
        mu_id = np.fromfile(os.path.join(
            modelpath, 'sub_mushape.bin'), np.float32)
        base_exp = np.fromfile(os.path.join(modelpath, 'sub_b_exp.bin'),
                               np.float32).reshape(-1, 3*self.point_num)[:exp_dim, :]
        mu_exp = np.fromfile(os.path.join(
            modelpath, 'sub_muexp.bin'), np.float32)
        mu = mu_id + mu_exp
        mu = mu.reshape(-1, 3)
        for i in range(3):
            mu[:, i] -= np.mean(mu[:, i])
        mu = mu.reshape(-1)
        self.base_id = torch.as_tensor(base_id).cuda()/1000.0
        self.base_exp = torch.as_tensor(base_exp).cuda()/1000.0
        self.mu = torch.as_tensor(mu).cuda()/1000.0

        base_tex = np.fromfile(os.path.join(modelpath, 'sub_b_tex.bin'),
                               np.float32).reshape(-1, 3*self.point_num)[:tex_dim, :]
        mu_tex = np.fromfile(os.path.join(
            modelpath, 'sub_mutex.bin'), np.float32)
        self.base_tex = torch.as_tensor(base_tex).cuda()
        self.mu_tex = torch.as_tensor(mu_tex).cuda()

        sig_id = np.fromfile(os.path.join(
            modelpath, 'sig_shape.bin'), np.float32)[:id_dim]
        sig_tex = np.fromfile(os.path.join(
            modelpath, 'sig_tex.bin'), np.float32)[:tex_dim]
        sig_exp = np.fromfile(os.path.join(
            modelpath, 'sig_exp.bin'), np.float32)[:exp_dim]
        self.sig_id = torch.as_tensor(sig_id).cuda()
        self.sig_tex = torch.as_tensor(sig_tex).cuda()
        self.sig_exp = torch.as_tensor(sig_exp).cuda()

    def forward_geo_sub(self, id_para, exp_para, sub_index):
        id_para = id_para*self.sig_id
        exp_para = exp_para*self.sig_exp
        sel_index = torch.cat((3*sub_index.unsqueeze(1), 3*sub_index.unsqueeze(1)+1,
                               3*sub_index.unsqueeze(1)+2), dim=1).reshape(-1)
        geometry = torch.mm(id_para, self.base_id[:, sel_index]) + \
            torch.mm(exp_para, self.base_exp[:,
                                             sel_index]) + self.mu[sel_index]
        return geometry.reshape(-1, sub_index.shape[0], 3)

    def forward_geo(self, id_para, exp_para):
        id_para = id_para*self.sig_id
        exp_para = exp_para*self.sig_exp
        geometry = torch.mm(id_para, self.base_id) + \
            torch.mm(exp_para, self.base_exp) + self.mu
        return geometry.reshape(-1, self.point_num, 3)

    def forward_tex(self, tex_para):
        tex_para = tex_para*self.sig_tex
        texture = torch.mm(tex_para, self.base_tex) + self.mu_tex
        return texture.reshape(-1, self.point_num, 3)
