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

torch.pi = torch.acos(torch.zeros(1)).item() * 2
print_num = 0

def check_Apart1(Jacobi,Vindex,A,Vnum,Pnum):
    rows=(3*torch.arange(Pnum, device=Vindex.device).view(-1,1).repeat(1,Jacobi.shape[0]//Pnum).view(-1,1)+torch.arange(3, device=Vindex.device).view(1,-1)).view(-1,3,1).repeat(1,1,6).reshape(-1)
    cols=(6*Vindex.view(-1,1)+torch.arange(6,device=Vindex.device).view(1,-1)).view(-1,1,6).repeat(1,3,1).reshape(-1)
    J=torch.sparse.FloatTensor(torch.cat([rows.view(1,-1),cols.view(1,-1)],dim=0), Jacobi.view(-1), torch.Size([Pnum*3,Vnum*6])).to_dense()
    gtA=J.t()@J
    check=(gtA-A).abs()
    mask=check>1.e-5
    num=mask.sum()
    if num>0:
        # check=check[mask]
        # nonzero=mask.nonzero()        
        # print(Vnum,(nonzero[:,0]-nonzero[:,1]).abs().max())
        print('A is not gt: %d, %f, %f'%(num.item(),check.max().item(),check.mean().item()))
        return False
    return True

def check_Aregu(A,Jacobi1,Jacobi2,Graph_Edge,Vnum):
    rows=(3*torch.arange(Jacobi1.shape[0], device=Graph_Edge.device).view(-1,1)+torch.arange(3, device=Graph_Edge.device)).view(-1,3,1).repeat(1,1,6).reshape(-1)
    rows=torch.cat([rows,rows],dim=0).view(1,-1)
    cols1=(6*Graph_Edge[0].view(-1,1)+torch.arange(6,device=Graph_Edge.device)).view(-1,1,6).repeat(1,3,1).reshape(-1)
    cols2=(6*Graph_Edge[1].view(-1,1)+torch.arange(6,device=Graph_Edge.device)).view(-1,1,6).repeat(1,3,1).reshape(-1)
    cols=torch.cat([cols1,cols2],dim=0).view(1,-1)
    # print(Graph_Edge.shape,rows.shape,cols.shape)
    values=torch.cat([Jacobi1.view(-1),Jacobi2.view(-1)],dim=0)
    J=torch.sparse.FloatTensor(torch.cat([rows,cols],dim=0), values.view(-1), torch.Size([Graph_Edge.shape[1]*3,Vnum*6])).to_dense()
    gtA=J.t()@J
    check=(gtA-A).abs()
    mask=check>1.e-5
    num=mask.sum()
    if num>0:
        # check=check[mask]
        print('A regu is not gt: %d, %f, %f'%(num.item(),check.max().item(),check.mean().item()))
        return False
    return True


def check_symmetric(A):
    check=(A-A.t()).abs()
    num=(check>1.e-6).sum()
    if num>0:
        print('A is not symmetric: %d, %f, %f'%(num.item(),check.max().item(),check.mean().item()))
        return False
    return True
# def transpose_LUdata(A_lu):
#     A_lu=A_lu.detach().clone().transpose(0,1)
#     mask=torch.ones(A_lu.shape,device=A_lu.device,dtype=torch.long)
#     d=A_lu.diag()
#     sm=mask.tril(-1)
#     A_lu[sm]=(A_lu/d.view(1,-1))[sm]
#     sm=mask.triu(1)
#     A_lu[sm]=(A_lu*d.view(-1,1))[sm]
    
#     return A_lu

# def construct_P(pivots):
#     pivots_zero_idx = pivots - 1
#     final_order = [i for i in range(pivots_zero_idx.shape[0])]
#     for k, j, in enumerate(pivots_zero_idx):
#         final_order[k], final_order[j] = final_order[j], final_order[k]
#     P=torch.eye(pivots.shape[0], device=pivots.device, dtype=torch.long).index_select(1,torch.as_tensor(final_order, device=pivots.device))
#     return P

# the backward is correct only for symmetric A, if A is not symmetric, grad_b should use A transpose to  solve
# it seems that WholeSolve is faster than BatchSolve
class WholeSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, b):
        A_lu,pivots,check=torch.lu(A,get_infos=True)
        if check.abs().item()>1.e-5: #non-zero, fail:
            print('A matrix is singular1')
            A_lu,pivots,check=torch.lu(A+0.5*torch.eye(A.shape[0],dtype=A.dtype,device=A.device),get_infos=True)
            if check.abs().item()>1.e-5:
                print('A matrix is singular2')
                x=torch.zeros((b.shape[0], 1),device=b.device)
                A_lu.zero_()
                ctx.save_for_backward(A_lu,pivots,x)
                return x
        x=torch.lu_solve(b.view(-1,1),A_lu,pivots)
        # if (torch.isnan(x)).any().item() or (torch.isinf(x)).any().item():
        #     print('solution is nan')
        #     x=torch.zeros((b.shape[0], 1),device=b.device)
        #     A_lu.zero_()
        #     ctx.save_for_backward(A_lu,pivots,x)
        #     return x
        ctx.save_for_backward(A_lu,pivots,x)
        return x
    @staticmethod
    def backward(ctx, grad):
        A_lu,pivots,x=ctx.saved_tensors
        grad_b=torch.lu_solve(grad,A_lu,pivots)
        # if (torch.isnan(grad_b)).any().item() or (torch.isinf(grad_b)).any().item():
        #     print('backward is nan')
        #     grad_b=torch.zeros(grad.shape,device=grad.device)
        grad_A=-grad_b.view(-1,1).matmul(x.view(1,-1))
        return grad_A, grad_b.view(-1)
# the backward is correct only for symmetric A, if A is not symmetric, grad_b should use A transpose to  solve
class BatchSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, b, Graph_extend_batch):
        x=[]
        A_lu=[]
        for bid in range(Graph_extend_batch[-1].item()+1):
            mask=Graph_extend_batch==bid
            Mask=mask[:,None]&mask[None,:]
            Ab=A[Mask].reshape(-1,mask.sum().item())           
            bb=b[mask]
            Ab_lu,Ab_pivots,check=torch.lu(Ab,get_infos=True)
            if check.abs().item()>1.e-5:
                Ab_lu,Ab_pivots,check=torch.lu(Ab+0.5*torch.eye(Ab.shape[0],dtype=Ab.dtype,device=Ab.device),get_infos=True)
                if check.abs().item()>1.e-5:
                    xb=torch.zeros_like(bb).view(-1,1)
                    Ab_lu.zero_()
                    x.append(xb)
                    A_lu.extend([Ab_lu,Ab_pivots])
                    continue            
            xb=torch.lu_solve(bb.view(-1,1),Ab_lu,Ab_pivots)            
            x.append(xb)
            A_lu.extend([Ab_lu,Ab_pivots])        
        ctx.save_for_backward(*A_lu,*x,Graph_extend_batch)
        x=torch.cat(x,dim=0)
        return x
    @staticmethod  
    def backward(ctx, grad):
        Graph_extend_batch=ctx.saved_tensors[-1]
        batch_num=Graph_extend_batch[-1].item()+1
        grad_A=torch.zeros(grad.shape[0],grad.shape[0],dtype=grad.dtype,device=grad.device)
        grad_b=torch.zeros(grad.shape[0],dtype=grad.dtype,device=grad.device)
        for bid in range(batch_num):
            Ab_lu=ctx.saved_tensors[2*bid:2*bid+2]
            xb=ctx.saved_tensors[2*batch_num+bid]
            mask=Graph_extend_batch==bid
            Mask=mask[:,None]&mask[None,:]
            grad_bb=torch.lu_solve(grad[mask],*Ab_lu)
            grad_Ab=-grad_bb.matmul(xb.view(1,-1))
            grad_A[Mask]=grad_Ab.view(-1)
            grad_b[mask]=grad_bb.view(-1)
        return grad_A, grad_b, None

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0, 0.01)
        if m.bias is not None:
            m.bias.data = torch.ones(m.bias.data.size(), device=m.weight.device)

def show(x, mask, Graph_nodes_ids, v_ids, nodes_mask, edges_ids, edges_mask):
    color = x[0, :3, :, :].permute(1, 2, 0).cpu().numpy()
    test_img = color.copy()
    pos = list((mask[0, 0].cpu().numpy()).nonzero())
    for i in range(nodes_mask.sum().item()):
        cv2.circle(test_img, (pos[1][Graph_nodes_ids[0, i]], pos[0][Graph_nodes_ids[0, i]]), 2 , (0,0,255))
    ex = np.random.randint(0, mask.sum().item(), (10, 1))
    # for i in range(10):
    #     start = (pos[1][ex[i, 0]], pos[0][ex[i, 0]])
    #     cv2.circle(test_img, start, 2 , (0,0,255))
    #     for j in range(6):
    #         r = np.random.randint(256)
    #         g = np.random.randint(256)
    #         b = np.random.randint(256)
    #         end = (pos[1][v_ids[ex[i, 0], j]], pos[0][v_ids[ex[i, 0], j]])
    #         cv2.line(test_img, start, end, (r, g, b), 2)
    start = (pos[1][edges_ids[i, 0]], pos[0][edges_ids[i, 0]])
    for i in range(edges_mask.sum().item()):
        start1 = (pos[1][edges_ids[i, 0]], pos[0][edges_ids[i, 0]])
        if start1 != start:
            continue
        end = (pos[1][edges_ids[i, 1]], pos[0][edges_ids[i, 1]])
        cv2.line(test_img, start, end, (255, 0, 0), 2)
    cv2.imwrite("./test.png", test_img)

def write_mesh(path, P, mask):
    fout=open(path, "w")
    # print(path)
    for i in range(480*640):
        if mask[0, i]:
            fout.write("v " + str(P[0, i, 0].item()) + " " + str(P[0, i, 1].item()) + " " + str(P[0, i, 2].item()) + "\n")
    fout.close()
    return

class DFF(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers1 = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU()
        )
        self.layers2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU()
        )
        self.layers3 = nn.Sequential(
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU()
        )
        self.layers4 = nn.Sequential(
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU()
        )
        self.layers5 = nn.Sequential(
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.BatchNorm2d(512, affine=False),
            nn.ReLU()
        )
        self.layers6 = nn.Sequential(
            nn.Conv2d(512, 1024, 3, 1, 1),
            nn.BatchNorm2d(1024, affine=False),
            nn.ReLU()
        )
        self.layers7 = nn.Sequential(
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.BatchNorm2d(512, affine=False),
            nn.ReLU()
        )

        #decode
        self.layers8 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU()
        )
        self.layers9 = nn.Sequential(
            nn.ConvTranspose2d(512, 128, 4, 2, 1),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU()
        )
        self.layers10 = nn.Sequential(
            nn.ConvTranspose2d(256, 64, 4, 2, 1),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU()
        )
        self.layers11 = nn.Sequential(
            nn.ConvTranspose2d(128, 32, 4, 2, 1),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU()
        )
        self.layers12 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32, affine=False),
            nn.ReLU()
        )


    def forward(self, x):
        x1 = self.layers1(x)
        x2 = self.layers2(x1)
        x3 = self.layers3(x2)
        x4 = self.layers4(x3)
        x5 = self.layers5(x4)
        x6 = self.layers6(x5)
        x7 = self.layers7(x6)

        x8 = self.layers8(x7)
        x_cat1 = torch.cat((x4, x8), 1)
        x9 = self.layers9(x_cat1)
        x_cat2 = torch.cat((x3, x9), 1)
        x10 = self.layers10(x_cat2)
        x_cat3 = torch.cat((x2, x10), 1)
        x11 = self.layers11(x_cat3)
        x_cat4 = torch.cat((x1, x11), 1)
        x12 = self.layers12(x_cat4)

        return x12

class BlendSkinWNet(torch.nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        
        fdim = self.opt.fdim
        if opt.use_batch_norm:
            custom_batch_norm = torch.nn.BatchNorm2d
        else:
            custom_batch_norm = Identity

        self.FeatureExtract = nn.Sequential(
            make_conv_2d(6, fdim//4, n_blocks=1, normalization=custom_batch_norm),
            ResBlock2d(fdim//4, normalization=custom_batch_norm),
            ResBlock2d(fdim//4, normalization=custom_batch_norm),
            ResBlock2d(fdim//4, normalization=custom_batch_norm),
            nn.Conv2d(fdim//4, fdim, kernel_size=3, padding=1),
        )
        self.RegressNet = nn.Sequential(
            nn.Linear(2*self.opt.fdim, self.opt.fdim),
            nn.BatchNorm1d(self.opt.fdim),
            nn.ReLU(),
            nn.Linear(self.opt.fdim, self.opt.fdim//2),
            nn.BatchNorm1d(self.opt.fdim//2),
            nn.ReLU(),
            nn.Linear(self.opt.fdim//2, 1)
        )

    def forward(self, x, mask, v_ids, Graph_nodes_ids, nodes_mask, Graph_Edge, edges_mask, points):
        #x:(B, 6, H, W)
        #feature: (B, fdim, H, W)
        #Graph_nodes_ids:(B, Nv)
        #v_ids: (B, num_adja, H, W)
        B = mask.shape[0]
        # mask_copy = mask.clone()
        mask = mask.view(B, -1)
        Nvs = torch.sum(mask, 1)
        # print(Nvs)
        mask = mask.view(-1)
        non_zero= torch.nonzero(mask, as_tuple=False)
        #将v_ids的值映射到整个点云
        v_ids = v_ids.permute(0, 2, 3, 1).reshape(-1)
        ids=torch.arange(B, device=v_ids.device).view(-1,1).repeat(1,self.opt.num_adja*self.opt.height*self.opt.width).view(-1)
        v_ids = Graph_nodes_ids[ids, v_ids].view(B, -1)
        # print(torch.cumsum(Nvs,dim=0))
        temp=torch.cumsum(Nvs,dim=0)[:-1]
        v_ids[1:, :]+=temp.reshape(-1, 1)
        v_ids = v_ids.view(-1, self.opt.num_adja)
        
        v_ids = v_ids[mask]
        
        # ids_edges = Graph_Edge.view(-1)
        # ids1 = torch.arange(B).view(-1,1).repeat(1,2*Graph_Edge.shape[2]).view(-1).to(v_ids.device)
        # edges_ids = Graph_nodes_ids[ids1, ids_edges].view(B, 2, -1)
        # edges_ids[1:, :, :] += temp.reshape(-1, 1, 1)
        # edges_ids = edges_ids.permute(0, 2, 1).view(-1, 2)[edges_mask.view(-1)]
        # show(x, mask_copy, Graph_nodes_ids, v_ids, nodes_mask, edges_ids, edges_mask)
        # exit(0)

        v_ids = v_ids.view(-1)
        v_alpha = torch.zeros(B*self.opt.height*self.opt.width, self.opt.num_adja, device=v_ids.device)
        # print("bsw: ", bsw)
        #test: fixed weights 
        points = x[:, 3:, :, :]
        points = points.permute(0, 2, 3, 1).reshape(-1, 3)[mask]
        X = points[:, None, :].repeat(1, self.opt.num_adja, 1).view(-1, 3)
        Y = points[v_ids]
        dist = -((X - Y)**2).sum(1) / 0.075 / 0.075 / 2.0
        dist = dist.view(-1, self.opt.num_adja)
        W = torch.softmax(dist, 1)
        v_alpha[mask] = W

        return v_alpha.view(B, self.opt.height, self.opt.width, self.opt.num_adja).permute(0, 3, 1, 2)

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
            nn.Conv2d(fn_1, 1, kernel_size=3, padding=1),
            torch.nn.Sigmoid()
        )

    def forward(self, x, features):
        # Reduce number of channels and upscale to highest resolution
        features = self.conv1(features)
        x = torch.cat([features, x], 1)
        assert x.shape[1] == self.input_fn
        return self.model(x)

class NeuralNRT(nn.Module):
    def __init__(self, opt, path=None, device="cuda:0"):
        super(NeuralNRT, self).__init__()
        self.opt = opt
        self.CorresPred = RAFT_ALL(opt)
        self.ImportanceW = ImportanceWeights(opt)
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
            self.ImportanceW.apply(weights_init)

        self.Nc = self.opt.num_corres
        self.lambda_2d = self.opt.lambda_2d
        self.lambda_depth = self.opt.lambda_depth
        self.lambda_reg = self.opt.lambda_reg
        self.num_adja = self.opt.num_adja
        
        #(N*2*H*W)
        # mesh grid 
        self.opt.width = self.opt.width
        self.opt.height = self.opt.height

    def deformation(self, R, T, src_input, V, object_mask, Vindex, Valpha):
        N = src_input.shape[0]
        src_points = src_input[:, 3:, :, :]
        object_mask=object_mask.view(-1)
        
        Vindex = Vindex[object_mask]
        Valpha = Valpha[object_mask]
        alpha=Valpha/Valpha.sum(1,keepdim=True)
        alpha=alpha.view(-1)
        src_points=src_points.view(N,3,-1).permute(0,2,1).reshape(-1,3)
        P=src_points[object_mask]
        corre_ids=torch.arange(P.shape[0],device=P.device)[:,None].repeat(1,Vindex.shape[-1]).reshape(-1)
        Vindex=Vindex.view(-1)
        PV=P[corre_ids]-V[Vindex]
        Q=(torch.bmm(R[Vindex],PV[:,:,None]).squeeze(-1)+V[Vindex]+T[Vindex])*alpha[:,None]
        Q=Q.view(-1,self.opt.num_adja,3).sum(1)
        return Q
    
    def convert_dense_batch(self, src_input, tar_input, Q, object_mask, tar_pts_mask, tar_mask, weights_bw, src_neigb_id):
        N = src_input.shape[0]
        src_im = src_input[:, :3, :, :]
        src_points = src_input[:, 3:, :, :]
        tar_im = tar_input[:, :3, :, :]
        tar_points = tar_input[:, 3:, :, :]

        object_mask=object_mask.view(-1)
        tar_mask=tar_mask.view(-1)
        tar_pts_mask=tar_pts_mask.view(-1)

        src_neigb_id = src_neigb_id.view(N,  self.opt.neighbour_num, -1).permute(0, 2, 1).reshape(-1,  self.opt.neighbour_num)
        src_neigb_id = src_neigb_id[object_mask]#(-1, num_neigb)
        weights_bw = weights_bw.view(-1, 1)[tar_mask]
        tarPs=tar_points.view(N, 3, -1).permute(0,2,1).reshape(-1,3)[tar_mask]
        tarPts=tar_points.view(N, 3, -1).permute(0,2,1).reshape(-1,3)[tar_mask&tar_pts_mask]
        src_im = src_im.view(N, 3, -1).permute(0,2,1).reshape(-1,3)[object_mask]
        tar_im = tar_im.view(N, 3, -1).permute(0,2,1).reshape(-1,3)[tar_mask]

        src_batch=torch.arange(N,device=src_im.device).view(-1,1).repeat(1,self.opt.width*self.opt.height).reshape(-1)
        src_batch=src_batch[object_mask]
        src_Fill_im,src_bool=to_dense_batch(src_im,src_batch) #(B * max_num_nodes, 3), (B * max_num_nodes)
        Q_Fill, _ = to_dense_batch(Q,src_batch)
        P=(src_points.view(N,3,-1).permute(0,2,1).reshape(-1,3))[object_mask]
        P_Fill, _ = to_dense_batch(P, src_batch)
        neigb_id_Fill, _ = to_dense_batch(src_neigb_id, src_batch)

        tar_batch=torch.arange(N,device=tar_im.device).view(-1,1).repeat(1,self.opt.width*self.opt.height).reshape(-1)
        tar_batch=tar_batch[tar_mask]
        tar_Fill_im,tar_bool=to_dense_batch(tar_im,tar_batch) #(B * max_num_nodes, 3), (B * max_num_nodes)
        tarPs_Fill, _=to_dense_batch(tarPs,tar_batch)
        weights_bw_Fill, _ = to_dense_batch(weights_bw, tar_batch)

        tar_pts_batch=torch.arange(N,device=tar_im.device).view(-1,1).repeat(1,self.opt.width*self.opt.height).reshape(-1)
        tar_pts_batch=tar_pts_batch[tar_mask&tar_pts_mask]
        tarPts_Fill, tar_pts_bool=to_dense_batch(tarPts,tar_pts_batch)

        Q_Fill = Q_Fill.view(N, -1, 3)
        P_Fill = P_Fill.view(N, -1, 3)
        weights_bw_Fill = weights_bw_Fill.view(N, -1, 1)
        src_Fill_im = src_Fill_im.view(N, -1, 3)
        tarPs_Fill = tarPs_Fill.view(N, -1, 3)
        tarPts_Fill = tarPts_Fill.view(N, -1, 3)
        tar_Fill_im = tar_Fill_im.view(N, -1, 3)
        src_bool = src_bool.view(N, -1)
        tar_bool = tar_bool.view(N, -1)
        tar_pts_bool = tar_pts_bool.view(N, -1)
        
        # global num
        # write_mesh("deform_"+str(num)+".obj", Q_Fill, src_bool)
        # num += 1

        return P_Fill, Q_Fill, src_Fill_im, src_bool, weights_bw_Fill, neigb_id_Fill, tarPs_Fill, tarPts_Fill, tar_Fill_im, tar_bool, tar_pts_bool

    def updateRT(self, Trans, R, T):
        axis_mat_1 = torch.zeros(R.shape, device=R.device)
        axis_mat_1[:, 0, 1] = -1
        axis_mat_1[:, 1, 0] = 1
        axis_mat_2 = torch.zeros(R.shape, device=R.device)
        axis_mat_2[:, 0, 2] = 1
        axis_mat_2[:, 2, 0] = -1
        axis_mat_3 = torch.zeros(R.shape, device=R.device)
        axis_mat_3[:, 1, 2] = -1
        axis_mat_3[:, 2, 1] = 1

        eye_mat = (torch.eye(3, device=R.device)[None, :, :].repeat(R.shape[0], 1, 1))
        Trans = Trans.clone()
        tmp = (Trans[:, :3]**2).sum(dim=1)
        Trans[tmp==0, 0] = 0.01
        theta = Trans[:, :3].norm(dim=1)
        axis = torch.nn.functional.normalize(Trans[:, :3], dim=1)
        cos_theta = torch.cos(theta[:, None, None])
        sin_theta = torch.sin(theta[:, None, None])        
        skew_sym_mat = axis.view(-1,3,1,1)[:,2,:,:].mul(axis_mat_1) + axis.view(-1,3,1,1)[:,1,:,:].mul(axis_mat_2) + axis.view(-1,3,1,1)[:,0,:,:].mul(axis_mat_3)
        outer_dot = torch.matmul(axis[:, :, None], axis[:, None, :])
        delta_R = cos_theta.mul(eye_mat) + (1 - cos_theta).mul(outer_dot) + sin_theta.mul(skew_sym_mat)
        return delta_R.matmul(R), T+Trans[:, 3:]
    #here target_d is C.shape[0]
    def ConstructAB(self, R, T, V,Graph_batch,Graph_Edge, W, C, P, Target_d, Vindex, Valpha, Corres_batch, sigma, fx, fy, ox, oy):
        N = Corres_batch[-1].item()+1
        C_num = P.shape[0]        
        Vindex=Vindex.view(-1)
        alpha=Valpha/Valpha.sum(1,keepdim=True)
        alpha=alpha.view(-1)
        PV=P[:,None,:]-V[Vindex].reshape(C_num,self.num_adja,3)
        PV=PV.view(-1,3,1)
        RPV=R[Vindex].matmul(PV).view(-1,3)
        RPV_mat=torch.zeros(RPV.shape[0],3,3,device=RPV.device)
        RPV_mat[:,0,1]=-RPV[:,2]
        RPV_mat[:,0,2]=RPV[:,1]
        RPV_mat[:,1,0]=RPV[:,2]
        RPV_mat[:,1,2]=-RPV[:,0]
        RPV_mat[:,2,0]=-RPV[:,1]
        RPV_mat[:,2,1]=RPV[:,0]
        Q=(RPV+V[Vindex]+T[Vindex])*alpha.view(-1,1)
        Q=Q.view(-1,self.num_adja,3).sum(1,keepdim=True).view(C_num,3)
        # print("Q:", Q[:, 2].max(), Q[:, 2].min(), Valpha.sum(1,keepdim=True).min())
        tar_fx=fx.view(-1)[Corres_batch]
        tar_fy=fy.view(-1)[Corres_batch]
        ox=ox.view(-1)[Corres_batch]
        oy=oy.view(-1)[Corres_batch]
        W=W.view(-1)
        x=tar_fx*Q[:,0]/(Q[:,2]+1.e-5)+ox
        y=tar_fy*Q[:,1]/(Q[:,2]+1.e-5)+oy
        Residual_2d=math.sqrt(self.lambda_2d)*W[:,None]*(torch.cat([x[:,None],y[:,None]],dim=-1)-C)

        intri_mat=torch.zeros(Q.shape[0],2,3,device=Q.device)
        intri_mat[:,0,0]=tar_fx/(Q[:,2]+1.e-5)
        intri_mat[:,0,2]=-tar_fx*Q[:,0]/((Q[:,2]+1.e-5)**2)
        intri_mat[:,1,1]=tar_fy/(Q[:,2]+1.e-5)
        intri_mat[:,1,2]=-tar_fy*Q[:,1]/((Q[:,2]+1.e-5)**2)

        intri_mat=intri_mat[:,None,:,:].expand(C_num,self.num_adja,2,3)
        # C_num, num_adj, 2, 6
        Jacobi_2d=math.sqrt(self.lambda_2d)*torch.cat([-intri_mat@RPV_mat.view(C_num,self.num_adja,3,3),intri_mat],dim=-1)*W.view(C_num,-1,1,1)*alpha.view(C_num,-1,1,1)
        Residual_depth=math.sqrt(self.lambda_depth)*W*(Q[:,2]-Target_d)
        intri_mat=torch.zeros(Q.shape[0],self.num_adja,3,device=Q.device)
        intri_mat[:,:,-1]=1
        Jacobi_depth=math.sqrt(self.lambda_depth)*torch.cat([-RPV_mat.view(C_num,self.num_adja,3,3)[:,:,-1,:],intri_mat],dim=-1)*W.view(C_num,-1,1)*alpha.view(C_num,-1,1)
        #C_num, num_adj, 3, 6
        Jacobi=torch.cat([Jacobi_2d,Jacobi_depth[:,:,None,:]],dim=2)
        #C_num, 3
        Residual=torch.cat([Residual_2d,Residual_depth[:,None]],dim=-1)
        b=(-Jacobi*Residual.view(C_num,1,3,1)).sum(-2).reshape(-1,6)
        b=scatter(b,Vindex,dim=0,dim_size=V.shape[0]).view(-1)
        Jacobi=Jacobi.view(-1,3,6)
        rows=torch.arange(C_num*self.num_adja, device=P.device).view(-1,1).repeat(1,self.num_adja).view(-1)
        cols=torch.arange(C_num*self.num_adja, device=P.device).view(-1,1,self.num_adja).repeat(1,self.num_adja,1).reshape(-1)
        A=Jacobi[rows].permute(0,2,1)@Jacobi[cols]
        indices=Vindex[rows]*V.shape[0]+Vindex[cols]
        # construct method 1    
        A=scatter(A,indices,dim=0,dim_size=V.shape[0]**2)
        A=A.view(V.shape[0],V.shape[0],6,6).permute(0,1,3,2).reshape(V.shape[0],6*V.shape[0],6).permute(0,2,1).reshape(6*V.shape[0],6*V.shape[0])        
        A_regu,b_regu=self.ConstructABRegu(R, T, V,Graph_batch,Graph_Edge)
        return A+A_regu,b+b_regu


    def ConstructABRegu(self,R, T, V, Graph_batch, Graph_Edge):
        Vi=V[Graph_Edge[0]]
        Vj=V[Graph_Edge[1]]
        Ti=T[Graph_Edge[0]]
        Tj=T[Graph_Edge[1]]
        Ri=R[Graph_Edge[0]]
        RiVij=Ri.matmul((Vj-Vi)[:,:,None]).squeeze(-1)
        Residual=RiVij+Vi+Ti-Vj-Tj

        Jacobi1=torch.cat([torch.zeros((Vi.shape[0],3,3), device=V.device),torch.eye(3, device=V.device)[None,:,:].repeat(Vi.shape[0],1,1)],dim=-1)
        Jacobi1[:,0,1]=RiVij[:,2]
        Jacobi1[:,0,2]=-RiVij[:,1]
        Jacobi1[:,1,0]=-RiVij[:,2]
        Jacobi1[:,1,2]=RiVij[:,0]
        Jacobi1[:,2,0]=RiVij[:,1]
        Jacobi1[:,2,1]=-RiVij[:,0]
        Jacobi2=torch.cat([torch.zeros((Vi.shape[0],3,3), device=V.device),-torch.eye(3, device=V.device)[None,:,:].repeat(Vi.shape[0],1,1)],dim=-1)

        b=scatter(torch.bmm(-Residual[:,None,:],Jacobi1).squeeze(1), Graph_Edge[0], dim=0, dim_size=V.shape[0])
        b=b+scatter(torch.bmm(-Residual[:,None,:],Jacobi2).squeeze(1), Graph_Edge[1], dim=0, dim_size=V.shape[0])
        A=torch.zeros(V.shape[0],V.shape[0],6,6,device=V.device)        
        indices=Graph_Edge.min(0)[0]*V.shape[0]+Graph_Edge.max(0)[0]
        temp_indices=torch.arange(indices.shape[0],device=indices.device)
        permute_indices=scatter(temp_indices,indices,dim=0,dim_size=V.shape[0]**2)
        permute_indices=permute_indices[indices]-temp_indices
        temp=torch.bmm(Jacobi1.permute(0,2,1),Jacobi2)
        A[Graph_Edge[0],Graph_Edge[1],:]=temp+temp.permute(0,2,1)[permute_indices]
        temp=Jacobi1.permute(0,2,1)@Jacobi1
        temp[:,[3,4,5],[3,4,5]]+=torch.ones(Jacobi1.shape[0],3,device=V.device)
        temp=scatter(temp,Graph_Edge[0],dim=0,dim_size=V.shape[0])
        temp_indices=torch.arange(V.shape[0],device=V.device)
        A[temp_indices,temp_indices]=temp

        A=A.permute(0,1,3,2).reshape(V.shape[0],6*V.shape[0],6).permute(0,2,1).reshape(6*V.shape[0],6*V.shape[0])
        return self.lambda_reg*A,self.lambda_reg*b.view(-1)

    def convert_graph_rep(self,src_points,src_mask,Graph_Edge,edges_mask,Graph_nodes_ids,nodes_mask,src_Vindex):
        #nodes_mask, Graph_nodes_ids:(B, Nv)
        B = src_mask.shape[0]
        Nvs=nodes_mask.sum(1)
        temp=torch.cumsum(Nvs,dim=0)[:-1]

        Nps = src_mask.view(B, -1).sum(1)
        temp_p = torch.cumsum(Nps,dim=0)[:-1]
        Graph_nodes_ids[1:,:]+=temp_p.reshape(-1,1)
        nodes_mask=nodes_mask.reshape(-1)
        Graph_nodes_ids=Graph_nodes_ids.reshape(-1)[nodes_mask]
        Graph_V = (src_points.permute(0,2,3,1).reshape(-1, 3))[src_mask.view(-1)][Graph_nodes_ids]
        
        Graph_Edge[1:,:,:]+=temp.reshape(-1,1,1)
        edges_mask=edges_mask.reshape(-1)
        Graph_Edge=Graph_Edge.permute(0,2,1).reshape(-1,2)[edges_mask]
        Graph_Edge=Graph_Edge.transpose(0,1)
        Graph_batch=[nv.item()*[ind] for ind,nv in enumerate(Nvs)]
        Graph_batch=[v for vv in Graph_batch for v in vv]
        Graph_batch=torch.from_numpy(np.array(Graph_batch,dtype=np.int64)).to(src_points.device)

        src_Vindex[1:,:,:,:]+=temp.view(-1,1,1,1)

        return Graph_batch,Graph_V,Graph_Edge,src_Vindex

    def construct_corres(self,src_input,tar_input,src_crop_im,tar_crop_im,src_mask,tar_mask,src_parsing_mask,tar_parsing_mask,src_Vindex, src_Valpha, src_pts, tar_pts, \
        src_recon_pts, tar_recon_pts, Crop_param):

        N=src_input.shape[0]
        src_points = src_input[:, 3:, :, :]
        tar_points = tar_input[:, 3:, :, :]
        # write_mesh("./debug/src.obj", src_points.view(-1, 3, self.opt.height*self.opt.width).permute(0, 2, 1), src_mask.view(N, -1))
        # write_mesh("./debug/tar.obj", tar_points.view(-1, 3, self.opt.height*self.opt.width).permute(0, 2, 1), tar_mask.view(N, -1))
        # cv2.imwrite("./debug/crop_src.png", src_crop_im.permute(0, 2, 3, 1)[0].cpu().numpy())
        # cv2.imwrite("./debug/crop_tar.png", tar_crop_im.permute(0, 2, 3, 1)[0].cpu().numpy())
        # cv2.imwrite("./debug/raw_src.png", src_input[:, :3, :, :].permute(0, 2, 3, 1)[0].cpu().numpy()*255)
        # cv2.imwrite("./debug/raw_tar.png", tar_input[:, :3, :, :].permute(0, 2, 3, 1)[0].cpu().numpy()*255)

        flow_fw_crop, feature_fw_crop = self.CorresPred(src_crop_im, tar_crop_im, iters=self.opt.iters)
        flow_bw_crop, feature_bw_crop = self.CorresPred(tar_crop_im, src_crop_im, iters=self.opt.iters)

        new_size = (8 * feature_fw_crop.shape[2], 8 * feature_fw_crop.shape[3])
        feature_fw_crop = F.interpolate(feature_fw_crop, size=new_size, mode='bilinear')
        feature_bw_crop = F.interpolate(feature_bw_crop, size=new_size, mode='bilinear')

        xx = torch.arange(self.opt.width, device=src_input.device).view(1,-1).repeat(self.opt.height,1)
        yy = torch.arange(self.opt.height, device=src_input.device).view(-1,1).repeat(1,self.opt.width)
        xx = xx.view(1,1,self.opt.height,self.opt.width).repeat(N,1,1,1)
        yy = yy.view(1,1,self.opt.height,self.opt.width).repeat(N,1,1,1)
        grid = torch.cat((xx,yy),1).float()

        scale_value = torch.ones((1, 2, 1, 1), device=src_input.device)
        scale_value[:, 0, :, :] *= (2.0 / max(self.opt.width-1,1))
        scale_value[:, 1, :, :] *= (2.0 / max(self.opt.height-1,1))

        grid_crop = grid[:, :, :self.opt.crop_height, :self.opt.crop_width]
        scale_crop_value = torch.ones((1, 2, 1, 1), device=src_input.device)
        scale_crop_value[:, 0, :, :] *= (2.0 / max(self.opt.crop_width-1,1))
        scale_crop_value[:, 1, :, :] *= (2.0 / max(self.opt.crop_height-1,1))

        corres_crop = grid_crop + flow_fw_crop[-1]
        vgrid_crop = (corres_crop.mul(scale_crop_value) - 1.0).permute(0,2,3,1)
        new_tar_im_crop = nn.functional.grid_sample(tar_crop_im, vgrid_crop, mode='bilinear', padding_mode='border')
        cat_im_crop = torch.cat((src_crop_im, new_tar_im_crop), 1)
        weights_fw_crop = self.ImportanceW(cat_im_crop, feature_fw_crop)

        corres_bw_crop = grid_crop + flow_bw_crop[-1]
        vgrid_bw_crop = (corres_bw_crop.mul(scale_crop_value) - 1.0).permute(0,2,3,1)
        new_src_im_crop = nn.functional.grid_sample(src_crop_im, vgrid_bw_crop, mode='bilinear', padding_mode='border')
        cat_im_crop_bw = torch.cat((tar_crop_im, new_src_im_crop), 1)
        weights_bw_crop = self.ImportanceW(cat_im_crop_bw, feature_bw_crop)

        weights = torch.zeros((N, weights_fw_crop.shape[1], self.opt.height, self.opt.width), device=grid.device)
        weights_bw = torch.zeros((N, weights_bw_crop.shape[1], self.opt.height, self.opt.width), device=grid.device)

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
                if j==((len(flow_bw_crop))-1):
                    weights_fw_cropi = F.interpolate(weights_fw_crop[i:(i+1)], ((Crop_param[i, 3, 0]-Crop_param[i, 2, 0]).item(), (Crop_param[i, 1, 0]-Crop_param[i, 0, 0]).item()), mode='bilinear')
                    weights_bw_cropi = F.interpolate(weights_bw_crop[i:(i+1)], ((Crop_param[i, 7, 0]-Crop_param[i, 6, 0]).item(), (Crop_param[i, 5, 0]-Crop_param[i, 4, 0]).item()), mode='bilinear')
                    weights[i, :, Crop_param[i, 2, 0]:Crop_param[i, 3, 0], Crop_param[i, 0, 0]:Crop_param[i, 1, 0]] = weights_fw_cropi[0]
                    weights_bw[i, :, Crop_param[i, 6, 0]:Crop_param[i, 7, 0], Crop_param[i, 4, 0]:Crop_param[i, 5, 0]] = weights_bw_cropi[0]


            flow_fw_lists.append(flow_fwj)
            flow_bw_lists.append(flow_bwj)

        flow_fw = flow_fw_lists[-1]
        flow_bw = flow_bw_lists[-1]
        corres = grid + flow_fw
        corres_bw = grid + flow_bw
        src_mask = src_mask.view(N, -1)
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

        src_points = src_points.view(N, 3, -1)
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
        M_Corres_valid_candidate = ~fw_outrange_mask

        M_Corres_valid = torch.full(M_Corres_valid_candidate.shape, False, device=M_Corres_valid_candidate.device, dtype=torch.bool)

        # valid_pts_mask = ((src_pts[:, :, 0]>=0) & (src_pts[:, :, 0]<self.opt.height) & (src_pts[:, :, 1]>=0) & (src_pts[:, :, 1]<self.opt.width) & \
        #     (tar_pts[:, :, 0]>=0) & (tar_pts[:, :, 0]<self.opt.height) & (tar_pts[:, :, 1]>=0) & (tar_pts[:, :, 1]<self.opt.width)).view(-1)

        # valid_recon_pts_mask = ((src_recon_pts[:, :, 0]>=0) & (src_recon_pts[:, :, 0]<self.opt.height) & (src_recon_pts[:, :, 1]>=0) & (src_recon_pts[:, :, 1]<self.opt.width) & \
        #     (tar_recon_pts[:, :, 0]>=0) & (tar_recon_pts[:, :, 0]<self.opt.height) & (tar_recon_pts[:, :, 1]>=0) & (tar_recon_pts[:, :, 1]<self.opt.width) & \
        #         (src_recon_pts[:, :, 2]==1) & (tar_recon_pts[:, :, 2]==1)).view(-1)

        # Bs = torch.arange(N, device=src_input.device).view(-1, 1).repeat(1, self.opt.num_pts).view(-1)
        # idsi = src_pts[:, :, 0].view(-1)
        # idsj = src_pts[:, :, 1].view(-1)
        # idsi[~valid_pts_mask] = 0
        # idsj[~valid_pts_mask] = 0
        # valid_pts_mask = valid_pts_mask & (src_mask.view(N, self.opt.height, self.opt.width)[(Bs, idsi, idsj)])

        # Bs = Bs[valid_pts_mask]
        # src_pts_pos = (src_pts[:, :, 0] * self.opt.width + src_pts[:, :, 1]).view(-1)[valid_pts_mask]
        # zeros_ = torch.zeros((N*self.opt.num_pts), device=src_input.device, dtype=torch.long)[valid_pts_mask]
        # ones_ = torch.ones((N*self.opt.num_pts), device=src_input.device, dtype=torch.long)[valid_pts_mask]
        # M_Corres_valid[(Bs, src_pts_pos)] = True

        # cv2.imwrite("./debug/valid_corres_pts.png", M_Corres_valid.view(N, self.opt.height, self.opt.width)[0].float().cpu().numpy()*255)


        # Bs_recon = torch.arange(N, device=src_input.device).view(-1, 1).repeat(1, self.opt.num_recon_pts).view(-1)
        # idsi_recon = src_recon_pts[:, :, 0].view(-1)
        # idsj_recon = src_recon_pts[:, :, 1].view(-1)
        # idsi_recon[~valid_recon_pts_mask] = 0
        # idsj_recon[~valid_recon_pts_mask] = 0
        
        # valid_recon_pts_mask = valid_recon_pts_mask & (src_mask.view(N, self.opt.height, self.opt.width)[(Bs_recon, idsi_recon, idsj_recon)])
        # Bs_recon = Bs_recon[valid_recon_pts_mask]
        # src_recon_pts_pos = (src_recon_pts[:, :, 0] * self.opt.width + src_recon_pts[:, :, 1]).view(-1)[valid_recon_pts_mask]
        # zeros_recon = torch.zeros((N*self.opt.num_recon_pts), device=src_input.device, dtype=torch.long)[valid_recon_pts_mask]
        # ones_recon = torch.ones((N*self.opt.num_recon_pts), device=src_input.device, dtype=torch.long)[valid_recon_pts_mask]
        # M_Corres_valid[(Bs_recon, src_recon_pts_pos)] = True

        # cv2.imwrite("./debug/valid_corres_recon_pts.png", M_Corres_valid.view(N, self.opt.height, self.opt.width)[0].float().cpu().numpy()*255)


        corres_clone = corres.permute(0,2,1).clone()
        # print(corres_clone.max(), corres_clone.min(), tar_pts.max(), tar_pts.min())
        # corres_clone[(Bs, src_pts_pos, zeros_)] = tar_pts[:, :, 1].view(-1)[valid_pts_mask].float()
        # corres_clone[(Bs, src_pts_pos, ones_)] = tar_pts[:, :, 0].view(-1)[valid_pts_mask].float()
        # corres_clone[(Bs_recon, src_recon_pts_pos, zeros_recon)] = tar_recon_pts[:, :, 1].view(-1)[valid_recon_pts_mask].float()
        # corres_clone[(Bs_recon, src_recon_pts_pos, ones_recon)] = tar_recon_pts[:, :, 0].view(-1)[valid_recon_pts_mask].float()
        # cv2.imwrite("mask1.png", M_Corres_valid[0].view(self.opt.height, self.opt.width).cpu().numpy()*255)


        # combine_img_pts = torch.cat((src_input[0, :3, :, :], tar_input[0, :3, :, :]), 2).permute(1, 2, 0).cpu().numpy()
        # print(combine_img_pts.shape, combine_img_pts.dtype)
        # for i in range(Bs.shape[0]):
        #     start = (src_pts_pos[i].item()%self.opt.width, src_pts_pos[i].item()//self.opt.width)
        #     end = (corres_clone[0, src_pts_pos[i], 0].int().item()+self.opt.width, corres_clone[0, src_pts_pos[i], 1].int().item())
        #     r = np.random.randn(1)[0]
        #     g = np.random.randn(1)[0]
        #     b = np.random.randn(1)[0]
        #     combine_img_pts = cv2.line(np.ascontiguousarray(combine_img_pts), start, end, (r, g, b), 2)
        # cv2.imwrite("./debug/combine_pts.png", combine_img_pts*255)
        
        # combine_img_recon_pts = torch.cat((src_input[0, :3, :, :], tar_input[0, :3, :, :]), 2).permute(1, 2, 0).cpu().numpy()
        # for i in range(Bs_recon.shape[0]):
        #     a = np.random.randint(0, 20)
        #     if a>=2:
        #         continue
        #     start = (src_recon_pts_pos[i].item()%self.opt.width, src_recon_pts_pos[i].item()//self.opt.width)
        #     end = (corres_clone[0, src_recon_pts_pos[i], 0].int().item()+self.opt.width, corres_clone[0, src_recon_pts_pos[i], 1].int().item())
        #     r = np.random.randn(1)[0]
        #     g = np.random.randn(1)[0]
        #     b = np.random.randn(1)[0]
        #     combine_img_recon_pts = cv2.line(np.ascontiguousarray(combine_img_recon_pts), start, end, (r, g, b), 2)
        # cv2.imwrite("./debug/combine_recon_pts.png", combine_img_recon_pts*255)


        # debug
        M_Corres_valid_candidate = M_Corres_valid_candidate & (~M_Corres_valid)
        if (M_Corres_valid_candidate.sum(1)==0).any():
            print('here1:', end='')
        
        M_Corres_valid_rand=torch.rand(M_Corres_valid_candidate.shape,device=M_Corres_valid_candidate.device)
        M_Corres_valid_rand=M_Corres_valid_rand<float(self.Nc)/(M_Corres_valid_candidate.sum(1,keepdim=True)+0.01).to(torch.float)
        M_Corres_valid_rand=M_Corres_valid_rand&M_Corres_valid_candidate
        M_Corres_valid[M_Corres_valid_rand] = True

        # cv2.imwrite("./debug/valid_corres.png", M_Corres_valid.view(N, self.opt.height, self.opt.width)[0].float().cpu().numpy()*255)


        if (M_Corres_valid.sum().item()%N != 0):
            num_rest = N - (M_Corres_valid.sum().item()) % N
            valid_rest = (~M_Corres_valid) & M_Corres_valid_candidate
            if (valid_rest.sum().item()<num_rest):
                M_Corres_valid[M_Corres_valid] = False
            else:
                non_zero = valid_rest.nonzero(as_tuple=False)
                valid_rest[non_zero[num_rest:, 0], non_zero[num_rest:, 1]] = False
                M_Corres_valid = M_Corres_valid | valid_rest

        assert(M_Corres_valid.sum().item()%N == 0)

        is_not_enough = False
        if (M_Corres_valid.sum(1)==0).any():
            print('here2:', end='')
            print("candidate: ", M_Corres_valid_candidate.sum(1))
            is_not_enough = True
            return is_not_enough, None, None, None, None, None, None, None, flow_fw_lists, flow_bw_lists, fw_outrange_mask, bw_outrange_mask, weights, weights_bw


        # test_corres = torch.zeros(corres_clone.shape, device=corres_clone.device).long()
        # test_corres[(Bs_recon, src_recon_pts_pos, zeros_recon)] = tar_recon_pts[:, :, 1].view(-1)[valid_recon_pts_mask]
        # test_corres[(Bs_recon, src_recon_pts_pos, ones_recon)] = tar_recon_pts[:, :, 0].view(-1)[valid_recon_pts_mask]
        # test_corres = test_corres.view((N, self.opt.height, self.opt.width, 2))
        # test_mask = np.zeros((self.opt.height, self.opt.width), dtype=np.uint8)
        # test_mask_tar = np.zeros((self.opt.height, self.opt.width), dtype=np.uint8)

        # src_color =  np.ascontiguousarray(src_im[0].permute(1, 2, 0).cpu().numpy())
        # tar_color =  np.ascontiguousarray(tar_im[0].permute(1, 2, 0).cpu().numpy())
        # for i in range(self.opt.height):
        #     for j in range(self.opt.width):
        #         if test_corres[0, i, j, 0] != 0:
        #             src_color = cv2.circle(src_color, (j, i), 1, (0, 0, 255))
        #             tar_color = cv2.circle(tar_color, (test_corres[0, i, j, 0].item(), test_corres[0, i, j, 1].item()), 1, (0, 0, 255))

        # cv2.imwrite("src.png", src_color[:, :, ::-1])
        # cv2.imwrite("tar.png", tar_color[:, :, ::-1])
        # exit(0)

        weights_clone = weights.view(N, -1).clone()
        # weights_clone[(Bs, src_pts_pos)] = 5.0
        # weights_clone[(Bs_recon, src_recon_pts_pos)] = 2.0
        # cv2.imwrite("./debug/test_weights.png", weights_clone[0].view(self.opt.height, self.opt.width).detach().cpu().numpy()*40)
        # cv2.imwrite("mask.png", M_Corres_valid[0].view(self.opt.height, self.opt.width).cpu().numpy()*255)
        # exit(0)

        M_Corres_valid=M_Corres_valid.view(-1)
        C=corres_clone.reshape(-1,2)[M_Corres_valid]
        W=weights_clone.view(-1)[M_Corres_valid]
        P=src_points.view(N,3,-1).permute(0,2,1).reshape(-1,3)[M_Corres_valid]

        Vindex=src_Vindex.view(N,self.num_adja,-1).permute(0,2,1).reshape(-1,self.num_adja)[M_Corres_valid]
        Valpha=src_Valpha.view(N,self.num_adja,-1).permute(0,2,1).reshape(-1,self.num_adja)[M_Corres_valid]
        Corres_batch=torch.arange(N,device=C.device).view(-1,1).repeat(1,self.opt.width*self.opt.height).reshape(-1)
        Corres_batch=Corres_batch[M_Corres_valid]
        C_Fill,back_bool=to_dense_batch(C,Corres_batch)
        vgrid1 = ((C_Fill.view(N, -1, 1, 2)).mul(scale_value.view(1,1,1,2)) - 1.0)
        Target_d = nn.functional.grid_sample(tar_points[:,2:,:,:], vgrid1, padding_mode='border').view(N,-1)
        Target_d=Target_d[back_bool].reshape(-1)
        return is_not_enough, W, C, P, Target_d, Vindex, Valpha, Corres_batch, flow_fw_lists, flow_bw_lists, fw_outrange_mask, bw_outrange_mask, weights, weights_bw

    def forward(self, src_input, tar_input, src_crop_im, tar_crop_im, Graph_Edge, edges_mask, Graph_nodes_ids, nodes_mask, Camera, Crop_param, src_Vindex, src_Valpha, src_mask, tar_mask, src_parsing_mask, tar_parsing_mask, tar_pts_mask, src_neigb_id, src_pts, tar_pts, src_recon_pts, tar_recon_pts, mean_img, sigma):
        src_points=src_input[:, 3:, :, :]
        src_Vindex=src_Vindex[:,:self.num_adja,:,:]
        src_Valpha=src_Valpha[:,:self.num_adja,:,:]
        fx = Camera[:, 0, 0]
        fy = Camera[:, 1, 0]
        ox = Camera[:, 2, 0]
        oy = Camera[:, 3, 0]

        # cv2.imwrite("./src.png", src_input[0, :3, :, :].permute(1, 2, 0).cpu().numpy()[:, :, ::-1]*255)
        # cv2.imwrite("./tar.png", tar_input[0, :3, :, :].permute(1, 2, 0).cpu().numpy()[:, :, ::-1]*255)
        # cv2.imwrite("./src_p.png", src_parsing_mask[0, 0, :, :].cpu().numpy()*10)
        # cv2.imwrite("./tar_p.png", tar_parsing_mask[0, 0, :, :].cpu().numpy()*10)
        # exit(0)


        Graph_batch,Graph_V,Graph_Edge,src_Vindex=self.convert_graph_rep(src_points,src_mask,Graph_Edge,edges_mask,Graph_nodes_ids,nodes_mask,src_Vindex)
        is_not_enough, W, C, P, Target_d, Vindex, Valpha, Corres_batch, flow_lists, flow_bw_lists, fw_outrange_mask, bw_outrange_mask, weights, weights_bw=\
            self.construct_corres(src_input,tar_input,src_crop_im,tar_crop_im,src_mask,tar_mask,src_parsing_mask,tar_parsing_mask,src_Vindex,src_Valpha, src_pts, tar_pts, src_recon_pts, tar_recon_pts, Crop_param)

        object_Vindex=src_Vindex.view(-1,self.num_adja,self.opt.height*self.opt.width).permute(0,2,1).reshape(-1,self.num_adja)
        object_Valpha=src_Valpha.view(-1,self.num_adja,self.opt.height*self.opt.width).permute(0,2,1).reshape(-1,self.num_adja)

        R_init = (torch.eye(3, device=Graph_V.device).unsqueeze(0).repeat(Graph_V.shape[0], 1, 1))
        T_init = torch.zeros((Graph_V.shape[0],3), device=Graph_V.device)
        R = R_init.clone()
        T = T_init.clone()

        is_skip = torch.zeros(1, device=R.device)
        if is_not_enough:
            is_skip[0] = 1
            return is_skip, weights, weights_bw, torch.tensor([0.0], device=is_skip.device), torch.tensor([0.0], device=is_skip.device), torch.tensor([0.0], device=is_skip.device), torch.tensor([0.0], device=is_skip.device), torch.tensor([0.0], device=is_skip.device), torch.tensor([0.0], device=is_skip.device)

        for ind in range(3):
            # print(ind, R.mean().item(), R.max().item(), R.min().item(), T.mean().item(), T.max().item(), T.min().item())
            A,b=self.ConstructAB(R, T, Graph_V,Graph_batch,Graph_Edge, W, C, P, Target_d, Vindex, Valpha, Corres_batch, sigma, fx, fy, ox, oy)
            Trans=WholeSolve.apply(A,b)
            if (torch.isnan(Trans)).any().item() or (torch.isinf(Trans)).any().item():
                print("solve error, skip!")
                is_skip[0] = 1
                break
            R, T = self.updateRT(Trans.view(-1,6), R, T)
            if ind==2 and (torch.abs(T_old-T)).max().item()>3:
                print("is not convergence")
                is_skip[0] = 1
                break
            T_old = T


        #compute loss
        Q = self.deformation(R, T, src_input, Graph_V, src_mask, object_Vindex, object_Valpha)

        P_Fill, Q_Fill, src_Fill_im, src_bool, weights_bw_Fill, neigb_id_Fill, tarPs_Fill, tarPts_Fill, tar_Fill_im, tar_bool, tar_pts_bool = \
            self.convert_dense_batch(src_input, tar_input, Q, src_mask, tar_pts_mask, tar_mask, weights_bw, src_neigb_id)

        # if print_num % 10 == 0:
        #     fout=open("./test_single/warp"+str(print_num)+".obj", "w")
        #     for i in range(Q_Fill.shape[1]):
        #         if src_bool[0, i]:
        #             fout.write("v " + str(Q_Fill[0, i, 0].item()) + " " + str(Q_Fill[0, i, 1].item()) + " " + str(Q_Fill[0, i, 2].item()) + "\n")
        #     fout.close()
        # exit(0)

        loss_ph, loss_weights, loss_mulview, loss_arap, loss_smooth, loss_parsing = self.LossClass(flow_lists, flow_bw_lists, src_input, tar_input, src_mask, weights, weights_bw, \
            fw_outrange_mask, bw_outrange_mask, tar_mask, Q_Fill, src_Fill_im, src_bool, weights_bw_Fill, tarPs_Fill, tarPts_Fill, tar_Fill_im, tar_bool, tar_pts_bool, P_Fill, \
                neigb_id_Fill, mean_img, tar_parsing_mask, src_parsing_mask)
        return is_skip, weights, weights_bw, loss_ph.unsqueeze(0), loss_weights.unsqueeze(0), loss_mulview.unsqueeze(0), loss_arap.unsqueeze(0), loss_smooth.unsqueeze(0), loss_parsing.unsqueeze(0)

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
        self.point_renderer_ = Render(256, 256, 3, 3, 1e-5)
        self.model_DFF = DFF()
        self.model_DFF.load_state_dict(torch.load(opt.pretrain_DFF_path))
        for param in self.model_DFF.parameters():
            param.requires_grad = False

        self.threshold_ = 0.1
        self.view_divide = self.opt.view_divide
        self.all_rot_matrix_for_lfd = torch.zeros([self.view_divide**2,3,3], device="cuda:0")
        for view_id in range(self.view_divide**2):
            euler_x = view_id//(self.view_divide)
            euler_y = (view_id - euler_x * (self.view_divide))
            euler_angle = torch.tensor([[-torch.pi / 6.0+torch.pi / 3.0 * euler_x/(self.view_divide-1), -torch.pi / 6.0+torch.pi / 3.0*euler_y/(self.view_divide-1), 0]], dtype=torch.float32, device="cuda:0")
            self.all_rot_matrix_for_lfd[view_id] = euler2rot(euler_angle)
        
        self.all_rot_matrix_for_lfd = self.all_rot_matrix_for_lfd.view(-1,3)

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

    def parsing_loss(self, src_parsing_mask, corres_parsing_mask, object_mask, out_range_mask=None):
        # mask = torch.mul(object_mask.float(), weights)
        if out_range_mask is not None:
            mask = (object_mask & (~out_range_mask)).float()
        else:
            mask = object_mask.float()
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
        mask = (bw_outrange_mask & tar_mask).float()
        # mask = (bw_outrange_mask).float()
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

    def forward(self, flow_lists, flow_bw_lists, src_input, tar_input, src_mask, weights, weights_bw, fw_outrange_mask, bw_outrange_mask, tar_mask, Q_Fill, src_Fill_im, src_bool, weights_bw_Fill, \
            tarPs_Fill, tarPts_Fill, tar_Fill_im, tar_bool, tar_pts_bool, P_Fill, neigb_id_Fill, mean_img, tar_parsing_mask, src_parsing_mask):
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
        loss_parsing = 0.0
        loss_smooth = 0.0
        n_predictions = len(flow_lists)
        for i in range(n_predictions):
            i_weight = 0.8**(n_predictions - i - 1)
            corres = grid + flow_lists[i] 
            vgrid = (corres.mul(scale_value) - 1.0).permute(0,2,3,1)
            new_tar_im = nn.functional.grid_sample(tar_input[:, :3, :, :], vgrid, padding_mode='border')
            new_tar_parsing_mask = nn.functional.grid_sample(tar_parsing_mask.float(), vgrid, mode='nearest', padding_mode='border')

            corres_bw = grid + flow_bw_lists[i] 
            vgrid_bw = (corres_bw.mul(scale_value) - 1.0).permute(0,2,3,1)
            new_src_im = nn.functional.grid_sample(src_input[:, :3, :, :], vgrid_bw, padding_mode='border')
            new_src_parsing_mask = nn.functional.grid_sample(src_parsing_mask.float(), vgrid_bw, mode='nearest', padding_mode='border')

            loss_ph_fw = self.ph_loss(src_input[:, :3, :, :], new_tar_im, src_mask)
            loss_ph_bw = self.ph_loss(tar_input[:, :3, :, :], new_src_im, tar_mask)
            loss_ph += i_weight * (loss_ph_fw + loss_ph_bw) / 2.0

            loss_parsing_fw = self.parsing_loss(src_parsing_mask, new_tar_parsing_mask, src_mask)
            loss_parsing_bw = self.parsing_loss(tar_parsing_mask, new_src_parsing_mask, tar_mask)
            loss_parsing += i_weight * (loss_parsing_fw + loss_parsing_bw) / 2.0

            loss_smooth_fw = self.smooth_loss(flow_lists[i], src_input[:, :3, :, :])
            loss_smooth_bw = self.smooth_loss(flow_bw_lists[i], tar_input[:, :3, :, :])
            loss_smooth += i_weight * (loss_smooth_fw + loss_smooth_bw) / 2.0

        loss_weights = self.weights_loss(weights_bw, bw_outrange_mask, tar_mask)
        loss_mulview = self.multiview_loss(Q_Fill, src_Fill_im, src_bool, weights_bw_Fill, tarPs_Fill, tarPts_Fill, tar_Fill_im, tar_bool, tar_pts_bool, mean_img)
        # loss_arap = self.arap_loss(P_Fill, Q_Fill, src_bool, neigb_id_Fill)
        loss_arap = torch.tensor(0.0, device=loss_mulview.device)

        return loss_ph, loss_weights, loss_mulview, loss_arap, loss_smooth, loss_parsing
