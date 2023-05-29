from torch._C import dtype
from load_audface import load_audface_data_split
import os
import numpy as np
import imageio
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from run_nerf_helpers import *
from decoder import Decoder


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_config_file(config_path):
    with open(config_path, "r") as f:
        config_str = f.readlines()
    near = float(config_str[3].split("=")[-1].strip())
    far = float(config_str[4].split("=")[-1].strip())

    return near, far


def encode_signal(dataset, itr_obj, img_i, dim_aud, AudNet, ExpNet, AudAttNet, global_step, args, len_auds,
            embed_fn = None):
    
    if itr_obj == 0:
        self_audios = dataset[itr_obj]['auds']
        self_exps = dataset[itr_obj]['exp']
         
        if global_step >= args.nosmo_iters:
            smo_half_win = int(args.smo_size / 2)
            left_i = img_i - smo_half_win
            right_i = img_i + smo_half_win
            pad_left, pad_right = 0, 0
            if left_i < 0:
                pad_left = -left_i
                left_i = 0
            if right_i > len_auds:
                pad_right = right_i-len_auds
                right_i = len_auds
            auds_win = self_audios[left_i:right_i]
            exps_win = self_exps[left_i:right_i]
            if pad_left > 0:
                auds_win = torch.cat(
                    (torch.zeros_like(auds_win)[:pad_left], auds_win), dim=0)
                exps_win = torch.cat(
                    (torch.zeros_like(exps_win)[:pad_left], exps_win), dim=0)
            if pad_right > 0:
                auds_win = torch.cat(
                    (auds_win, torch.zeros_like(auds_win)[:pad_right]), dim=0)
                exps_win = torch.cat(
                    (exps_win, torch.zeros_like(exps_win)[:pad_right]), dim=0)
            auds_win = AudNet(auds_win)
            exps_win = ExpNet(exps_win)
            auds_win = torch.cat([auds_win, exps_win], axis=1)
            aud = AudAttNet(auds_win).unsqueeze(0)

        else:
            self_audios = AudNet(self_audios[img_i:img_i+1])
            self_exps = ExpNet(self_exps[img_i:img_i+1])
            aud = torch.cat([self_audios, self_exps], axis=1)
        
        signal = [aud, None]

    else:
        # self_audios = torch.zeros(1, dim_aud)
        self_exp = dataset[itr_obj]['exp'][img_i:img_i+1]
        signal = [None, self_exp]

    return signal


def encode_signal_torso(dataset, itr_obj, img_i, PoseAttNet, global_step, args, len_poses,
             embed_fn = None):


    poses = dataset[itr_obj]['poses']
    

    if global_step >= args.nosmo_iters:
            smo_half_win = int(args.smo_torse_size / 2)
            left_i = img_i - smo_half_win
            right_i = img_i + smo_half_win
            pad_left, pad_right = 0, 0
            if left_i < 0:
                pad_left = -left_i
                left_i = 0
            if right_i > len_poses:
                pad_right = right_i-len_poses
                right_i = len_poses
            et_win = pose_to_euler_trans(poses[left_i:right_i])
            if pad_left > 0:
                et_win = torch.cat(
                    (torch.zeros_like(et_win)[:pad_left], et_win), dim=0)
            if pad_right > 0:
                et_win = torch.cat(
                    (et_win, torch.zeros_like(et_win)[:pad_right]), dim=0)
            et_embed = torch.cat(
                (embed_fn(et_win[:, :3]), embed_fn(et_win[:, 3:])), dim=1)
            signal_torso = PoseAttNet(et_embed)
    else:
        head_et = pose_to_euler_trans(poses[img_i].unsqueeze(0))
        signal_torso = torch.cat(
            (embed_fn(head_et[:, :3]), embed_fn(head_et[:, 3:])), dim=1)

    return signal_torso


def render_rays(decoder, p_i, r_i, z_shape_i, z_app_i, signal, head_or_torso, batch_size, bc_rgb, 
                view_dir, z_vals, args, coarse_or_fine='coarse', raw_noise_std=0):

    feat, sigma = [], []
    feat_i, sigma_i = decoder(p_i, r_i, z_shape_i, z_app_i, signal, head_or_torso)
    if coarse_or_fine == 'coarse':
        sigma_i = sigma_i.reshape(batch_size, -1, args.N_samples)
        feat_i = feat_i.reshape(batch_size, -1, args.N_samples, 3)
    if coarse_or_fine == 'fine':
        sigma_i = sigma_i.reshape(batch_size, -1, args.N_samples+args.N_importance)
        feat_i = feat_i.reshape(batch_size, -1, args.N_samples+args.N_importance, 3)

    if args.concate_bg:
        feat_i = torch.cat((feat_i[..., :-1, :], bc_rgb), dim=-2)

    feat.append(feat_i)
    sigma.append(sigma_i)
    sigma = F.relu(torch.stack(sigma, dim=0))
    feat = torch.stack(feat, dim=0)

    # Composite
    sigma_sum, feat_weighted = composite_function(sigma, feat)

    # Get Volume Weights
    weights = calc_volume_weights(z_vals, view_dir, sigma_sum, coarse_or_fine=coarse_or_fine, \
                last_dist=args.last_dist, raw_noise_std=raw_noise_std)
    feat_map = torch.sum(weights.unsqueeze(-1) * feat_weighted, dim=-2) # [b, N_rays, hidden]        
    rgb = feat_map.squeeze(0)

    return rgb, weights


def composite_function(sigma, feat):
    use_max_composition = False
    n_boxes = sigma.shape[0]
    if n_boxes > 1:
        if use_max_composition: # false
            bs, rs, ns = sigma.shape[1:]
            sigma_sum, ind = torch.max(sigma, dim=0)
            feat_weighted = feat[ind, torch.arange(bs).reshape(-1, 1, 1),
                                    torch.arange(rs).reshape(
                                        1, -1, 1), torch.arange(ns).reshape(
                                            1, 1, -1)]
        else:
            denom_sigma = torch.sum(sigma, dim=0, keepdim=True)
            denom_sigma[denom_sigma == 0] = 1e-4
            w_sigma = sigma / denom_sigma
            sigma_sum = torch.sum(sigma, dim=0)
            feat_weighted = (feat * w_sigma.unsqueeze(-1)).sum(0)
    else:
        sigma_sum = sigma.squeeze(0)
        feat_weighted = feat.squeeze(0)
    return sigma_sum, feat_weighted


def calc_volume_weights(z_vals, ray_vector, sigma, last_dist=1e10):
    dists = z_vals[..., 1:] - z_vals[..., :-1]
    dists = torch.cat([dists, torch.Tensor([last_dist]).expand(
        dists[..., :1].shape)], dim=-1)
    dists = dists * torch.norm(ray_vector, dim=-1, keepdim=True)
    alpha = 1.-torch.exp(-(F.relu(sigma)+1e-6)*dists)
    weights = alpha * \
        torch.cumprod(torch.cat([
            torch.ones_like(alpha[:, :, :1]),
            (1. - alpha + 1e-10), ], dim=-1), dim=-1)[..., :-1]
    return weights


def rot_to_euler(R):
    batch_size, _, _ = R.shape
    e = torch.ones((batch_size, 3)).cuda()

    R00 = R[:, 0, 0]
    R01 = R[:, 0, 1]
    R02 = R[:, 0, 2]
    R10 = R[:, 1, 0]
    R11 = R[:, 1, 1]
    R12 = R[:, 1, 2]
    R20 = R[:, 2, 0]
    R21 = R[:, 2, 1]
    R22 = R[:, 2, 2]
    e[:, 2] = torch.atan2(R00, -R01)
    e[:, 1] = torch.asin(-R02)
    e[:, 0] = torch.atan2(R22, R12)
    return e


def pose_to_euler_trans(poses):
    e = rot_to_euler(poses)
    t = poses[:, :3, 3]
    return torch.cat((e, t), dim=1)


def euler2rot(euler_angle):
    batch_size = euler_angle.shape[0]
    theta = euler_angle[:, 0].reshape(-1, 1, 1)
    phi = euler_angle[:, 1].reshape(-1, 1, 1)
    psi = euler_angle[:, 2].reshape(-1, 1, 1)
    one = torch.ones((batch_size, 1, 1), dtype=torch.float32,
                     device=euler_angle.device)
    zero = torch.zeros((batch_size, 1, 1), dtype=torch.float32,
                       device=euler_angle.device)
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
    
    return torch.bmm(rot_x, torch.bmm(rot_y, rot_z))


def config_parser():

    import configargparse
    parser = configargparse.ArgumentParser()
    parser.add_argument('--config', is_config_file=True,
                        help='config file path')
    parser.add_argument("--expname", type=str,
                        help='experiment name')
    parser.add_argument("--basedir", type=str, default='./logs/',
                        help='where to store ckpts and logs')
    parser.add_argument("--datadir", type=str, default='./data/llff/fern',
                        help='input data directory')
    # training options
    parser.add_argument("--netdepth", type=int, default=8,
                        help='layers in network')
    parser.add_argument("--netwidth", type=int, default=256,
                        help='channels per layer')
    parser.add_argument("--netdepth_fine", type=int, default=8,
                        help='layers in fine network')
    parser.add_argument("--netwidth_fine", type=int, default=256,
                        help='channels per layer in fine network')
    parser.add_argument("--N_rand", type=int, default=2048,
                        help='batch size (number of random rays per gradient step)')
    parser.add_argument("--lrate", type=float, default=5e-4,
                        help='learning rate')
    parser.add_argument("--lrate_decay", type=int, default=500,
                        help='exponential learning rate decay (in 1000 steps)')
    parser.add_argument("--chunk", type=int, default=4096,
                        help='number of rays processed in parallel, decrease if running out of memory')
    parser.add_argument("--netchunk", type=int, default=1024*64,
                        help='number of pts sent through network in parallel, decrease if running out of memory')
    parser.add_argument("--no_batching", action='store_false',
                        help='only take random rays from 1 image at a time')
    parser.add_argument("--no_reload", action='store_true',
                        help='do not reload weights from saved ckpt')
    parser.add_argument("--ft_path", type=str, default=None,
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--N_iters", type=int, default=400000,
                        help='number of iterations')

    # rendering options
    parser.add_argument("--N_samples", type=int, default=64,
                        help='number of coarse samples per ray')
    parser.add_argument("--N_importance", type=int, default=128,
                        help='number of additional fine samples per ray')
    parser.add_argument("--perturb", type=float, default=1.,
                        help='set to 0. for no jitter, 1. for jitter')
    parser.add_argument("--use_viewdirs", action='store_false',
                        help='use full 5D input instead of 3D')
    parser.add_argument("--i_embed", type=int, default=0,
                        help='set 0 for default positional encoding, -1 for none')
    parser.add_argument("--multires", type=int, default=10,
                        help='log2 of max freq for positional encoding (3D location)')
    parser.add_argument("--multires_views", type=int, default=4,
                        help='log2 of max freq for positional encoding (2D direction)')
    parser.add_argument("--raw_noise_std", type=float, default=0.,
                        help='std dev of noise added to regularize sigma_a output, 1e0 recommended')
    parser.add_argument("--render_only", action='store_true',
                        help='do not optimize, reload weights and render out render_poses path')
    parser.add_argument("--render_test", action='store_true',
                        help='render the test set instead of render_poses path')
    parser.add_argument("--render_factor", type=int, default=0,
                        help='downsampling factor to speed up rendering, set 4 or 8 for fast preview')

    # training options
    parser.add_argument("--precrop_iters", type=int, default=0,
                        help='number of steps to train on central crops')
    parser.add_argument("--precrop_frac", type=float,
                        default=.5, help='fraction of img taken for central crops')

    # dataset options
    parser.add_argument("--dataset_type", type=str, default='audface',
                        help='options: llff / blender / deepvoxels')
    parser.add_argument("--testskip", type=int, default=1,
                        help='will load 1/N images from test/val sets, useful for large datasets like deepvoxels')

    # deepvoxels flags
    parser.add_argument("--shape", type=str, default='greek',
                        help='options : armchair / cube / greek / vase')

    # blender flags
    parser.add_argument("--white_bkgd", action='store_false',
                        help='set to render synthetic data on a white bkgd (always use for dvoxels)')
    parser.add_argument("--half_res", action='store_true',
                        help='load blender synthetic data at 400x400 instead of 800x800')

    # face flags
    parser.add_argument("--with_test", type=int, default=0,
                        help='whether to use test set')
    parser.add_argument("--dim_aud", type=int, default=64,
                        help='dimension of audio features for NeRF')
    parser.add_argument("--sample_rate", type=float, default=0.95,
                        help="sample rate in a bounding box")
    parser.add_argument("--near", type=float, default=0.3,
                        help="near sampling plane")
    parser.add_argument("--far", type=float, default=0.9,
                        help="far sampling plane")
    parser.add_argument("--test_file", type=str, default='',
                        help='test file')
    parser.add_argument("--aud_file", type=str, default='aud.npy',
                        help='test audio deepspeech file')
    parser.add_argument("--exp_file", type=str, default='exp.pt',
                        help='exp file')
    parser.add_argument("--win_size", type=int, default=16,
                        help="windows size of audio feature")
    parser.add_argument("--smo_size", type=int, default=8,
                        help="window size for smoothing audio features")
    parser.add_argument("--smo_torse_size", type=int, default=4,
                        help="window size for smoothing torso features")
    parser.add_argument('--nosmo_iters', type=int, default=300000,
                        help='number of iterations befor applying smoothing on audio features')
    parser.add_argument('--noexp_iters', type=int, default=300000,
                        help='number of iterations befor applying expression feature on audio features')

    # llff flags
    parser.add_argument("--factor", type=int, default=8,
                        help='downsample factor for LLFF images')
    parser.add_argument("--no_ndc", action='store_true',
                        help='do not use normalized device coordinates (set for non-forward facing scenes)')
    parser.add_argument("--lindisp", action='store_true',
                        help='sampling linearly in disparity rather than depth')
    parser.add_argument("--spherify", action='store_true',
                        help='set for spherical 360 scenes')
    parser.add_argument("--llffhold", type=int, default=8,
                        help='will take every 1/N images as LLFF test set, paper uses 8')

    # logging/saving options
    parser.add_argument("--i_print",   type=int, default=100,
                        help='frequency of console printout and metric loggin')
    parser.add_argument("--i_img",     type=int, default=500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--i_weights", type=int, default=10000,
                        help='frequency of weight ckpt saving')
    parser.add_argument("--i_video",   type=int, default=50000,
                        help='frequency of render_poses video saving')

    # giraffe options
    parser.add_argument("--z_dim",          type=int, default=256,
                        help='dimension of latent code z')
    parser.add_argument("--n_feat",         type=int, default=256,
                        help='number of features')
    parser.add_argument("--image_size",     type=int, default=256,
                        help='for neural render, we need to resize image to square')
    parser.add_argument("--n_object",     type=int, default=2,
                        help='number of objects in MLP space')
    parser.add_argument("--use_giraffe",     action='store_true', 
                        help='use giraffe or graf')
    parser.add_argument("--resume",     type=str, 
                        help='resume from certain ckpt')
    parser.add_argument("--render_video",     action='store_true', 
                        help='render video')
    parser.add_argument("--render_together",     action='store_true', 
                        help='render img together')
    parser.add_argument("--alpha_sigma_loss",     type=float, 
                        help='rate of sigma loss, 1e-3 is ok')
    parser.add_argument("--concate_bg_render",     action='store_true', 
                        help='whether concate background img when rendering')
    parser.add_argument("--concate_bg",     action='store_true', 
                        help='whether concate background img when training')
    parser.add_argument("--stride",     type=int, default=2 ,
                        help='scale of img size when testing')
    parser.add_argument("--render_person",     action='store_true', 
                        help='render person with head and torso')
    parser.add_argument("--i_test_separate", type=int, default=1000,
                        help='frequency of testset saving')
    parser.add_argument("--i_test_person",     type=int, default=1000, 
                        help='test one person')
    parser.add_argument("--train_together",     action='store_true', 
                        help='train with a person loss')
    parser.add_argument("--train_separate",     action='store_true', 
                        help='train head and torso separately')
    parser.add_argument("--dim_signal", type=int, default=128,
                        help='dimension of conditional signal')
    parser.add_argument("--last_dist", type=float, default=1e10,
                        help='value of last dist concated with dist')
    parser.add_argument("--use_deformation_field", action='store_true', 
                        help='use mlp before NeRF')
    parser.add_argument("--use_expression", action='store_true', 
                        help='use expression in z_app')
    parser.add_argument("--use_et_embed", action='store_true', 
                        help='use P.E. for pose')
    parser.add_argument("--use_ba", action='store_true', 
                        help='use bundle adjustment')
    parser.add_argument("--render_final_video", action='store_true', 
                        help='render test video after the last iter')
    parser.add_argument("--no_com", action='store_true', 
                        help='data without com')
    parser.add_argument("--use_L1", action='store_true', 
                        help='data without com')
    parser.add_argument("--all_speaker", action='store_true', 
                        help='define everyone as speaker')
    parser.add_argument("--sample_rate_mouth", type=float, default=0.7,
                        help='sample rate of mouth')
    parser.add_argument("--use_exp", action='store_true', 
                        help='whether to use exp from .pt')
    parser.add_argument("--use_aud_net", action='store_true', 
                        help='whether to use aud net to z_app')
    parser.add_argument("--use_ori", action='store_true', 
                        help='whether to use origin imgs')
    
    parser.add_argument("--test_offset", type=int, default=0)
    return parser


def train():

    parser = config_parser()
    args = parser.parse_args()
    print("args.near: ", args.near)
    print("args.far: ", args.far)

    # load data
    datasets = []

    datadir = [args.datadir]
    nears = [args.near]
    fars = [args.far]

    if args.use_ba:
        with open(os.path.join(args.datadir, 'transforms_train_ba.json'), 'r') as fp:
            trans = json.load(fp)
    else:
        with open(os.path.join(args.datadir, 'transforms_train.json'), 'r') as fp:
            trans = json.load(fp)

    pose_body = torch.Tensor(trans['frames'][0]['transform_matrix'])

    for i in range(len(datadir)):
        datasets.append(load_audface_data_split(
            datadir[i], args.testskip, test_file=args.test_file, aud_file=args.aud_file, exp_file=args.exp_file, 
                    no_com=args.no_com, all_speaker=args.all_speaker, use_ori=args.use_ori, use_ba=args.use_ba, test_offset=args.test_offset
                    ))
        datasets[i]['near'] = nears[i]
        datasets[i]['far'] = fars[i]
        if not args.render_person:
            i_train, i_val = datasets[i]['i_split']
            datasets[i]['i_train'] = i_train
            datasets[i]['i_val'] = i_val

    for i in range(len(datasets)):
        datasets[i]['poses'] = torch.Tensor(datasets[i]['poses']).to(device).float()
        datasets[i]['auds'] = torch.Tensor(datasets[i]['auds']).to(device).float()
        datasets[i]['bc_img'] = torch.Tensor(datasets[i]['bc_img']).to(device).float() / 255.0
        pose_body = pose_body.to(device).float()
        if datasets[i]['exp'] is not None:
            datasets[i]['exp'] = torch.Tensor(datasets[i]['exp']).to(device).float()
        #This is for training speaker respectively
        if not args.render_person:
            datasets[i]['i_train'] = np.intersect1d(datasets[i]['i_train'], np.where(datasets[i]['speak_frames'] > 0))

    speaker_ids = []
    for i in range(len(datasets) // 2):
        speaker_id = datasets[i*2]['speak_frames'] + 2 * datasets[i*2+1]['speak_frames']
        speaker_id[speaker_id >= 3] = 0
        speaker_ids.append(speaker_id - 1)

    batch_size = 1

    # Create log dir and copy the config file
    expname = args.expname
    imgdir = []
    basedir = os.path.join('dataset/train_together', expname)
    os.makedirs(basedir, exist_ok=True)
    for i in range(args.n_object):
        i_dir = os.path.join(basedir, datadir[i].split('/')[-1])
        os.makedirs(os.path.join(i_dir, 'person'), exist_ok=True)
        imgdir.append(i_dir)
    f = os.path.join(basedir, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    #create model
    embed_fn=None
    if args.use_et_embed:
        embed_fn, input_ch = get_embedder(3, 0)
        dim_torso_signal = 2*input_ch

    decoder = Decoder(z_dim=args.z_dim, hidden_size=args.n_feat, dim_signal=args.dim_signal, 
                    use_deformation_field=args.use_deformation_field, use_expression=args.use_expression,
                    use_aud_net=args.use_aud_net).to(device)

    optimizer_decoder = torch.optim.Adam(
        params=list(decoder.parameters()), lr=args.lrate, betas=(0.9, 0.999)
        )

    AudNet = AudioNet_W2L().to(device)

    ExpNet = ExpressionEnc().to(device)
    optimizer_Aud = torch.optim.Adam(
        params=list(AudNet.parameters()), lr=args.lrate, betas=(0.9, 0.999))
    optimizer_Exp = torch.optim.Adam(
        params=list(ExpNet.parameters()), lr=args.lrate, betas=(0.9, 0.999))

    AudAttNet = AudioAttNet(dim_aud=args.dim_aud, seq_len=args.smo_size).to(device)
    optimizer_AudAtt = torch.optim.Adam(
        params=list(AudAttNet.parameters()), lr=args.lrate, betas=(0.9, 0.999))

    if args.use_et_embed:
        PoseAttNet = AudioAttNet(dim_aud=dim_torso_signal, seq_len=args.smo_torse_size).to(device)
        optimizer_PoseAtt = torch.optim.Adam(
            params=list(PoseAttNet.parameters()), lr=args.lrate, betas=(0.9, 0.999))
    
    if args.use_L1:
        L1_loss = nn.L1Loss()
    
    z_shape = torch.randn(batch_size, args.n_object*2, args.z_dim).to(device)
    z_app = torch.randn(batch_size, args.n_object*2, args.z_dim).to(device)

    start = 0
    global_step = start

     # resume
    if args.resume is not None:
        state_dict = torch.load(args.resume)
        state_dict_keys = state_dict.keys()
        global_step = state_dict['global_step']
        z_shape = state_dict['z_shape']
        z_app = state_dict['z_app']
        decoder.load_state_dict(state_dict['network_decoder_state_dict'])
        optimizer_decoder.load_state_dict(state_dict['optimizer_decoder_state_dict'])
        if 'network_AudNet_state_dict' in state_dict_keys:
            AudNet.load_state_dict(state_dict['network_AudNet_state_dict'])
            print("load audnet.")
        if 'network_ExpNet_state_dict' in state_dict_keys:
            ExpNet.load_state_dict(state_dict['network_ExpNet_state_dict'])
            print("load expnet.")
        if 'optimizer_Aud_state_dict' in state_dict_keys:
            optimizer_Aud.load_state_dict(state_dict['optimizer_Aud_state_dict'])
        if 'optimizer_Exp_state_dict' in state_dict_keys:
            optimizer_Exp.load_state_dict(state_dict['optimizer_Exp_state_dict'])
        if 'network_AudAttNet_state_dict' in state_dict_keys:
            AudAttNet.load_state_dict(state_dict['network_AudAttNet_state_dict'])
            print("load audAttNet.")

        if 'optimizer_AudAtt_state_dict' in state_dict_keys:
            optimizer_AudAtt.load_state_dict(state_dict['optimizer_AudAtt_state_dict'])
        if 'network_PoseAttNet_state_dict' in state_dict_keys:
            PoseAttNet.load_state_dict(state_dict['network_PoseAttNet_state_dict'])
        if 'optimizer_PoseAtt_state_dict' in state_dict_keys:
            optimizer_PoseAtt.load_state_dict(state_dict['optimizer_PoseAtt_state_dict'])

    N_rand = args.N_rand
    print('N_rand', N_rand, 'no_batching',
          args.no_batching, 'sample_rate', args.sample_rate)

    N_iters = args.N_iters + 1
    print('Begin')
    
    # render one person
    if args.render_person:
        print('RENDER PERSON')
        with torch.no_grad():

            dataset = datasets

            for itr_obj in range(args.n_object): 
                testsavedir_person = os.path.join(
                    imgdir[itr_obj], 'person', 'render_com')
                testsavedir_head = os.path.join(
                    imgdir[itr_obj], 'person', 'render_head')
                
                os.makedirs(testsavedir_person, exist_ok=True)
                os.makedirs(testsavedir_head, exist_ok=True)
                
                poses = dataset[itr_obj]['poses']
                bc_img = dataset[itr_obj]['bc_img']
                hwfcxy = dataset[itr_obj]['hwfcxy']
                H, W, focal, cx, cy = hwfcxy
                H, W = int(H), int(W)
                hwfcxy = [H, W, focal, cx, cy]

                near = dataset[itr_obj]['near']
                far = dataset[itr_obj]['far']
                near, far = near * \
                    torch.ones((H*W, 1)), far * \
                    torch.ones((H*W, 1))
                t_vals = torch.linspace(0., 1., steps=args.N_samples)
                z_vals = near * (1.-t_vals) + far * (t_vals)
                z_vals = z_vals.expand([H*W, args.N_samples]).to(device)

                rgbs = []
                num_items = dataset[itr_obj]['auds'].shape[0]
                print("num_items: ", num_items)
                for img_i in range(num_items):
                    
                    #signal
                    signal = encode_signal(dataset, itr_obj, img_i, args.dim_aud, AudNet, ExpNet, AudAttNet, global_step, args, num_items,
                                     embed_fn=embed_fn)
                    signal_torso = encode_signal_torso(dataset, itr_obj, img_i, PoseAttNet, global_step, args, num_items,
                                     embed_fn=embed_fn)
                                    
                    #head
                    rays_o, rays_d = get_rays(   
                        H, W, focal, poses[img_i, :3, :4], cx, cy)                                                    # (H, W, 3), (H, W, 3)
                    rays_o = rays_o.reshape(-1, 3)                                                                    # (H * W, 3)
                    rays_d = rays_d.reshape(-1, 3)                                                                    # (H * W, 3)

                    p_i = rays_o[..., None, :] + rays_d[..., None, :] * \
                        z_vals[..., :, None]                                                                          # [H*W, N_samples, 3]
                    p_i = p_i.reshape(batch_size, -1, 3)                                                              # [B, H*W * N_samples, 3]
                    r_i = rays_d.unsqueeze(1).expand([H*W, args.N_samples, 3]).reshape(batch_size, -1, 3)             # [B, H*W * N_samples, 3]

                    #torso
                    rays_o_torso, rays_d_torso = get_rays(   
                        H, W, focal, pose_body[:3, :4], cx, cy)                                                       # (H, W, 3), (H, W, 3)
                    rays_o_torso = rays_o_torso.reshape(-1, 3)                                                        # (H * W, 3)
                    rays_d_torso = rays_d_torso.reshape(-1, 3)                                                        # (H * W, 3)
                    p_i_torso = rays_o_torso[..., None, :] + rays_d_torso[..., None, :] * \
                        z_vals[..., :, None]                                                                          # [H*W, N_samples, 3]
                    p_i_torso = p_i_torso.reshape(batch_size, -1, 3)                                                  # [B, H*W * N_samples, 3]
                    r_i_torso = rays_d_torso.unsqueeze(1).expand([H*W, args.N_samples, 3]).reshape(batch_size, -1, 3) # [B, H*W * N_samples, 3]

                    chunk = args.chunk
                    rgb_map, rgb_map_torso = [], []
                    for i_chunk in list(np.arange(H*W / chunk, dtype=np.int32)):
                        p_chunk_i = p_i[:, i_chunk*args.N_samples*chunk : (i_chunk+1)*args.N_samples*chunk, :]
                        r_chunk_i = r_i[:, i_chunk*args.N_samples*chunk : (i_chunk+1)*args.N_samples*chunk, :]
                        p_chunk_i_torso = p_i_torso[:, i_chunk*args.N_samples*chunk : (i_chunk+1)*args.N_samples*chunk, :]
                        r_chunk_i_torso = r_i_torso[:, i_chunk*args.N_samples*chunk : (i_chunk+1)*args.N_samples*chunk, :]

                        feat, sigma = [], []
                        feat_torso, sigma_torso = [], []
                        
                        z_shape_i = z_shape[:,itr_obj*2]
                        z_app_i = z_app[:,itr_obj*2]
                        feat_i, sigma_i = decoder(p_chunk_i, r_chunk_i, z_shape_i, z_app_i, signal, 'head')
                        sigma_i = sigma_i.reshape(batch_size, -1, args.N_samples)
                        feat_i = feat_i.reshape(batch_size, -1, args.N_samples, 3)
                        if args.concate_bg:
                            bc_rgb = bc_img.reshape(batch_size, H*W, 1, 3)[:, i_chunk*chunk : (i_chunk+1)*chunk, :, :]
                            feat_i = torch.cat((feat_i[..., :-1, :], bc_rgb), dim=-2)
                        
                        z_shape_i_torso = z_shape[:,itr_obj*2+1]
                        z_app_i_torso = z_app[:,itr_obj*2+1]
                        feat_i_torso, sigma_i_torso = decoder(p_chunk_i_torso, r_chunk_i_torso, z_shape_i_torso, z_app_i_torso, signal_torso, 'torso')
                        sigma_i_torso = sigma_i_torso.reshape(batch_size, -1, args.N_samples)
                        feat_i_torso = feat_i_torso.reshape(batch_size, -1, args.N_samples, 3)
                        if args.concate_bg:
                            sigma_i_torso[:, :, -1] = 0

                        feat_torso.append(feat_i)
                        sigma_torso.append(sigma_i)
                        feat_torso.append(feat_i_torso)
                        sigma_torso.append(sigma_i_torso)
                        feat.append(feat_i)
                        sigma.append(sigma_i)
                        
                        sigma = F.relu(torch.stack(sigma, dim=0))
                        feat = torch.stack(feat, dim=0)
                        sigma_torso = F.relu(torch.stack(sigma_torso, dim=0))
                        feat_torso = torch.stack(feat_torso, dim=0)
                        if args.concate_bg:
                            sigma[-1, :, :, -1] = sigma[-1, :, :, -1] + 1e-6
                            sigma_torso[-1, :, :, -1] = sigma_torso[-1, :, :, -1] + 1e-6
                            
                        # Composite
                        sigma_sum, feat_weighted = composite_function(sigma, feat)
                        sigma_torso_sum, feat_torso_weighted = composite_function(sigma_torso, feat_torso)

                        # Get Volume Weights
                        z_val_chunk_i = z_vals.unsqueeze(0)[:, i_chunk*chunk : (i_chunk+1)*chunk, :]
                        rays_d_torso_chunk_i = rays_d_torso.unsqueeze(0).repeat(batch_size,1,1)[:, i_chunk*chunk : (i_chunk+1)*chunk, :]
                        rays_d_head_chunk_i = rays_d.unsqueeze(0).repeat(batch_size,1,1)[:, i_chunk*chunk : (i_chunk+1)*chunk, :]
                        weights = calc_volume_weights(z_val_chunk_i, rays_d_head_chunk_i, sigma_sum, last_dist=args.last_dist)
                        weights_torso = calc_volume_weights(z_val_chunk_i, rays_d_torso_chunk_i, sigma_torso_sum, last_dist=args.last_dist)
                        feat_map = torch.sum(weights.unsqueeze(-1) * feat_weighted, dim=-2) ## [b, chunk, hidden]
                        rgb_map.append(feat_map)
                        feat_map_torso = torch.sum(weights_torso.unsqueeze(-1) * feat_torso_weighted, dim=-2) ## [b, chunk, hidden]
                        rgb_map_torso.append(feat_map_torso)

                    
                    rgb = torch.cat(rgb_map_torso, dim=1).squeeze(0).reshape(H, W, 3)
                    rgb_head = torch.cat(rgb_map, dim=1).squeeze(0).reshape(H, W, 3)
                    rgb8 = to8b(rgb.detach().cpu().numpy())
                    rgb8_head = to8b(rgb_head.detach().cpu().numpy())

                    ##########################
                    if img_i % 1 == 0:
                        filename = os.path.join(testsavedir_person, 'test_{:06d}.jpg'.format(img_i))
                        imageio.imwrite(filename, rgb8)
                        filename = os.path.join(testsavedir_head, 'test_{:06d}.jpg'.format(img_i))
                        imageio.imwrite(filename, rgb8_head)
                        print('Saved test img at {}'.format(filename))
                    ##########################
                    if args.render_video:
                        rgbs.append(rgb8)
                        print("Object: " + str(itr_obj) + "  Img: " + str(img_i))

                if args.render_video:
                    filename = os.path.join(testsavedir_person, '{}.mp4'.format(expname))
                    imageio.mimwrite(filename, rgbs, fps=25, quality=8)
                    print('Saved test video')

            return
                    
    # start training
    start = global_step + 1
    for i in trange(start, N_iters):
        time0 = time.time()

        dataset = datasets

        for itr_obj in range(args.n_object):

            images_com = dataset[itr_obj]['imgs_com']
            images_head_neck = dataset[itr_obj]['imgs']
            # images_ori = dataset[itr_obj]['imgs_ori']
            poses = dataset[itr_obj]['poses']
            bc_img = dataset[itr_obj]['bc_img']
            hwfcxy = dataset[itr_obj]['hwfcxy']
            sample_rects = dataset[itr_obj]['sample_rects']
            near = dataset[itr_obj]['near']
            far = dataset[itr_obj]['far']

            # transform ray to 3d point and view direction
            near, far = near * \
                torch.ones((N_rand, 1)), far * \
                torch.ones((N_rand, 1))

            t_vals = torch.linspace(0., 1., steps=args.N_samples)
            z_vals = near * (1.-t_vals) + far * (t_vals)
            z_vals = z_vals.expand([N_rand, args.N_samples]).to(device)

            H, W, focal, cx, cy = hwfcxy
            H, W = int(H), int(W)
            hwfcxy = [H, W, focal, cx, cy]

            # Random from one image
            i_train = dataset[itr_obj]['i_train']
            img_i = np.random.choice(i_train)
            target_com = torch.as_tensor(imageio.imread(
                images_com[img_i])).to(device).float()/255.0
            target_head_neck = torch.as_tensor(imageio.imread(
                images_head_neck[img_i])).to(device).float()/255.0
            pose = poses[img_i, :3, :4]
            pose_torso = poses[0, :3, :4]
            rect = sample_rects[img_i]

            signal = encode_signal(dataset, itr_obj, img_i, args.dim_aud, AudNet, ExpNet, AudAttNet, global_step, args, len(i_train),
                             embed_fn=embed_fn)
            signal_torso = encode_signal_torso(dataset, itr_obj, img_i, PoseAttNet, global_step, args, len(i_train),
                             embed_fn=embed_fn)

            loss = 0

            # select coods
            coords = torch.stack(torch.meshgrid(torch.linspace(
                0, H-1, H), torch.linspace(0, W-1, W)), -1)  # (H, W, 2)
            coords = torch.reshape(coords, [-1, 2])  # (H * W, 2)

            if args.sample_rate > 0:  # 0.95
                rect_inds = (coords[:, 0] >= rect[0]) & (
                    coords[:, 0] <= rect[0] + rect[2]) & (
                        coords[:, 1] >= rect[1]) & (
                            coords[:, 1] <= rect[1] + rect[3])
                rect_torso = [1*H/2, 0, H/2, W]
                rect_inds_torso = (coords[:, 0] >= rect_torso[0]) & (
                    coords[:, 0] <= rect_torso[0] + rect_torso[2]) & (
                        coords[:, 1] >= rect_torso[1]) & (
                            coords[:, 1] <= rect_torso[1] + rect_torso[3])
                rect_inds = rect_inds | rect_inds_torso
                coords_rect = coords[rect_inds]
                coords_norect = coords[~rect_inds]
                rect_num = int(N_rand*args.sample_rate)
                norect_num = N_rand - rect_num
                select_inds_rect = np.random.choice(
                    coords_rect.shape[0], size=[rect_num], replace=False)  # (N_rand,)
                # (N_rand, 2)
                select_coords_rect = coords_rect[select_inds_rect].long()
                select_inds_norect = np.random.choice(
                    coords_norect.shape[0], size=[norect_num], replace=False)  # (N_rand,)
                # (N_rand, 2)
                select_coords_norect = coords_norect[select_inds_norect].long(
                )
                select_coords = torch.cat(
                    (select_coords_rect, select_coords_norect), dim=0)
            else:
                select_inds = np.random.choice(
                    coords.shape[0], size=[N_rand], replace=False)  # (N_rand,)
                select_coords = coords[select_inds].long()

            target_s_com = target_com[select_coords[:, 0],
                                select_coords[:, 1]]  # (N_rand, 3)
            target_s_head_neck = target_head_neck[select_coords[:, 0],
                                select_coords[:, 1]]  # (N_rand, 3)
            bc_rgb = bc_img[select_coords[:, 0],
                            select_coords[:, 1]]

            #head
            # signal_exp_aud = get_half_signals(H, W, signal)
            rays_o, rays_d = get_rays(   
                H, W, focal, pose, cx, cy)  # (H, W, 3), (H, W, 3)
            rays_o = rays_o[select_coords[:, 0],
                            select_coords[:, 1]]  # (N_rand, 3)
            rays_d = rays_d[select_coords[:, 0],
                            select_coords[:, 1]]  # (N_rand, 3)

            p_i = rays_o[..., None, :] + rays_d[..., None, :] * \
                z_vals[..., :, None]  # [N_rand, N_samples, 3]
            p_i = p_i.reshape(batch_size, -1, 3) # [B, N_rand * N_samples, 3]
            r_i = rays_d.unsqueeze(1).expand([N_rand, args.N_samples, 3]).reshape(batch_size, -1, 3) # [B, N_rand * N_samples, 3]

            #torso
            rays_o_torso, rays_d_torso = get_rays(   
                H, W, focal, pose_torso, cx, cy)  # (H, W, 3), (H, W, 3)
            rays_o_torso = rays_o_torso[select_coords[:, 0],
                            select_coords[:, 1]]  # (N_rand, 3)
            rays_d_torso = rays_d_torso[select_coords[:, 0],
                            select_coords[:, 1]]  # (N_rand, 3)
            p_i_torso = rays_o_torso[..., None, :] + rays_d_torso[..., None, :] * \
                z_vals[..., :, None]  # [N_rand, N_samples, 3]
            p_i_torso = p_i_torso.reshape(batch_size, -1, 3) # [B, N_rand * N_samples, 3]
            r_i_torso = rays_d_torso.unsqueeze(1).expand([N_rand, args.N_samples, 3]).reshape(batch_size, -1, 3) # [B, N_rand * N_samples, 3]

            #####  Core optimization loop  #####
            feat, sigma = [], []
            feat_torso, sigma_torso = [], []
            z_shape_i = z_shape[:,itr_obj*2]
            z_app_i = z_app[:,itr_obj*2]        
            feat_i, sigma_i = decoder(p_i, r_i, z_shape_i, z_app_i, signal, 'head')               
            sigma_i = sigma_i.reshape(batch_size, N_rand, args.N_samples)
            feat_i = feat_i.reshape(batch_size, N_rand, args.N_samples, -1)
            if args.concate_bg:
                feat_i = torch.cat((feat_i[..., :-1, :], bc_rgb.reshape(batch_size, N_rand, 1, 3)), dim=-2)

            z_shape_i_torso = z_shape[:,itr_obj*2+1]
            z_app_i_torso = z_app[:,itr_obj*2+1]        
            feat_i_torso, sigma_i_torso = decoder(p_i_torso, r_i_torso, z_shape_i_torso, z_app_i_torso, signal_torso, 'torso')               
            sigma_i_torso = sigma_i_torso.reshape(batch_size, N_rand, args.N_samples)
            feat_i_torso = feat_i_torso.reshape(batch_size, N_rand, args.N_samples, -1)
            if args.concate_bg:
                sigma_i_torso[:, :, -1] = 0
           
            feat_torso.append(feat_i)
            sigma_torso.append(sigma_i)
            feat_torso.append(feat_i_torso)
            sigma_torso.append(sigma_i_torso)
            feat.append(feat_i)
            sigma.append(sigma_i)


            sigma = F.relu(torch.stack(sigma, dim=0))
            sigma_torso = F.relu(torch.stack(sigma_torso, dim=0))
            if args.concate_bg:
                sigma[-1, :, :, -1] = sigma[-1, :, :, -1] + 1e-6
                sigma_torso[-1, :, :, -1] = sigma_torso[-1, :, :, -1] + 1e-6
            feat = torch.stack(feat, dim=0)
            feat_torso = torch.stack(feat_torso, dim=0)

            # Composite
            sigma_sum, feat_weighted = composite_function(sigma, feat)
            sigma_torso_sum, feat_torso_weighted = composite_function(sigma_torso, feat_torso)
            # Get Volume Weights
            weights = calc_volume_weights(z_vals.unsqueeze(0), rays_d.unsqueeze(0).repeat(batch_size,1,1), sigma_sum, last_dist=args.last_dist)
            weights_torso = calc_volume_weights(z_vals.unsqueeze(0), rays_d_torso.unsqueeze(0).repeat(batch_size,1,1), sigma_torso_sum, last_dist=args.last_dist)
            feat_map = torch.sum(weights.unsqueeze(-1) * feat_weighted, dim=-2) ## [b, N_rays, hidden]        
            rgb_com = feat_map.squeeze(0)
            feat_map_torso = torch.sum(weights_torso.unsqueeze(-1) * feat_torso_weighted, dim=-2) ## [b, N_rays, hidden]        
            rgb_com_torso = feat_map_torso.squeeze(0)

            # loss
            img_loss_head_neck = img2mse(rgb_com, target_s_head_neck)
            img_loss_com = img2mse(rgb_com_torso, target_s_com)
            psnr_head_neck = mse2psnr(img_loss_head_neck)
            psnr_com = mse2psnr(img_loss_com)
            loss += img_loss_com
            loss += img_loss_head_neck

            if args.use_L1:
                loss = 0
                if args.train_together:
                    loss += L1_loss(rgb_com, target_s_com)
                

            #optimize
            optimizer_decoder.zero_grad()
            optimizer_Aud.zero_grad()
            optimizer_Exp.zero_grad()
            optimizer_AudAtt.zero_grad()
            if args.use_et_embed:
                optimizer_PoseAtt.zero_grad()
            
            loss.backward()
            optimizer_decoder.step()
            optimizer_Aud.step()
            if global_step >= args.nosmo_iters:
                optimizer_AudAtt.step()
                if args.use_et_embed:
                    optimizer_PoseAtt.step()
            if global_step >= args.noexp_iters:
                optimizer_Exp.step()

            #log
            if i % args.i_print == 0:
                f = os.path.join(basedir, 'loss.txt')
                tqdm.write(
                    f"[TRAIN] Iter: {i} Object: {itr_obj} Com Loss: {img_loss_com.item()}  Head Neck PSNR: {psnr_head_neck.item()} Com PSNR: {psnr_com.item()}")
                with open(f, 'a') as file:
                    file.write(
                    f"[TRAIN] Iter: {i} Object: {itr_obj} Com Loss: {img_loss_com.item()}  Head Neck PSNR: {psnr_head_neck.item()} Com PSNR: {psnr_com.item()}\n")


        if (i % args.i_test_person == 0 and i > 0) or (i in [100, 500, 1000, 3000]):
            with torch.no_grad():
                dataset = datasets

                for itr_obj in range(args.n_object):
                    testsavedir_person = os.path.join(
                        imgdir[itr_obj], 'person', 'test_{}'.format(i))
                    os.makedirs(testsavedir_person, exist_ok=True)

                    poses = dataset[itr_obj]['poses']
                    bc_img = dataset[itr_obj]['bc_img']
                    hwfcxy = dataset[itr_obj]['hwfcxy']
                    images_com = dataset[itr_obj]['imgs_com']
                    H, W, focal, cx, cy = hwfcxy
                    H, W = int(H), int(W)
                    hwfcxy = [H, W, focal, cx, cy]

                    near = dataset[itr_obj]['near']
                    far = dataset[itr_obj]['far']
                    near, far = near * \
                        torch.ones((H*W, 1)), far * \
                        torch.ones((H*W, 1))
                    t_vals = torch.linspace(0., 1., steps=args.N_samples)
                    z_vals = near * (1.-t_vals) + far * (t_vals)
                    z_vals = z_vals.expand([H*W, args.N_samples]).to(device)

                    i_val = dataset[itr_obj]['i_val']
                    for testimg_i in range(0, len(i_val), 100):
                        img_i = i_val[testimg_i]

                        signal = encode_signal(dataset, itr_obj, img_i, args.dim_aud, AudNet, ExpNet, AudAttNet, global_step, args, len(i_train) + len(i_val),
                                            embed_fn=embed_fn)
                        signal_torso = encode_signal_torso(dataset, itr_obj, img_i, PoseAttNet, global_step, args, len(i_train) + len(i_val),
                                            embed_fn=embed_fn)

                        #head
                        rays_o, rays_d = get_rays(   
                            H, W, focal, poses[img_i, :3, :4], cx, cy)  # (H, W, 3), (H, W, 3)
                        rays_o = rays_o.reshape(-1, 3) # (H * W, 3)
                        rays_d = rays_d.reshape(-1, 3) # (H * W, 3)
                        p_i = rays_o[..., None, :] + rays_d[..., None, :] * \
                            z_vals[..., :, None]  # [H*W, N_samples, 3]
                        p_i = p_i.reshape(batch_size, -1, 3) # [B, H*W * N_samples, 3]
                        r_i = rays_d.unsqueeze(1).expand([H*W, args.N_samples, 3]).reshape(batch_size, -1, 3) # [B, H*W * N_samples, 3]

                        #torso
                        rays_o_torso, rays_d_torso = get_rays(   
                            H, W, focal, poses[0, :3, :4], cx, cy)  # (H, W, 3), (H, W, 3)
                        rays_o_torso = rays_o_torso.reshape(-1, 3) # (H * W, 3)
                        rays_d_torso = rays_d_torso.reshape(-1, 3) # (H * W, 3)
                        p_i_torso = rays_o_torso[..., None, :] + rays_d_torso[..., None, :] * \
                            z_vals[..., :, None]  # [H*W, N_samples, 3]
                        p_i_torso = p_i_torso.reshape(batch_size, -1, 3) # [B, H*W * N_samples, 3]
                        r_i_torso = rays_d_torso.unsqueeze(1).expand([H*W, args.N_samples, 3]).reshape(batch_size, -1, 3) # [B, H*W * N_samples, 3]

                        chunk = args.chunk
                        rgb_map, rgb_map_torso = [], []
                        for i_chunk in list(np.arange(H*W / chunk, dtype=np.int32)):
                            p_chunk_i = p_i[:, i_chunk*args.N_samples*chunk : (i_chunk+1)*args.N_samples*chunk, :]
                            r_chunk_i = r_i[:, i_chunk*args.N_samples*chunk : (i_chunk+1)*args.N_samples*chunk, :]
                            p_chunk_i_torso = p_i_torso[:, i_chunk*args.N_samples*chunk : (i_chunk+1)*args.N_samples*chunk, :]
                            r_chunk_i_torso = r_i_torso[:, i_chunk*args.N_samples*chunk : (i_chunk+1)*args.N_samples*chunk, :]

                            feat, sigma = [], []
                            feat_torso, sigma_torso = [], []
                            
                            z_shape_i = z_shape[:,itr_obj*2]
                            z_app_i = z_app[:,itr_obj*2]
                            feat_i, sigma_i = decoder(p_chunk_i, r_chunk_i, z_shape_i, z_app_i, signal, 'head')
                            sigma_i = sigma_i.reshape(batch_size, -1, args.N_samples)
                            feat_i = feat_i.reshape(batch_size, -1, args.N_samples, 3)
                            if args.concate_bg:
                                bc_rgb = bc_img.reshape(batch_size, H*W, 1, 3)[:, i_chunk*chunk : (i_chunk+1)*chunk, :, :]
                                feat_i = torch.cat((feat_i[..., :-1, :], bc_rgb), dim=-2)
                            
                            z_shape_i_torso = z_shape[:,itr_obj*2+1]
                            z_app_i_torso = z_app[:,itr_obj*2+1]
                            feat_i_torso, sigma_i_torso = decoder(p_chunk_i_torso, r_chunk_i_torso, z_shape_i_torso, z_app_i_torso,signal_torso, 'torso')
                            sigma_i_torso = sigma_i_torso.reshape(batch_size, -1, args.N_samples)
                            feat_i_torso = feat_i_torso.reshape(batch_size, -1, args.N_samples, 3)
                            if args.concate_bg:
                                sigma_i_torso[:, :, -1] = 0

                            feat_torso.append(feat_i)
                            sigma_torso.append(sigma_i)
                            feat_torso.append(feat_i_torso)
                            sigma_torso.append(sigma_i_torso)
                            # feat.append(feat_i_torso)
                            # sigma.append(sigma_i_torso)
                            feat.append(feat_i)
                            sigma.append(sigma_i)
            
                            sigma = F.relu(torch.stack(sigma, dim=0))
                            feat = torch.stack(feat, dim=0)
                            sigma_torso = F.relu(torch.stack(sigma_torso, dim=0))
                            feat_torso = torch.stack(feat_torso, dim=0)
                            if args.concate_bg:
                                sigma[-1, :, :, -1] = sigma[-1, :, :, -1] + 1e-6
                                sigma_torso[-1, :, :, -1] = sigma_torso[-1, :, :, -1] + 1e-6
                            # Composite
                            sigma_sum, feat_weighted = composite_function(sigma, feat)
                            sigma_torso_sum, feat_torso_weighted = composite_function(sigma_torso, feat_torso)
                            # Get Volume Weights
                            z_val_chunk_i = z_vals.unsqueeze(0)[:, i_chunk*chunk : (i_chunk+1)*chunk, :]
                            rays_d_torso_chunk_i = rays_d_torso.unsqueeze(0).repeat(batch_size,1,1)[:, i_chunk*chunk : (i_chunk+1)*chunk, :]
                            rays_d_head_chunk_i = rays_d.unsqueeze(0).repeat(batch_size,1,1)[:, i_chunk*chunk : (i_chunk+1)*chunk, :]
                            weights = calc_volume_weights(z_val_chunk_i, rays_d_head_chunk_i, sigma_sum, last_dist=args.last_dist)
                            weights_torso = calc_volume_weights(z_val_chunk_i, rays_d_torso_chunk_i, sigma_torso_sum, last_dist=args.last_dist)
                            feat_map = torch.sum(weights.unsqueeze(-1) * feat_weighted, dim=-2) ## [b, chunk, hidden]
                            rgb_map.append(feat_map)
                            feat_map_torso = torch.sum(weights_torso.unsqueeze(-1) * feat_torso_weighted, dim=-2) ## [b, chunk, hidden]
                            rgb_map_torso.append(feat_map_torso)

                        # save head imgs
                        rgb = torch.cat(rgb_map, dim=1).squeeze(0).reshape(H, W, 3)
                        rgb8 = to8b(rgb.detach().cpu().numpy())
                        target_img = imageio.imread(images_head_neck[img_i])
                        filename = os.path.join(testsavedir_person, 'test_head_{:03d}.jpg'.format(testimg_i))
                        imageio.imwrite(filename, np.concatenate((rgb8, target_img), axis=1))

                        # save torso imgs
                        rgb = torch.cat(rgb_map_torso, dim=1).squeeze(0).reshape(H, W, 3)
                        rgb8 = to8b(rgb.detach().cpu().numpy())
                        target_img = imageio.imread(images_com[img_i])
                        filename = os.path.join(testsavedir_person, 'test_{:03d}.jpg'.format(testimg_i))
                        imageio.imwrite(filename, np.concatenate((rgb8, target_img), axis=1))

                        target_rgb = torch.as_tensor(target_img).to(device).float()/255.0
                        img_loss_rgb = img2mse(rgb, target_rgb)
                        psnr_rgb = mse2psnr(img_loss_rgb)
                        print('Saved test person img, psnr: {}'.format(psnr_rgb.item()))
                        f = os.path.join(basedir, 'loss.txt')
                        with open(f, 'a') as file:
                            file.write(
                            f"[TEST] Iter: {i} Object: {itr_obj}_person PSNR: {psnr_rgb.item()}\n")

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1500
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer_decoder.param_groups:
            param_group['lr'] = new_lrate

        for param_group in optimizer_Aud.param_groups:
            param_group['lr'] = new_lrate

        for param_group in optimizer_AudAtt.param_groups:
            param_group['lr'] = new_lrate*2

        for param_group in optimizer_PoseAtt.param_groups:
            param_group['lr'] = new_lrate*2

        ################################
        global_step += 1
        # Rest is logging
        if i % args.i_weights == 0:
            path = os.path.join(basedir, '{:06d}.tar'.format(i))
            torch.save({
                'global_step': global_step,
                'z_shape':z_shape,
                'z_app':z_app,
                'network_decoder_state_dict': decoder.state_dict(),
                'network_AudNet_state_dict': AudNet.state_dict(),
                'network_ExpNet_state_dict': ExpNet.state_dict(),
                'optimizer_decoder_state_dict': optimizer_decoder.state_dict(),
                'optimizer_Aud_state_dict': optimizer_Aud.state_dict(),
                'optimizer_Exp_state_dict': optimizer_Exp.state_dict(),
                "network_AudAttNet_state_dict": AudAttNet.state_dict(),
                "optimizer_AudAtt_state_dict": optimizer_AudAtt.state_dict(),
                "network_PoseAttNet_state_dict": PoseAttNet.state_dict(),
                "optimizer_PoseAtt_state_dict": optimizer_PoseAtt.state_dict()
            }, path)

            print('Saved checkpoints at', path)

        
        dt = time.time()-time0
    
    if args.render_final_video:
        print('RENDER PERSON')
        with torch.no_grad():

            dataset = datasets
                    
            for itr_obj in range(args.n_object): 
                testsavedir_person = os.path.join(
                    imgdir[itr_obj], 'person')
                os.makedirs(testsavedir_person, exist_ok=True)

                imgs_com = dataset[itr_obj]['imgs_com']
                poses = dataset[itr_obj]['poses']
                bc_img = dataset[itr_obj]['bc_img']
                hwfcxy = dataset[itr_obj]['hwfcxy']
                H, W, focal, cx, cy = hwfcxy
                H, W = int(H), int(W)
                hwfcxy = [H, W, focal, cx, cy]

                near = dataset[itr_obj]['near']
                far = dataset[itr_obj]['far']
                near, far = near * \
                    torch.ones((H*W, 1)), far * \
                    torch.ones((H*W, 1))
                t_vals = torch.linspace(0., 1., steps=args.N_samples)
                z_vals = near * (1.-t_vals) + far * (t_vals)
                z_vals = z_vals.expand([H*W, args.N_samples]).to(device)
                
                rgbs = []
                for img_i in i_val:

                    signal = encode_signal(dataset, itr_obj, img_i, args.dim_aud, AudNet, ExpNet, AudAttNet, global_step, args, len(i_val),
                                     embed_fn=embed_fn)
                    signal_torso = encode_signal_torso(dataset, itr_obj, img_i, PoseAttNet, global_step, args, len(i_val),
                                     embed_fn=embed_fn)
                                    
                    # head
                    rays_o, rays_d = get_rays(   
                        H, W, focal, poses[img_i, :3, :4], cx, cy)  # (H, W, 3), (H, W, 3)
                    rays_o = rays_o.reshape(-1, 3) # (H * W, 3)
                    rays_d = rays_d.reshape(-1, 3) # (H * W, 3)
                    signal = signal.reshape(-1, signal.shape[-1]) # (H * W, 64)
                    p_i = rays_o[..., None, :] + rays_d[..., None, :] * \
                        z_vals[..., :, None]  # [H*W, N_samples, 3]
                    p_i = p_i.reshape(batch_size, -1, 3) # [B, H*W * N_samples, 3]
                    r_i = rays_d.unsqueeze(1).expand([H*W, args.N_samples, 3]).reshape(batch_size, -1, 3) # [B, H*W * N_samples, 3]
                    signal = signal.unsqueeze(1).expand([H*W, args.N_samples, -1])
                    signal = signal.reshape(batch_size, -1, signal.shape[-1]) # [B, H*W * N_samples, 3]

                    #torso
                    rays_o_torso, rays_d_torso = get_rays(   
                        H, W, focal, pose_body[:3, :4], cx, cy)  # (H, W, 3), (H, W, 3)
                    rays_o_torso = rays_o_torso.reshape(-1, 3) # (H * W, 3)
                    rays_d_torso = rays_d_torso.reshape(-1, 3) # (H * W, 3)
                    p_i_torso = rays_o_torso[..., None, :] + rays_d_torso[..., None, :] * \
                        z_vals[..., :, None]  # [H*W, N_samples, 3]
                    p_i_torso = p_i_torso.reshape(batch_size, -1, 3) # [B, H*W * N_samples, 3]
                    r_i_torso = rays_d_torso.unsqueeze(1).expand([H*W, args.N_samples, 3]).reshape(batch_size, -1, 3) # [B, H*W * N_samples, 3]

                    chunk = args.chunk
                    rgb_map, rgb_map_torso = [], []
                    for i_chunk in list(np.arange(H*W / chunk, dtype=np.int32)):
                        p_chunk_i = p_i[:, i_chunk*args.N_samples*chunk : (i_chunk+1)*args.N_samples*chunk, :]
                        r_chunk_i = r_i[:, i_chunk*args.N_samples*chunk : (i_chunk+1)*args.N_samples*chunk, :]
                        p_chunk_i_torso = p_i_torso[:, i_chunk*args.N_samples*chunk : (i_chunk+1)*args.N_samples*chunk, :]
                        r_chunk_i_torso = r_i_torso[:, i_chunk*args.N_samples*chunk : (i_chunk+1)*args.N_samples*chunk, :]
                        signal_chunk_i = signal[:, i_chunk*args.N_samples*chunk : (i_chunk+1)*args.N_samples*chunk, :]

                        feat, sigma = [], []
                        feat_torso, sigma_torso = [], []
                        
                        z_shape_i = z_shape[:,itr_obj*2]
                        z_app_i = z_app[:,itr_obj*2]
                        feat_i, sigma_i = decoder(p_chunk_i, r_chunk_i, z_shape_i, z_app_i, signal_chunk_i, 'head')
                        sigma_i = sigma_i.reshape(batch_size, -1, args.N_samples)
                        feat_i = feat_i.reshape(batch_size, -1, args.N_samples, 3)
                        if args.concate_bg:
                            bc_rgb = bc_img.reshape(batch_size, H*W, 1, 3)[:, i_chunk*chunk : (i_chunk+1)*chunk, :, :]
                            feat_i = torch.cat((feat_i[..., :-1, :], bc_rgb), dim=-2)
                        
                        z_shape_i_torso = z_shape[:,itr_obj*2+1]
                        z_app_i_torso = z_app[:,itr_obj*2+1]
                        feat_i_torso, sigma_i_torso = decoder(p_chunk_i_torso, r_chunk_i_torso, z_shape_i_torso, z_app_i_torso, signal_torso, 'torso')
                        sigma_i_torso = sigma_i_torso.reshape(batch_size, -1, args.N_samples)
                        feat_i_torso = feat_i_torso.reshape(batch_size, -1, args.N_samples, 3)
                        if args.concate_bg:
                            sigma_i_torso[:, :, -1] = 0

                        feat_torso.append(feat_i)
                        sigma_torso.append(sigma_i)
                        feat_torso.append(feat_i_torso)
                        sigma_torso.append(sigma_i_torso)
                        feat.append(feat_i)
                        sigma.append(sigma_i)
                        
                        sigma = F.relu(torch.stack(sigma, dim=0))
                        feat = torch.stack(feat, dim=0)
                        sigma_torso = F.relu(torch.stack(sigma_torso, dim=0))
                        feat_torso = torch.stack(feat_torso, dim=0)
                        if args.concate_bg:
                            sigma[-1, :, :, -1] = sigma[-1, :, :, -1] + 1e-6
                            sigma_torso[-1, :, :, -1] = sigma_torso[-1, :, :, -1] + 1e-6
                            
                        # Composite
                        sigma_sum, feat_weighted = composite_function(sigma, feat)
                        sigma_torso_sum, feat_torso_weighted = composite_function(sigma_torso, feat_torso)

                        # Get Volume Weights
                        z_val_chunk_i = z_vals.unsqueeze(0)[:, i_chunk*chunk : (i_chunk+1)*chunk, :]
                        rays_d_torso_chunk_i = rays_d_torso.unsqueeze(0).repeat(batch_size,1,1)[:, i_chunk*chunk : (i_chunk+1)*chunk, :]
                        rays_d_head_chunk_i = rays_d.unsqueeze(0).repeat(batch_size,1,1)[:, i_chunk*chunk : (i_chunk+1)*chunk, :]
                        weights = calc_volume_weights(z_val_chunk_i, rays_d_head_chunk_i, sigma_sum, last_dist=args.last_dist)
                        weights_torso = calc_volume_weights(z_val_chunk_i, rays_d_torso_chunk_i, sigma_torso_sum, last_dist=args.last_dist)
                        feat_map = torch.sum(weights.unsqueeze(-1) * feat_weighted, dim=-2) ## [b, chunk, hidden]
                        rgb_map.append(feat_map)
                        feat_map_torso = torch.sum(weights_torso.unsqueeze(-1) * feat_torso_weighted, dim=-2) ## [b, chunk, hidden]
                        rgb_map_torso.append(feat_map_torso)

                    rgb = torch.cat(rgb_map_torso, dim=1).squeeze(0).reshape(H, W, 3)
                    rgb8 = to8b(rgb.detach().cpu().numpy())
                    rgbs.append(rgb8)
                    
                filename = os.path.join(testsavedir_person, 'test_{}_{}.mp4'.format(i, expname))
                imageio.mimwrite(filename, rgbs, fps=25, quality=8)
                print('Saved test video')


if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
        
                