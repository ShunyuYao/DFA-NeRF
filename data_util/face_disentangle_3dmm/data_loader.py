import torch
import cv2
import numpy as np
import os
import json
import glob
from torch.utils.data import Dataset

def load_dir(path, start, end):
    lmss = []
    for i in range(start, end):
        if os.path.isfile(os.path.join(path, str(i) + '.lms')):
            lms = np.loadtxt(os.path.join(
                path, str(i) + '.lms'), dtype=np.float32)
            lmss.append(lms)
    lmss = np.stack(lmss)
    lmss = torch.as_tensor(lmss).cuda()
    return lmss

def load_id(path):
    with open(path,'r') as f:
        json_data = json.load(f)
        id = json_data['id']
        focal = json_data['focal']
    return id, focal

def load_exp(path):
    exp_list = []
    euler_list = []
    trans_list = []
    for file in glob.glob(path+'*.json'):
        name = os.path.basename(file)
        if name == 'static_params.json':
            continue
        with open(file,'r') as data:
            data_info = json.load(data)
            exp_para = data_info['exp']
            # exp_para = [(x-min(exp_temp))/(max(exp_temp)-min(exp_temp)) for x in exp_temp]
            # euler_para = data_info['euler']
            # trans_para = data_info['trans']
        exp_list.append(exp_para)
        # euler_list.append(euler_para)
        # trans_list.append(trans_para)
    return exp_list # , euler_list, trans_list


class FaceSJTUDataset(Dataset):
    def __init__(self, root_path) -> None:
        super().__init__()
        # print("path: ", os.path.join(root_path, "*", "*.pt"))
        self.datalist = sorted(glob.glob(os.path.join(root_path, "*", "*.pt")))
    
    def __len__(self) -> int:
        return len(self.datalist)
    
    def __getitem__(self, index):
        data = torch.load(self.datalist[index])['exp']
        data_path = os.path.dirname(self.datalist[index])
        return data, data_path


class Face3dmmDataset(Dataset):
    def __init__(self, param_paths):
        super().__init__()
        self.datalist = []
        for param_path in param_paths:
            print("param_path: ", param_path)
            datalist = sorted(glob.glob(os.path.join(param_path,'*.json')))
            for file_path in datalist:
                if "static_param" not in file_path:
                    self.datalist.append(file_path)
        # print("self.datalist: ", self.datalist)
    
    def __len__(self) -> int:
        return len(self.datalist)
    
    def __getitem__(self, index: int):
        with open(self.datalist[index], 'r') as data:
            data_info = json.load(data)
            exp_para = np.array(data_info['exp'], dtype=np.float32)
        return exp_para


def get_image_list(data_root, split):
	filelist = []

	with open('filelists/{}.txt'.format(split)) as f:
		for line in f:
			line = line.strip()
			if ' ' in line: line = line.split()[0]
			filelist.append(os.path.join(data_root, line))

	return filelist


class FaceLSR2Dataset(Dataset):
    def __init__(self, data_root, split, is_train=True):
        super().__init__()

        self.is_train = is_train
        self.all_exps_paths = get_image_list(os.path.join(data_root, 'lrs2_3dmm_params_new'), split)
        exp_params = []
        num_params = []
        all_exps_paths_dup = []
        for exp_path in self.all_exps_paths:
            exp_param = torch.load(os.path.join(exp_path, 'face_params.pt'))['exp']
            exp_params.append(exp_param)
            num_param = exp_param.shape[0]
            all_exps_paths_dup += [exp_path] * num_param
            num_params += [num_param] * num_param
        
        self.num_params = num_params
        self.all_exps_paths_dup = all_exps_paths_dup
        self.exp_params = torch.cat(exp_params, dim=0)
    
    def __len__(self) -> int:
        return self.exp_params.shape[0]
    
    def __getitem__(self, index: int):
        if self.is_train:
            return self.exp_params[index]
        else:
            return self.exp_params[index], self.all_exps_paths_dup[index], self.num_params[index]


class FaceLSR2DatasetTest(Dataset):
    def __init__(self, data_root, split, is_train=True):
        super().__init__()

        self.is_train = is_train
        self.all_exps_paths = get_image_list(os.path.join(data_root, 'lrs2_3dmm_params_new'), split)
        exp_params = []
        num_params = []
        for exp_path in self.all_exps_paths:
            exp_param = torch.load(os.path.join(exp_path, 'face_params.pt'))['exp']
            exp_params.append(exp_param)
            num_param = exp_param.shape[0]
            num_params.append(num_param)
        
        self.num_params = num_params
        self.exp_params = exp_params
    
    def __len__(self) -> int:
        return len(self.exp_params)
    
    def __getitem__(self, index: int):
        if self.is_train:
            return self.exp_params[index]
        else:
            return self.exp_params[index], self.all_exps_paths[index]


class FaceLSR2Exp2KptsDatasetGen(Dataset):
    def __init__(self, data_root, split, dim_o, dim_m, is_train=True):
        super().__init__()

        self.is_train = is_train
        self.all_exps_paths = get_image_list(os.path.join(data_root, 'lrs2_3dmm_params_new'), split)
        exp_params = []
        exp_o_params = []
        exp_m_params = []
        num_params = []
        path_names = []
        for exp_path in self.all_exps_paths:
            path_name = "/".join(exp_path.split("/")[-2:])
            path_names.append(path_name)
            exp_param = torch.load(os.path.join(exp_path, 'face_params.pt'))['exp']
            exp_dis_param = torch.load(os.path.join(exp_path, 'face_params_dis_{}_{}.pt'.format(dim_o, dim_m)))
            exp_o_params.append(exp_dis_param["exp_o"])
            exp_m_params.append(exp_dis_param["exp_m"])
            exp_params.append(exp_param)
            num_param = exp_param.shape[0]
            num_params.append(num_param)
        
        self.num_params = num_params
        self.exp_params = exp_params
        self.exp_o_params = exp_o_params
        self.exp_m_params = exp_m_params
        self.path_names = path_names
    
    def __len__(self) -> int:
        return len(self.exp_params)
    
    def __getitem__(self, index: int):
        if self.is_train:
            return self.exp_params[index], self.exp_o_params[index], self.exp_m_params[index], self.path_names[index]
        else:
            return self.exp_params[index], self.all_exps_paths[index]


class FaceExp2KptsDataset(Dataset):
    def __init__(self, exp_path, kpts_path):
        self.exp = np.load(exp_path)
        self.kpts = np.load(kpts_path)
    
    def __len__(self) -> int:
        return self.exp.shape[0]
    
    def __getitem__(self, index: int):
        return self.exp[index], self.kpts[index]
