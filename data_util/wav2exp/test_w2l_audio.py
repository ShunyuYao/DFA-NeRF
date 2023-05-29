import sys
import torch
from glob import glob
import os, random, cv2
import numpy as np
import torch.nn as nn
from os.path import dirname, join, basename, isfile
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pickle as pkl
import collections
import argparse

from models import AudioEncoder
from hparams import hparams
import audio

syncnet_T = 5
syncnet_mel_step_size = 16

parser = argparse.ArgumentParser()
parser.add_argument('--input_path', type=str,
                    default='obama', help='input audio path')
parser.add_argument('--save_path', type=str,
                    default='obama', help='save path')
args = parser.parse_args()

class AudDataset(object):
    def __init__(self, wavpath):
        wav = audio.load_wav(wavpath, hparams.sample_rate)

        self.orig_mel = audio.melspectrogram(wav).T
        self.data_len = int( (self.orig_mel.shape[0] - syncnet_mel_step_size) / 80. * float(hparams.fps))

    def get_frame_id(self, frame):
        return int(basename(frame).split('.')[0])

    def get_window(self, start_id):

        window_fnames = []
        for frame_id in range(start_id, start_id + syncnet_T):
            window_fnames.append(frame_id)
        return window_fnames

    def read_window(self, window_fnames):
        if window_fnames is None: return None
        window = []
        for fname in window_fnames:
            img = cv2.imread(fname)
            if img is None:
                return None
            try:
                img = cv2.resize(img, (hparams.img_size, hparams.img_size))
            except Exception as e:
                return None

            window.append(img)

        return window

    def crop_audio_window(self, spec, start_frame):
        if type(start_frame) == int:
            start_frame_num = start_frame
        else:
            start_frame_num = self.get_frame_id(start_frame)
        start_idx = int(80. * (start_frame_num / float(hparams.fps)))
        
        end_idx = start_idx + syncnet_mel_step_size

        return spec[start_idx : end_idx, :]

    def get_segmented_mels(self, spec, start_frame):
        mels = []
        assert syncnet_T == 5
        start_frame_num = self.get_frame_id(start_frame) + 1 # 0-indexing ---> 1-indexing
        if start_frame_num - 2 < 0: return None
        for i in range(start_frame_num, start_frame_num + syncnet_T):
            m = self.crop_audio_window(spec, i - 2)
            if m.shape[0] != syncnet_mel_step_size:
                return None
            mels.append(m.T)

        mels = np.asarray(mels)

        return mels

    def prepare_window(self, window):
        # 3 x T x H x W
        x = np.asarray(window) / 255.
        x = np.transpose(x, (3, 0, 1, 2))

        return x

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        window_idxes = self.get_window(idx)

        # window = self.all_exps[window_idxes]

        mel = self.crop_audio_window(self.orig_mel.copy(), idx)
        # print("mel.shape: ", mel.shape)

        if (mel.shape[0] != syncnet_mel_step_size):
            raise Exception('mel.shape[0] != syncnet_mel_step_size')
        # x = window.float()
        # x = torch.FloatTensor(x)
        mel = torch.FloatTensor(mel.T).unsqueeze(0)
        # indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1)

        return mel


device = torch.device('cuda')
model = AudioEncoder()
ckpt = torch.load('checkpoints/audio_encoder.pth')
new_state_dict = collections.OrderedDict()
for key, value in ckpt.items():
   new_state_dict['audio_encoder.' + key] = value
model.load_state_dict(new_state_dict)
model = model.to(device).eval()

dataset = AudDataset(args.input_path)
save_path = args.save_path
data_loader = DataLoader(dataset, batch_size=64, shuffle=False)

outputs = []
with torch.no_grad():
    for mel in data_loader:
        # x, mel = data[i]
        mel = mel.to(device) # .unsqueeze(0)
        print("mel.shape: ", mel.shape)
        out = model(mel)
        outputs.append(out)
outputs = torch.cat(outputs, dim=0)
outputs = outputs.cpu()
print("outputs.shape: ", outputs.shape)
torch.save(outputs, save_path)