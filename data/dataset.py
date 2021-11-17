import os
from glob import glob
import torch
import torch.nn as nn
import torchaudio.transforms as transforms
import torchaudio.transforms
import argparse
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.autograd import Variable
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import scipy.io.wavfile
import pandas as pd

## DataSet definition
class SoundLoader(Dataset):
    def __init__(self, args, mode='train'):
        # Argparse
        self.args = args

        # Positive and Negative Sample
        self.mode = mode
        if mode == 'train':
            self.label_list = dict(np.load(os.path.join(args.base, args.label_folder, 'meta_dict_%s.npz'%args.label_type)))
            self.positive_list = self.label_list['1'].tolist()

            self.num_0 = len(self.label_list['0'])
            self.num_1 = len(self.label_list['1'])

            if int(self.num_1 / 2) > self.num_0:
                self.negative_list = self.label_list['0'].tolist() + \
                                     self.label_list['-1'][np.random.choice(range(len(self.label_list['-1'].tolist())), self.num_1 - self.num_0, replace=False).tolist()].tolist()
            else:
                self.negative_list = self.label_list['0'][np.random.choice(range(len(self.label_list['0'].tolist())), int(self.num_1 * 0.8), replace=False).tolist()].tolist() + \
                                     self.label_list['-1'][np.random.choice(range(len(self.label_list['-1'].tolist())), int(self.num_1 * 0.1), replace=False).tolist()].tolist()

            self.file_index = self.positive_list + self.negative_list
            self.label_list = [1] * len(self.positive_list) + [0] * len(self.negative_list)

        else:
            self.label_list = dict(np.load(os.path.join(args.base, 'val_0.5', 'meta_dict_%s_pre.npz'%args.label_type)))

            self.positive_list = self.label_list['1'].tolist()
            self.negative_list = self.label_list['0'].tolist() + \
                                 self.label_list['-1'].tolist()

            self.file_index = self.positive_list + self.negative_list
            self.label_list = [1] * len(self.positive_list) + [0] * len(self.negative_list)

        # Pre-processing
        self.transform = True

        self.feature_type = args.feature_type
        self.window_func = args.window_func

        self.func_dict = {
            'bartlett': torch.bartlett_window,
            'blackman': torch.blackman_window,
            'hamming': torch.hamming_window,
            'hann': torch.hann_window
        }

        if args.feature_type == 'Spec':
            self.transform_func = torchaudio.transforms.Spectrogram(
                n_fft=512,
                win_length=400,
                hop_length=200,
                window_fn=self.func_dict[self.window_func],
                normalized=True,
                power=2
            )

        elif args.feature_type == 'MelSpec':
            self.transform_func = torchaudio.transforms.Spectrogram(
                n_fft=512,
                win_length=400,
                hop_length=200,
                window_fn=self.func_dict[self.window_func],
                normalized=True,
                power=2
            )

            self.melscale = torchaudio.transforms.MelScale(
                n_mels=40,
                f_max=8000,
                f_min =300
            )

        elif args.feature_type == 'MFCC':
            self.transform_func = torchaudio.transforms.MFCC(
                n_mfcc=40,
                log_mels=True
            )

        elif args.feature_type == 'RawSignal':
            self.transform_func = None
            self.transform = False
        else:
            raise ValueError

    def __getitem__(self, index):
        self.file_index = self.positive_list + self.negative_list
        self.label_list = [1] * len(self.positive_list) + [0] * len(self.negative_list)

        name, ix = self.file_index[index]
        if self.mode == 'train':
            file = dict(np.load(os.path.join(self.args.base, self.args.label_folder, self.args.label_type, name)))
        else:
            file = dict(np.load(os.path.join(self.args.base, 'val_0.5', self.args.label_type, name)))

        signal, label = file['audio'][int(ix)], int(file['label'][int(ix)])

        x_data = signal / 30000
        x_data = torch.Tensor(x_data)
        x_data = x_data.unsqueeze(dim=0)

        # Transform
        if self.transform:
            self.transform_func = self.transform_func.cuda()
            x_data = self.transform_func(x_data.cuda())
            x_data = x_data.cpu().detach()
            if self.feature_type == 'MelSpec':
                x_data = self.melscale(x_data)
                x_data = torch.log(x_data + 1e-4)

        # Load y data if the mode is not a test!
        if label != 1:
            label = 0

        y_data = torch.Tensor([label]).long()
        y_data = y_data.item()
        return x_data, y_data

    def __len__(self):
        return len(self.file_index)

if __name__=='__main__':
    # Save and Resume Options
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Label Option
    parser.add_argument('--base', type=str, default='/data_2/sung/dataset/dongwoon')
    parser.add_argument('--label_type', type=str, default='voice')
    parser.add_argument('--label_folder', type=str, default='label_0.5')

    # Pre-processing
    parser.add_argument('--window_func', type=str, default='hann')
    parser.add_argument('--feature_type', type=str, default='Spec')
    args = parser.parse_args()

    data = SoundLoader(args, mode='val')
    data.__getitem__(1)