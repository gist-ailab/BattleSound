from torchvision.transforms import transforms
import torchvision
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from random import shuffle
import torch
import random
from data.custom_dataset import SoundLoader, MultiSoundLoader


def load_battle(option):
    if option.result['train']['multi_class']:
        tr_dataset = MultiSoundLoader(option, mode='train')
        val_dataset = MultiSoundLoader(option, mode='val')
    else:
        tr_dataset = SoundLoader(option, mode='train')
        val_dataset = SoundLoader(option, mode='val')
    return tr_dataset, val_dataset


def load_data(option, data_type='train'):
    tr_d, val_d = load_battle(option)

    if data_type == 'train':
        return tr_d
    else:
        return val_d