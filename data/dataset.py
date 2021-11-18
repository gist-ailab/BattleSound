from torchvision.transforms import transforms
import torchvision
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
from random import shuffle
import torch
import random
from data.custom_dataset import SoundLoader


def load_battle(option):
    tr_dataset = SoundLoader(option, mode='train')
    val_dataset = SoundLoader(option, mode='val')
    return tr_dataset, val_dataset


def load_data(option, data_type='train'):
    tr_d, val_d = load_battle(option)

    if data_type == 'train':
        return tr_d
    else:
        return val_d
