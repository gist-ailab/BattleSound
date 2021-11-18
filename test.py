import numpy as np
import torch.nn as nn
import timm
import os
import pandas as pd
import sys
sys.path.append('../utility')
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import default_loader
import torch
from scipy import io as mat_io
from glob import glob
from torchvision.transforms import transforms
from utility.utils import config, train_module
from module.load_module import load_model, load_loss, load_optimizer, load_scheduler

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class CUB(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.root = root
        self.transform = transform
        self.loader = default_loader
        self.train = train
                
        self._load_metadata()

        self.targets = [self.data.iloc[idx].target-1 for idx in range(len(self.data))]


    def _load_metadata(self):
        images = pd.read_csv(os.path.join(self.root, 'images.txt'), sep=' ',
                            names=['img_id', 'filepath'])
        image_class_labels = pd.read_csv(os.path.join(self.root, 'image_class_labels.txt'),
                                        sep=' ', names=['img_id', 'target'])
        train_test_split = pd.read_csv(os.path.join(self.root, 'train_test_split.txt'),
                                    sep=' ', names=['img_id', 'is_training_img'])

        data = images.merge(image_class_labels, on='img_id')
        self.data = data.merge(train_test_split, on='img_id')

        if self.train:
            self.data = self.data[self.data.is_training_img == 1]
        else:
            self.data = self.data[self.data.is_training_img == 0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        path = os.path.join(self.root, 'images', sample.filepath)
        target = sample.target - 1  # Targets start at 1 by default, so shift to 0
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


def load_cub(root):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    
    tr_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.Scale(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    tr_dataset = CUB(root, train=True, transform=tr_transform)
    val_dataset = CUB(root, train=False, transform=val_transform)
    return tr_dataset, val_dataset

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class hook_manager(object):
    def __init__(self):
        pass

    def register_target_hook(self):
        pass
    
    def remove_hook(self):
        pass
    
    

if __name__=='__main__':
    # Option
    root = '/data/sung/dataset/cub'
    num_class = 200
    device = 'cuda:2'
    
    lr = 0.01
    weight_decay = 1e-4
    epoch = 100
    batch_size = 256
    
    save_folder = '/data/sung/checkpoint//cloud/imagenet100-selection/0'
    config_path = os.path.join(save_folder, 'last_config.json')
    resume_path = os.path.join(save_folder, 'last_dict.pt')
    
    # Data
    tr_dataset, val_dataset = load_cub(root)
    tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    
    # Checkpoint
    option = config(save_folder)
    option.import_config(config_path)

    save_module = train_module(100, [], False)
    save_module.import_module(resume_path)
        
    # Model 
    model_list, _ = load_model(option)
    model = model_list[0]
    model.load_state_dict(save_module.save_dict['model'][0])
    
    # to GPU
    model = model.to(device)

    # Optimizer
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    # optim = RAdam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Run
    for ix in range(epoch):
        # Train
        model.train()
        
        tr_loss = 0.
        tr_acc = 0.
        for image, label in tr_loader:            
            image, label = image.to(device), label.to(device)
            
            optim.zero_grad()
            output = model(image)
            tr_loss_ix = nn.CrossEntropyLoss()(output, label)
            tr_loss_ix.backward()  
            optim.step()
            
            tr_loss += tr_loss_ix.item()
            tr_acc += accuracy(output, label)[0]
            
        tr_loss /= len(tr_loader)
        tr_acc /= len(tr_loader)
        print('Epoch--%d/%d-tr_loss:%.2f, tr_acc:%.2f' %(ix, epoch, tr_loss, tr_acc))

        
        # Validation
        model.eval()
        
        val_loss = 0.
        val_acc = 0.
        for image, label in val_loader:            
            image, label = image.to(device), label.to(device)
            
            with torch.no_grad():
                output = model(image)
            val_loss_ix = nn.CrossEntropyLoss()(output, label)
            
            val_loss += val_loss_ix.item()
            val_acc += accuracy(output, label)[0]
            
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        print('Epoch--%d/%d-val_loss:%.2f, val_acc:%.2f' %(ix, epoch, val_loss, val_acc))