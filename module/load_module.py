import torch
import torch.nn as nn
import json
import os
from utility.utils import config
from utility.warmup_scheduler import GradualWarmupScheduler
from copy import copy, deepcopy
from .radam import RAdam
import numpy as np
from .model import Conv1DNet, Conv2DNet, CRNN_1D, CRNN_2D


# Dummy
class Dummy(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x


class DummyConfig(object):
    def __init__(self, network_type, num_class):
        self.result = {'network': {'network_type': network_type},
                       'data': {'num_class': num_class},
                       'train': {'init_path': None, 'pretrained_imagenet': True, 'add_on': False}}
    def forward(self):
        return self.result

    
class model_manager(object):
    def __init__(self, config_module):
        self.option = config_module
        self.network_type = config_module.result['network']['network_type']

    def load_network(self):
        # Load Pre-trained Models
        if self.option.result['network']['network_type'] == 'conv1d':
            self.model = Conv1DNet()
        elif self.option.result['network']['network_type'] == 'conv2d':
            self.model = Conv2DNet(self.option.result['train']['feature_type'], duration=self.option.result['train']['duration'])
        elif self.option.result['network']['network_type'] == 'crnn_1d':
            self.model = CRNN_1D(hidden_dim=64, num_layers=2)
        elif self.option.result['network']['network_type'] == 'crnn_2d':
            self.model = CRNN_2D(hidden_dim=64, num_layers=2, num_classes=2)
        else:
            raise('Select Proper Network Type')
        
    def load_weight(self, merge_path):
        print('load_pretrained_weights')
        
        in_features = copy(self.get_infeatures())
        out_features = copy(self.get_outfeatures())
        num_class =  self.option.result['data']['num_class']
        
        weight = torch.load(merge_path)['model'][0]
        
        if out_features != num_class:
            self.remove_classifier()
        
            weight_dict = {}
            for name, param in weight.items():
                if 'fc' not in name and 'classifier' not in name:
                    weight_dict[name] = param
        else:
            weight_dict = weight

        self.model.load_state_dict(weight_dict)
        
        if out_features != num_class:
            self.model.fc = nn.Linear(in_features, self.option.result['data']['num_class'])

    def update_classifier(self, in_features):
        self.model.fc = nn.Linear(in_features, self.option.result['data']['num_class'])
        

    def remove_classifier(self):
        if self.network_type in ['resnet18', 'resnet34', 'resnet50']:
            self.model.fc = Dummy()
        elif self.network_type == 'mobilenetv3_small_075':
            self.model.classifier = Dummy()
        else:
            raise ('Select Proper Network Type')
    
    def get_classifier(self):
        if self.network_type in ['resnet18', 'resnet34', 'resnet50']:
            return self.model.fc
        elif self.network_type == 'mobilenetv3_small_075':
            return self.model.classifier
        else:
            raise ('Select Proper Network Type')
    
    def get_infeatures(self):
        if self.network_type in ['resnet18', 'resnet34', 'resnet50']:
            return copy(self.model.fc.in_features)
        elif self.network_type == 'mobilenetv3_small_075':
            return copy(self.model.classifier.in_features)
        else:
            raise ('Select Proper Network Type')
    
    def get_outfeatures(self):
        if self.network_type in ['resnet18', 'resnet34', 'resnet50']:
            return copy(self.model.fc.out_features)
        elif self.network_type == 'mobilenetv3_small_075':
            return copy(self.model.classifier.out_features)
        else:
            raise ('Select Proper Network Type')
        
    
    def init_weight(self):
        print('initialize_weights')
        
        # For ResNet
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                
                
def load_model(option):
    manager = model_manager(option)
    
    train_method = option.result['train']['train_method']
    if train_method == 'base':
        manager.load_network()
                        
        model = manager.model
        model_list = [model]
        addon_list = []

    else:
        raise('Select Proper Train Method')

    return model_list, addon_list

def load_optimizer(option, model_list, addon_list):
    train_method = option.result['train']['train_method']
    weight_decay = option.result['train']['weight_decay']
    
    # Classifier
    param_cls = [p for p in model_list[0].parameters() if p.requires_grad]
    
    # Optimizer    
    if option.result['train']['optimizer'] == 'sgd':
        optim_cls = torch.optim.SGD(param_cls, lr=option.result['train']['lr'], momentum=0.9, weight_decay=weight_decay)
        
    elif option.result['train']['optimizer'] == 'adam':
        optim_cls = torch.optim.Adam(param_cls, lr=option.result['train']['lr'])
    
    elif option.result['train']['optimizer'] == 'radam':
        optim_cls = RAdam(param_cls, lr=option.result['train']['lr'], weight_decay=weight_decay)
                
    else:
        raise('Select Proper Optimizer')
    
    optimizer_list = [optim_cls]
    return optimizer_list

def load_scheduler(option, optimizer_list):
    if option.result['train']['scheduler'] == 'step':
        scheduler_warmup = torch.optim.lr_scheduler.MultiStepLR(optimizer_list[0], [int(option.result['train']['total_epoch']/3), int(option.result['train']['total_epoch']*2/3)])
    elif option.result['train']['scheduler'] == 'anealing':
        scheduler_warmup = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_list[0], T_max=option.result['train']['total_epoch'])
        # scheduler_warmup = GradualWarmupScheduler(optimizer_list[0], multiplier=1, total_epoch=int(option.result['train']['total_epoch'] / 20), after_scheduler=scheduler_cls)
    elif option.result['train']['scheduler'] == 'anealing_warmup':
        scheduler_cls = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_list[0], T_max=option.result['train']['total_epoch'])
        scheduler_warmup = GradualWarmupScheduler(optimizer_list[0], multiplier=1, total_epoch=int(option.result['train']['total_epoch'] / 20), after_scheduler=scheduler_cls)
    else:
        raise('Select Proper Scheduler')
    
    scheduler_list = [scheduler_warmup]
    return scheduler_list

def load_loss(option):
    criterion_cls = nn.CrossEntropyLoss()
    criterion_list = [criterion_cls]
    return criterion_list

