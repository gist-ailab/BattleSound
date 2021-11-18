import json
import numpy as np
import pickle
import argparse
import os
import neptune

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18

from module.load_module import load_model, load_loss, load_optimizer, load_scheduler
from utility.utils import config, train_module
from utility.earlystop import EarlyStopping

from data.dataset import load_data

from copy import deepcopy
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from utility.distributed import apply_gradient_allreduce, reduce_tensor

from visualize.plot import attention_manager, cam_manager

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def main(rank, option, resume, save_folder):
    # Basic Options
    resume_path = os.path.join(save_folder, 'last_dict.pt')

    num_gpu = len(option.result['train']['gpu'].split(','))

    multi_gpu = len(option.result['train']['gpu'].split(',')) > 1
    if multi_gpu:
        ddp = option.result['train']['ddp']
    else:
        ddp = False

    batch_size = option.result['train']['batch_size']

    # # Logger
    # if (rank == 0) or (rank == 'cuda'):
    #     token = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5MTQ3MjY2Yy03YmM4LTRkOGYtOWYxYy0zOTk3MWI0ZDY3M2MifQ=='
    #     monitoring_hardware = True
    #     mode = 'async'
        
    #     run = neptune.init('sunghoshin/%s' %option.result['meta']['project_folder'], api_token=token,
    #                         capture_stdout=monitoring_hardware,
    #                         capture_stderr=monitoring_hardware,
    #                         capture_hardware_metrics=monitoring_hardware,
    #                         mode = mode
    #                         )
        
    #     run['exp_name'] = 'inference_%s' %str(args.exp_name)
    #     run['exp_num'] = args.exp_num
    # else:
    #     run = None
        
    # Load Model
    model_list, addon_list = load_model(option)
    criterion_list = load_loss(option)
    save_module = train_module(200, criterion_list, multi_gpu)

    if resume:
        save_module.import_module(resume_path)
        
        # Load Model
        for ix, model in enumerate(model_list):
            model.load_state_dict(save_module.save_dict['model'][ix])

        # Load Add-on
        for ix, addon in enumerate(addon_list):
            if ix == 1:
                for id, (_, _, _, attn) in enumerate(addon):
                    attn.load_state_dict(save_module.save_dict['addon'][ix][id])
        
    # Multi-Processing GPUs
    if ddp:
        setup(rank, num_gpu)
        torch.manual_seed(0)
        torch.cuda.set_device(rank)

        for ix in range(len(model_list)):
            model_list[ix].to(rank)
            model_list[ix] = DDP(model_list[ix], device_ids=[rank])
            model_list[ix] = apply_gradient_allreduce(model_list[ix])

        for ix in range(len(criterion_list)):
            criterion_list[ix].to(rank)

    else:
        for ix in range(len(model_list)):
            if multi_gpu:
                model_list[ix] = nn.DataParallel(model_list[ix]).to(rank)
            else:
                model_list[ix] = model_list[ix].to(rank)

    # Dataset and DataLoader
    val_dataset = load_data(option, data_type='val')

    if ddp:
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset=val_dataset,
                                                                     num_replicas=num_gpu, rank=rank)

        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=4*num_gpu, pin_memory=True,
                                                  sampler=val_sampler)
    else:
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4*num_gpu)


    # Training
    model = model_list[0]
    
    manager = cam_manager(model_list[0], target='layer4', multi_gpu=multi_gpu)
    for x, y in val_loader:
        x = x.to(rank)
        features, gradients = manager.get_gradcam(x)
        break
    
    manager.remove_handler()

    if ddp:
        cleanup()


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--save_dir', type=str, default='/SSD1/sung/checkpoint/merge')
    parser.add_argument('--data_dir', type=str, default='/SSD1/sung/dataset')
    parser.add_argument('--exp_name', type=str, default='init_cifar100')
    parser.add_argument('--exp_num', type=int, default=5)

    parser.add_argument('--gpu', type=str, default='4')
    parser.add_argument('--data_type', type=str, default='cifar100')
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--ddp', type=lambda x: x.lower()=='true', default=False)
    
    args = parser.parse_args()


    # Configure
    resume = True

    save_folder = os.path.join(args.save_dir, args.exp_name, str(args.exp_num))
    os.makedirs(save_folder, exist_ok=True)
    option = config(save_folder)


    # Resume Configuration
    config_path = os.path.join(save_folder, 'last_config.json')
    option.import_config(config_path)

    option.result['train']['gpu'] = args.gpu
    option.result['train']['ddp'] = args.ddp
    option.result['train']['batch_size'] = args.batch
    option.result['data_type'] = args.data_type

    # Data Directory
    option.result['data']['data_dir'] = os.path.join(args.data_dir, option.result['data']['data_type'])

    # GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = option.result['train']['gpu']
    num_gpu = len(option.result['train']['gpu'].split(','))
    multi_gpu = num_gpu > 1
    if multi_gpu:
        ddp = option.result['train']['ddp']
    else:
        ddp = False

    if ddp:
        mp.spawn(main, args=(option,resume,save_folder,), nprocs=num_gpu, join=True)
    else:
        main('cuda', option, resume, save_folder)

