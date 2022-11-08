import sys
sys.path.append('../external_API')
import json
import numpy as np
import pickle
import argparse
import os
import neptune.new as neptune
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from module.trainer import naive_trainer
from module.load_module import load_model, load_loss, load_optimizer, load_scheduler
from utility.utils import config, train_module
from utility.earlystop import EarlyStopping

from data.dataset import load_data

from copy import deepcopy
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from utility.distributed import apply_gradient_allreduce, reduce_tensor
import pathlib
import random


def set_random_seed(seed_value, use_cuda=True):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu  vars
    random.seed(seed_value) # Python
    os.environ['PYTHONHASHSEED'] = str(seed_value) # Python hash buildin
    if use_cuda: 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

def setup(rank, world_size, master_port):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = master_port

    # initialize the process group
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


def main(rank, option, resume, save_folder, log, master_port):
    # Basic Options
    train_type = option.result['train']['train_type']

    resume_path = os.path.join(save_folder, 'last_dict.pt')
    total_epoch = option.result['train']['total_epoch']

    # GPU Configuration
    num_gpu = len(option.result['train']['gpu'].split(','))
    multi_gpu = len(option.result['train']['gpu'].split(',')) > 1
    if multi_gpu:
        ddp = option.result['train']['ddp']
    else:
        ddp = False


    scheduler_list = option.result['train']['scheduler']
    batch_size, pin_memory = option.result['train']['batch_size'], option.result['train']['pin_memory']

    # Logger
    if (rank == 0) or (rank == 'cuda'):
        token = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5MTQ3MjY2Yy03YmM4LTRkOGYtOWYxYy0zOTk3MWI0ZDY3M2MifQ=='

        if log:
            mode = 'async'
        else:
            mode = 'debug'

        monitoring_hardware = True
        if resume and option.result['meta']['neptune_id'] is not None:
            run = neptune.init('sunghoshin/%s' %option.result['meta']['project_folder'], api_token=token,
                               capture_stdout=monitoring_hardware,
                               capture_stderr=monitoring_hardware,
                               capture_hardware_metrics=monitoring_hardware,
                               run = option.result['meta']['neptune_id'],
                               mode = mode
                               )
        else:
            run = neptune.init('sunghoshin/%s' %option.result['meta']['project_folder'], api_token=token,
                               capture_stdout=monitoring_hardware,
                               capture_stderr=monitoring_hardware,
                               capture_hardware_metrics=monitoring_hardware,
                               mode = mode
                               )

        # neptune_id = str(run.__dict__['_short_id'])
        # option.result['meta']['neptune_id'] = neptune_id

        exp_name, exp_num = save_folder.split('/')[-2], save_folder.split('/')[-1]
        run['exp_name'] = exp_name
        run['exp_num'] = exp_num

        cfg = option.result
        for key in cfg.keys():
            for key_ in cfg[key].keys():
                cfg_name = 'config/%s/%s' %(key, key_)
                run[cfg_name] = cfg[key][key_]
    else:
        run = None

    # Load Model
    model_list, addon_list = load_model(option)
    
    criterion_list = load_loss(option)
    save_module = train_module(total_epoch, criterion_list, multi_gpu)

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
        
        # Load Epoch
        if save_module.save_dict['save_epoch'] == (int(option.result['train']['total_epoch']) - 1):
            return None

    # Multi-Processing GPUs
    if ddp:
        setup(rank, num_gpu, master_port)
        torch.manual_seed(0)
        torch.cuda.set_device(rank)

        for ix in range(len(model_list)):
            model_list[ix].to(rank)
            model_list[ix] = DDP(model_list[ix], device_ids=[rank])
            model_list[ix] = apply_gradient_allreduce(model_list[ix])

        for ix in range(len(addon_list)):
            addon_list[ix] = addon_list[ix].to(rank)
        
        for ix in range(len(criterion_list)):
            criterion_list[ix].to(rank)

    else:
        for ix in range(len(model_list)):
            if multi_gpu:
                model_list[ix] = nn.DataParallel(model_list[ix]).to(rank)
            else:
                model_list[ix] = model_list[ix].to(rank)
            
        for ix in range(len(addon_list)):
            if multi_gpu:
                addon_list[ix] = nn.DataParallel(addon_list[ix]).to(rank)
            else:
                addon_list[ix] = addon_list[ix].to(rank)

    # Optimizer and Scheduler
    optimizer_list = load_optimizer(option, model_list, addon_list)
    
    if resume:
        # Load Optimizer
        for ix, optimizer in enumerate(optimizer_list):
            optimizer.load_state_dict(save_module.save_dict['optimizer'][ix])

    if scheduler_list is not None:
        scheduler_list = load_scheduler(option, optimizer_list)
        
        if resume:
            for ix, scheduler in enumerate(scheduler_list):
                scheduler.load_state_dict(save_module.save_dict['scheduler'][ix])
            

    # Early Stopping
    early = EarlyStopping(patience=option.result['train']['patience'])

    # Dataset and DataLoader
    tr_dataset = load_data(option, data_type='train')
    val_dataset = load_data(option, data_type='val')

    # Data Loader
    if ddp:
        tr_sampler = torch.utils.data.distributed.DistributedSampler(dataset=tr_dataset,
                                                                     num_replicas=num_gpu, rank=rank)
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset=val_dataset,
                                                                     num_replicas=num_gpu, rank=rank)

        tr_loader = torch.utils.data.DataLoader(dataset=tr_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=option.result['train']['num_workers'], pin_memory=pin_memory,
                                                  sampler=tr_sampler)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=option.result['train']['num_workers'], pin_memory=pin_memory,
                                                  sampler=val_sampler)

    else:
        tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=option.result['train']['num_workers'])
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=option.result['train']['num_workers'])


    # Mixed Precision
    mixed_precision = option.result['train']['mixed_precision']
    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None        

    # Run
    for epoch in range(save_module.init_epoch, save_module.total_epoch):        
        # Scheduler with Warm-up
        save_module.save_dict['scheduler'] = []
        
        if scheduler_list is not None:
            for scheduler in scheduler_list:
                
                if option.result['train']['scheduler'] == 'anealing':
                    scheduler.step(epoch)
                else:
                    scheduler.step()
                    
                save_module.save_dict['scheduler'].append(scheduler.state_dict())

        # Train
        save_module = naive_trainer.train(option, rank, epoch, model_list, addon_list, criterion_list, optimizer_list, multi_gpu, \
                                                                        tr_loader, scaler, save_module, run, save_folder)
        # Evaluation
        result = naive_trainer.validation(option, rank, epoch, model_list, addon_list, criterion_list, multi_gpu, val_loader, scaler, run)

        # Log Learning Rate
        if run is not None:
            for param_group in optimizer_list[0].param_groups:
                run['debug/current_lr'].log(param_group['lr'])
                
        # Save the last-epoch module
        if (rank == 0) or (rank == 'cuda'):
            save_module_path = os.path.join(save_folder, 'last_dict.pt')
            save_module.export_module(save_module_path)

            save_config_path = os.path.join(save_folder, 'last_config.json')
            option.export_config(save_config_path)

        # Early Stopping
        param_list = []
        if multi_gpu:
            for model in model_list:
                param_list.append(deepcopy(model.module.state_dict()))
        else:
            for model in model_list:
                param_list.append(deepcopy(model.state_dict()))

        if option.result['train']['early_loss']:
            early(result['val_loss'], param_list, result)
        else:
            early(-result['acc1'], param_list, result)

        if early.early_stop == True:
            break


    if (rank == 0) or (rank == 'cuda'):
        # Save the best_model
        torch.save(early.model, os.path.join(save_folder, 'best_model.pt'))

    if ddp:
        cleanup()

    return None


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--save_dir', type=str, default='/data/sung/checkpoint/cloud')
    parser.add_argument('--exp_name', type=str, default='im100-merge-init-50cls')
    parser.add_argument('--exp_num', type=int, default=0)
    parser.add_argument('--log', type=lambda x: x.lower()=='true', default=False)
    args = parser.parse_args()

    # Configure
    save_folder = os.path.join(args.save_dir, args.exp_name, str(args.exp_num))
    os.makedirs(save_folder, exist_ok=True)
    option = config(save_folder)
    option.get_config_data()
    option.get_config_network()
    option.get_config_train()
    option.get_config_meta()
    option.get_config_tune()

    option.result['train']['save_folder'] = save_folder


    # Resume Configuration
    resume = option.result['train']['resume']
    resume_path = os.path.join(save_folder, 'last_dict.pt')
    config_path = os.path.join(save_folder, 'last_config.json')

    # BASE FOLDER
    option.result['train']['base_folder'] = str(pathlib.Path(__file__).parent.resolve())

    # Target Class
    if option.result['train']['target_list'] is not None:
        option.result['data']['num_class'] = len(option.result['train']['target_list'])
    
    
    # Resume
    if resume:
        if (os.path.isfile(resume_path) == False) or (os.path.isfile(config_path) == False):
            resume = False
        else:
            gpu = option.result['train']['gpu']

            option = config(save_folder)
            option.import_config(config_path)

            option.result['train']['gpu'] = gpu

    # GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = option.result['train']['gpu']
    num_gpu = len(option.result['train']['gpu'].split(','))
    multi_gpu = num_gpu > 1
    if multi_gpu:
        ddp = option.result['train']['ddp']
    else:
        ddp = False

    master_port = str(random.randint(100,10000))
    
    set_random_seed(option.result['train']['seed'])
        
    if ddp:
        mp.spawn(main, args=(option, resume, save_folder, args.log, master_port, ), nprocs=num_gpu, join=True)
    else:
        main('cuda', option, resume, save_folder, args.log, master_port)