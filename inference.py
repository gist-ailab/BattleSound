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
from visualize.gradcam import GradCAM


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
    # GPU Configuration
    num_gpu = len(option.result['train']['gpu'].split(','))
    multi_gpu = len(option.result['train']['gpu'].split(',')) > 1
    ddp = False
    option.result['train']['multi_class'] = False

    batch_size, pin_memory = option.result['train']['batch_size'], option.result['train']['pin_memory']

    run = None

    # Load Model
    model_list, addon_list = load_model(option)
    
    total_epoch = 100
    criterion_list = load_loss(option)
    save_module = train_module(total_epoch, criterion_list, multi_gpu)

    # Load Model
    save_module.import_module(resume_path)
    for ix, model in enumerate(model_list):
        model.load_state_dict(save_module.save_dict['model'][ix])

    for ix in range(len(model_list)):
        if multi_gpu:
            model_list[ix] = nn.DataParallel(model_list[ix]).to(rank)
        else:
            model_list[ix] = model_list[ix].to(rank)
            
    # Dataset and DataLoader
    val_dataset = load_data(option, data_type='val')

    # Data Loader
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=option.result['train']['num_workers'])

    # Mixed Precision
    scaler = None        

    # Evaluation
    epoch = 0
    result = naive_trainer.validation(option, rank, epoch, model_list, addon_list, criterion_list, multi_gpu, val_loader, scaler, run, confusion=True)
    return result


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--save_dir', type=str, default='/data/sung/checkpoint/battlesound/main')
    parser.add_argument('--exp_name', type=str, default='sensors_new')
    parser.add_argument('--exp_num', type=int, default=0)
    parser.add_argument('--gpu', type=str, default='1')
    args = parser.parse_args()

    # Configure
    save_folder = os.path.join(args.save_dir, args.exp_name, str(args.exp_num))
    os.makedirs(save_folder, exist_ok=True)
    
    option = config(save_folder)
    config_path = os.path.join(save_folder, 'last_config.json')
    option.import_config(config_path)
    
    # Resume Configuration
    resume_path = os.path.join(save_folder, 'last_dict.pt')

    # BASE FOLDER
    option.result['train']['base_folder'] = str(pathlib.Path(__file__).parent.resolve())

    # Target Class
    if option.result['train']['target_list'] is not None:
        option.result['data']['num_class'] = len(option.result['train']['target_list'])
    
    # Resume
    option.result['train']['gpu'] = args.gpu

    # GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = option.result['train']['gpu']
    num_gpu = len(option.result['train']['gpu'].split(','))
    assert num_gpu == 1
    multi_gpu = False
    ddp = False
    resume=False

    master_port = str(random.randint(100,10000))
    
    set_random_seed(option.result['train']['seed'])
    result = main('cuda', option, resume, save_folder, False, master_port)
    
    # with open('result_ablation.txt', 'a') as f:
    #     f.write('%d_%.4f\n' %(args.exp_num, result['acc1']))