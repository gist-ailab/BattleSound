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
from torchvision.models import resnet18

from module.load_module import load_model, load_loss, load_optimizer, load_scheduler
from utility.utils import config, train_module
from utility.earlystop import EarlyStopping

from data.dataset import load_data, IncrementalSet

from copy import deepcopy
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from utility.distributed import apply_gradient_allreduce, reduce_tensor
import pathlib

import ray
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
from tuning import load_configs
import shutil
from ray.tune.suggest.basic_variant import BasicVariantGenerator

def main(configs, option, log=False):
    for key, item in configs.items():
        option.result[key.split('/')[0]][key.split('/')[1]] = item

    # Basic Options
    train_type = option.result['train']['train_type']
    rank = 'cuda'
    resume = False
    multi_gpu = False

    total_epoch = option.result['train']['total_epoch']

    # Load Model
    model_list, addon_list = load_model(option)
    criterion_list = load_loss(option)
    save_module = train_module(total_epoch, criterion_list, multi_gpu)

    # GPU Configuration
    ddp = option.result['tune']['ddp']
    num_gpu = option.result['tune']['gpus_per_trial']

    if ddp:
        multi_gpu = True
    else:
        device = "cuda"
        if torch.cuda.device_count() > 1:
            multi_gpu = True
            
            for ix in range(len(model_list)):
                model_list[ix] = nn.DataParallel(model_list[ix])
        else:
            multi_gpu = False
        
        for ix in range(len(model_list)):
            model_list[ix].to(device)

    scheduler_list = option.result['train']['scheduler']
    batch_size, pin_memory = option.result['train']['batch_size'], option.result['train']['pin_memory']

    # Logger
    if (rank == 0) or (rank == 'cuda'):
        token = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5MTQ3MjY2Yy03YmM4LTRkOGYtOWYxYy0zOTk3MWI0ZDY3M2MifQ=='

        if log:
            mode = 'async'
        else:
            mode = 'debug'

        hard_ware_monitoring = True

        if resume and option.result['meta']['neptune_id'] is not None:
            run = neptune.init('sunghoshin/%s' %option.result['meta']['project_folder'], api_token=token,
                               capture_stdout=hard_ware_monitoring,
                               capture_stderr=hard_ware_monitoring,
                               capture_hardware_metrics=hard_ware_monitoring,
                               run = option.result['meta']['neptune_id'],
                               mode = mode,
                               tags=['TUNE']
                               )
        else:
            run = neptune.init('sunghoshin/%s' %option.result['meta']['project_folder'], api_token=token,
                               capture_stdout=hard_ware_monitoring,
                               capture_stderr=hard_ware_monitoring,
                               capture_hardware_metrics=hard_ware_monitoring,
                               mode = mode,
                               tags=['TUNE']
                               )

        neptune_id = str(run.__dict__['_short_id'])
        option.result['meta']['neptune_id'] = neptune_id

        exp_name, exp_num = save_folder.split('/')[-2], save_folder.split('/')[-1]
        run['exp_name'] = exp_name
        run['exp_num'] = exp_num

        # Log Basic Option
        cfg = option.result
        for key in cfg.keys():
            for key_ in cfg[key].keys():
                cfg_name = 'config/%s/%s' %(key, key_)
                run[cfg_name] = cfg[key][key_]


        # Log Tunning Params
        for key, item in configs.items():
            tune_name = 'tune/%s' %key.split('/')[1]
            run[tune_name] = item

    else:
        run = None

    optimizer_list = load_optimizer(option, model_list, addon_list)
    
    if scheduler_list is not None:
        scheduler_list = load_scheduler(option, optimizer_list)


    # Early Stopping
    early = EarlyStopping(patience=option.result['train']['patience'])

    # Dataset and DataLoader
    tr_dataset = load_data(option, data_type='train')
    val_dataset = load_data(option, data_type='val')

    target_list = list(range(0, option.result['data']['num_class']))

    tr_dataset = IncrementalSet(tr_dataset, target_list=target_list, shuffle_label=True, prop=option.result['train']['train_prop'])
    val_dataset = IncrementalSet(val_dataset, target_list=target_list, shuffle_label=False, prop=option.result['train']['val_prop'])

    if ddp:
        tr_sampler = torch.utils.data.distributed.DistributedSampler(dataset=tr_dataset,
                                                                     num_replicas=num_gpu, rank=rank)
        val_sampler = torch.utils.data.distributed.DistributedSampler(dataset=val_dataset,
                                                                     num_replicas=num_gpu, rank=rank)

        tr_loader = torch.utils.data.DataLoader(dataset=tr_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=option.result['tune']['cpus_per_trial'], pin_memory=pin_memory,
                                                  sampler=tr_sampler)
        val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=batch_size,
                                                  shuffle=False, num_workers=option.result['tune']['cpus_per_trial'], pin_memory=pin_memory,
                                                  sampler=val_sampler)

    else:
        tr_loader = DataLoader(tr_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, num_workers=option.result['tune']['cpus_per_trial'])
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=pin_memory, num_workers=option.result['tune']['cpus_per_trial'])


    # Mixed Precision
    mixed_precision = option.result['train']['mixed_precision']
    if mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    # Training
    for epoch in range(save_module.init_epoch, save_module.total_epoch):
        if train_type == 'naive':
            from module.trainer import naive_trainer

            # Train
            save_module = naive_trainer.train(option, rank, epoch, model_list, addon_list, criterion_list, optimizer_list, multi_gpu, \
                                              tr_loader, scaler, save_module, run, save_folder)

            # Evaluation
            result = naive_trainer.validation(option, rank, epoch, model_list, addon_list, criterion_list, multi_gpu, val_loader, scaler, run)

        else:
            raise('Select Proper Train-Type')


        # Scheduler
        save_module.save_dict['scheduler'] = []
        
        if scheduler_list is not None:
            for scheduler in scheduler_list:
                scheduler.step()
                save_module.save_dict['scheduler'].append(scheduler.state_dict())


        # Early Stopping
        param = None

        if option.result['train']['early_loss']:
            early(result['val_loss'], param, result)
        else:
            early(-result['acc1'], param, result)

        if early.early_stop == True:
            break

    return None


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--save_dir', type=str, default='/HDD1/sung/checkpoint/')
    parser.add_argument('--exp_name', type=str, default='imagenet_norm')
    parser.add_argument('--exp_num', type=int, default=1)
    parser.add_argument('--log', type=lambda x: x.lower()=='true', default=True)
    args = parser.parse_args()

    # Configure
    save_folder = os.path.join(args.save_dir, args.exp_name, str(args.exp_num))

    if os.path.isdir(os.path.join(save_folder, 'logging')):
        shutil.rmtree(os.path.join(save_folder, 'logging'))

    os.makedirs(save_folder, exist_ok=True)
    option = config(save_folder)
    option.get_config_data()
    option.get_config_network()
    option.get_config_train()
    option.get_config_meta()
    option.get_config_tune()
    
    
    # Option Manipulation
    if option.result['train']['attn_type'] == 'NAIVE':
        option.result['train']['w_attention'] = False
    
    if option.result['train']['w_attention']:
        assert option.result['train']['attn_type'] in ['CBAM', 'SE']
    else:
        option.result['train']['attn_type'] = 'NAIVE'
        

    # GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = option.result['train']['gpu']
    ddp = option.result['train']['ddp']

    # BASE FOLDER
    option.result['train']['base_folder'] = str(pathlib.Path(__file__).parent.resolve())
    option.result['data']['data_dir'] = os.path.join(option.result['data']['data_dir'], option.result['data']['data_type'])

    # Ray Option
    ray.init(log_to_driver=False)

    num_trials = option.result['tune']['num_trials']

    grace_period = int(option.result['train']['total_epoch'] / 3) # Only stop trials at least this old in time.

    tune_scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=option.result['train']['total_epoch'],
        grace_period=grace_period,
        reduction_factor=2)


    # Load Tuning Parameters
    configs = load_configs(args.exp_name)

    # RUN!
    os.environ["TUNE_DISABLE_AUTO_CALLBACK_LOGGERS"] = "1"
    result = tune.run(
        partial(main, option=option, log=args.log),
        resources_per_trial={"cpu": option.result['tune']['cpus_per_trial'], "gpu": option.result['tune']['gpus_per_trial']},
        config=configs,
        num_samples=num_trials,
        scheduler=tune_scheduler,
        local_dir=save_folder,
        name='logging',
        callbacks=[ray.tune.logger.JsonLoggerCallback()])

    # TOTAL Logger
    token = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiI5MTQ3MjY2Yy03YmM4LTRkOGYtOWYxYy0zOTk3MWI0ZDY3M2MifQ=='

    if args.log:
        mode = 'async'
    else:
        mode = 'debug'

    hard_ware_monitoring = True

    run = neptune.init('sunghoshin/%s' %option.result['meta']['project_folder'], api_token=token,
                       capture_stdout=hard_ware_monitoring,
                       capture_stderr=hard_ware_monitoring,
                       capture_hardware_metrics=hard_ware_monitoring,
                       mode=mode,
                       tags=['TOTAL', 'TUNE']
                       )


    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))