import os
import pathlib
base_folder = str(pathlib.Path(__file__).parent.resolve())
os.chdir(base_folder)

import numpy as np
import json
import subprocess
import os
from multiprocessing import Process
import argparse
import warnings
warnings.filterwarnings('ignore')

def load_json(json_path):
    with open(json_path, 'r') as f:
        out = json.load(f)
    return out


def save_json(json_data, json_path):
    with open(json_path, 'w') as f:
        json.dump(json_data, f)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=int, default=0)
    args = parser.parse_args()

    # Data Configuration
    json_data_path = '../config/base_data.json'
    json_data = load_json(json_data_path)

    # Network Configuration
    json_network_path = '../config/base_network.json'
    json_network = load_json(json_network_path)

    # Train Configuration
    json_train_path = '../config/base_train.json'
    json_train = load_json(json_train_path)

    # Meta Configuration
    json_meta_path = '../config/base_meta.json'
    json_meta = load_json(json_meta_path)

    # Meta Configuration
    json_tune_path = '../config/base_tune.json'
    json_tune = load_json(json_tune_path)

    # Global Option
    train_prop = 1.
    val_prop = 1.

    mixed_precision = True

    ddp = False

    tuning = True
    project_folder = 'revisit'

    cpus_per_trial = 1
    gpus_per_trial = 1
    
    # Setup Configuration for Each Experiments
    if args.exp == 0:
        server = 'nipa'
        save_dir = '/home/sung/tuning'
        data_dir = '/home/sung/dataset'
        data_type_and_num = ('imagenet', 1000)

        exp_name = 'tune_1'
        exp_num = 0
        gpu = '0,1,2'

        w_d = 1e-4
        lr_cls = 0.1
        lr_attn = 0.1
        epoch = 100

        train_prop = 0.1
        val_prop = 1.

        batch_size = 256
        mixed_precision = True
        ddp = False

        attn_type = 'CBAM'
        w_attention = True
        
        depth = 18

        num_trials = 20
        cpus_per_trial = 2
        gpus_per_trial = 0.3

    else:
        raise('Select Proper Experiment Number')

    # Tuning
    os.makedirs(os.path.join(save_dir, exp_name, str(exp_num)), exist_ok=True)

    # Modify the data configuration
    json_data['data_dir'] = data_dir
    json_data['data_type'] = data_type_and_num[0]
    json_data['num_class'] = data_type_and_num[1]
    save_json(json_data, os.path.join(save_dir, exp_name, str(exp_num), 'data.json'))

    # Modify the network configuration
    json_network['network_type'] = 'resnet%d' %int(depth)
    save_json(json_network, os.path.join(save_dir, exp_name, str(exp_num), 'network.json'))

    # Modify the train configuration
    json_train['gpu'] = str(gpu)

    json_train['lr_cls'] = lr_cls
    json_train['lr_attn'] = lr_attn
    
    json_train['weight_decay'] = w_d

    json_train['total_epoch'] = epoch
    json_train['batch_size'] = batch_size

    json_train["mixed_precision"] = mixed_precision

    json_train["train_prop"] = train_prop
    json_train["val_prop"] = val_prop

    json_train["ddp"] = ddp
    
    json_train["attn_type"] = attn_type
    json_train["w_attention"] = w_attention
    save_json(json_train, os.path.join(save_dir, exp_name, str(exp_num), 'train.json'))

    # Modify the meta configuration
    json_meta['server'] = str(server)
    json_meta['save_dir'] = str(save_dir)
    json_meta['project_folder'] = project_folder
    save_json(json_meta, os.path.join(save_dir, exp_name, str(exp_num), 'meta.json'))


    # Modify the tune configuration
    json_tune['ddp'] = ddp
    json_tune['tuning'] = tuning
    json_tune['num_trials'] = num_trials
    json_tune['cpus_per_trial'] = cpus_per_trial
    json_tune['gpus_per_trial'] = gpus_per_trial
    save_json(json_tune, os.path.join(save_dir, exp_name, str(exp_num), 'tune.json'))


    # Run !
    script = 'python ../hyper_tuning.py --save_dir %s --exp_name %s --exp_num %d' %(save_dir, exp_name, exp_num)
    subprocess.call(script, shell=True)
