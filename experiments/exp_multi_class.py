import os
import pathlib

from torch import set_default_dtype
base_folder = str(pathlib.Path(__file__).parent.resolve())
os.chdir(base_folder)

import numpy as np
import json
import subprocess
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

    project_folder = 'BattleSound'
    resume = False
    mixed_precision = True

    ddp = False
    log = True
    
    batch_size = 128

    # Setup Configuration for Each Experiments
    
    # Base
    if args.exp == 0:
        server = 'toast'
        save_dir = '/data/sung/checkpoint/battlesound/revision2'
        data_dir = '/data/sung/dataset/dongwoon'

        exp_name = 'sensors_new_multi_class'
        start = 0
        ix = 0
        comb_list = []
        epoch = 50


        train_prop = 1.
        val_prop = 1. 
        
        batch_size = 256
        mixed_precision = False
        ddp = False
        
        num_per_gpu = 1
        
        gpus = ['0', '1', '2', '3']
        
        # Selection
        feature_list = [('melspec', 'conv2d'), ('melspec', 'crnn_2d'), ('raw_signal', 'conv1d'), ('raw_signal', 'crnn_1d')]
        duration_list = [0.5]
        seed_list = [0, 1, 2, 3, 4]
        
        for duration in duration_list:
            for feature_ind in feature_list:
                for seed in seed_list:
                    feature = feature_ind[0]
                    network = feature_ind[1]

                    comb_list.append({'train': 
                                            {
                                                'train_method': 'base',
                                                'lr': 1e-3,
                                                'scheduler': 'anealing',
                                                'feature_type': feature,
                                                'seed': seed,
                                                'pretrained_imagenet': True,
                                                'duration': duration,
                                                'multi_class': True,
                                            },
                                    'network': 
                                            {
                                                'network_type': network
                                            },
                                    'data':
                                            {   'data_type': 'multi',
                                                'num_class': 3
                                            },
                                    'index': ix
                                    })
                    ix += 1

    
    else:
        raise('Select Proper Experiment Number')

    arr = np.array_split(comb_list, len(gpus))
    arr_dict = {}
    for ix in range(len(gpus)):
        arr_dict[ix] = arr[ix]

    def tr_gpu(comb, ix):
        comb = comb[ix]
        
        global json_data
        global json_network
        global json_train
        global json_meta
        global json_tune
        
        for i, comb_ix in enumerate(comb):
            exp_num = start + int(comb_ix['index'])
            os.makedirs(os.path.join(save_dir, exp_name, str(exp_num)), exist_ok=True)

            gpu = gpus[ix]

            ## 1. Common Options
            # Modify the data configuration
            json_data['data_dir'] = data_dir

            # Modify the train configuration
            json_train['gpu'] = str(gpu)

            json_train['total_epoch'] = epoch
            json_train['batch_size'] = batch_size

            json_train["mixed_precision"] = mixed_precision

            json_train["resume"] = resume

            json_train["train_prop"] = train_prop
            json_train["val_prop"] = val_prop

            json_train["ddp"] = ddp

            # Modify the meta configuration
            json_meta['server'] = str(server)
            json_meta['save_dir'] = str(save_dir)
            json_meta['project_folder'] = project_folder
           
            ## 2. Conditional Options
            for key in comb_ix.keys():
                if key == 'train':
                    module = json_train
                elif key == 'data':
                    module = json_data
                elif key == 'network':
                    module = json_network
                elif key == 'meta':
                    module = json_meta
                elif key == 'index':
                    continue
                else:
                    raise('Select Proper Configure Types')
                
                for key_ in comb_ix[key].keys():
                    module[key_] = comb_ix[key][key_]

                if key == 'train':
                    json_train = module
                elif key == 'data':
                    json_data = module
                elif key == 'network':
                    json_network = module
                elif key == 'meta':
                    json_meta = module
                else:
                    raise('Select Proper Configure Types')
            
                module = None
            
            # Save the Configure
            save_json(json_data, os.path.join(save_dir, exp_name, str(exp_num), 'data.json'))
            save_json(json_network, os.path.join(save_dir, exp_name, str(exp_num), 'network.json'))
            save_json(json_train, os.path.join(save_dir, exp_name, str(exp_num), 'train.json'))
            save_json(json_meta, os.path.join(save_dir, exp_name, str(exp_num), 'meta.json'))
            save_json(json_tune, os.path.join(save_dir, exp_name, str(exp_num), 'tune.json'))
                
            # Run !
            script = 'python ../main.py --save_dir %s --exp_name %s --exp_num %d --log %s' %(save_dir, exp_name, exp_num, log)
            subprocess.call(script, shell=True)


    for ix in range(len(gpus)):
        exec('thread%d = Process(target=tr_gpu, args=(arr_dict, %d))' % (ix, ix))

    for ix in range(len(gpus)):
        exec('thread%d.start()' % ix)