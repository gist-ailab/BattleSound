import subprocess
import numpy as np
from multiprocessing import Process
import os
import torch
import argparse

os.environ['MKL_THREADING_LAYER'] = 'GNU'

# Basic Python Environment
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--exp', type=int, default=-1)
args = parser.parse_args()

lr = 1e-3

# Training
if args.exp == 0:
    start = 0

    feature_type = ['RawSignal', 'Spec', 'MelSpec']
    label_type = ['voice']
    label_folder = ['label_0.5']

    gpus = ['0,1','2,3','4,5','6,7']

    # Combination
    comb_list = []
    ix = 0
    num_per_gpu = 1

    for f_t in feature_type:
        for l_t in label_type:
            for l_f in label_folder:
                comb_list.append([f_t, l_t, l_f, ix])
                ix += 1

else:
    raise('Select Proper Exp Num')

comb_list = comb_list * num_per_gpu
comb_list = [comb + [index] for index, comb in enumerate(comb_list)]

arr = np.array_split(comb_list, len(gpus))
arr_dict = {}

for ix in range(len(gpus)):
    arr_dict[ix] = arr[ix]

def tr_gpu(comb, ix):
    comb = comb[ix]
    for i, comb_ix in enumerate(comb):
        exp_num = start + int(comb_ix[-1])
        gpu = gpus[ix]
        script = 'python main.py --exp_num %d --gpu %s \
                  --feature_type %s --label_type %s --label_folder %s --lr %f' \
                  %(exp_num, gpu, comb_ix[0], comb_ix[1], comb_ix[2], lr)
        subprocess.call(script, shell=True)

for ix in range(len(gpus)):
    exec('thread%d = Process(target=tr_gpu, args=(arr_dict, %d))' %(ix, ix))

for ix in range(len(gpus)):
    exec('thread%d.start()' %ix)