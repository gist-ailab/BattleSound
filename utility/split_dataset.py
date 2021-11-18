import os
import numpy as np
import random
import subprocess
from glob import glob
from collections import defaultdict

random.seed(10)
np.random.seed(10)

# Options
data_dir = '/data/sung/dataset/sketches'
ratio = 0.7

# Category List Split
category_list = os.listdir(data_dir)

train_dict = {}
val_dict = {} 

for category in category_list:
    image_list = glob(os.path.join(data_dir, category, '*.png'))
    
    train_list = np.random.choice(image_list, int(len(image_list) * ratio), replace=False).tolist()
    train_dict[category] = train_list
    val_dict[category] = list(set(image_list) - set(train_list))

for category in category_list:
    # Make Folder    
    os.makedirs(os.path.join(data_dir, 'trainset', category), exist_ok=True)
    os.makedirs(os.path.join(data_dir, 'valset', category), exist_ok=True)
    
    # Move
    for old_path in train_dict[category]:
        new_path = os.path.join(data_dir, 'trainset', category)
        
        script = 'cp -r %s %s' %(old_path, new_path)
        subprocess.call(script, shell=True)
    
    for old_path in val_dict[category]:
        new_path = os.path.join(data_dir, 'valset', category)
    
        script = 'cp -r %s %s' %(old_path, new_path)
        subprocess.call(script, shell=True)