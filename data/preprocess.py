from glob import glob
import numpy as np
import random
import os
from glob import glob
from tqdm import tqdm

def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)

if __name__=='__main__':    
    seed = 1 
    set_seed(seed)
    
    # Choice
    # train=True
    all_classes = False
    longer_resolution = False
    crop_data = True
    
    
    # Classification using all classes (during revision)
    if all_classes:
        for train in [True, False]:
            if train:
                data_dir='/data/sung/dataset/dongwoon/train'
                save_path = '/data/sung/dataset/dongwoon/multi_class/train_meta_dict.npz'
                
            else:
                data_dir='/data/sung/dataset/dongwoon/val'
                save_path = '/data/sung/dataset/dongwoon/multi_class/val_meta_dict.npz'
            
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            sample_list = glob(os.path.join(data_dir, '*.npz'))
            class_dict = {'0':[], '1':[], '2':[], '3':[]}
            for sample_path in tqdm(sample_list):
                sample = dict(**np.load(sample_path))
                
                for ix, label in enumerate(sample['label']):
                    class_dict[str(label)].append([os.path.basename(sample_path), str(ix)])
            
            for key in class_dict.keys():
                class_dict[key] = np.array(class_dict[key])
            
            others_sampled_index = np.random.choice(range(len(class_dict['3'])), min(len(class_dict['0']), len(class_dict['1']), len(class_dict['2'])), replace=False)
            class_dict['3'] = class_dict['3'][others_sampled_index]
            
            np.savez(save_path, **class_dict)
        
    
    
    if longer_resolution:
        # Longer Resolution
        for duration in [8.0, 4.0, 2.0]:
            for data_type in ['event', 'voice']:
                meta_path = '/data/sung/dataset/dongwoon/label_%.1f/meta_dict_%s.npz' %(duration, data_type)
                meta_dict = {'0': [], '1': [], '-1': []}
                
                file_list = glob('/data/sung/dataset/dongwoon/label_%.1f/%s/*.npz' %(duration,data_type))
                for file_path in tqdm(file_list):
                    file_name = os.path.basename(file_path)
                    file = dict(np.load(file_path))
                    
                    chunk_size = int(duration / 0.5)
                    len = file['label'].shape[0] // chunk_size
                    for ix in range(len):
                        label_ix = str(file['label'][ix * chunk_size])
                        meta_dict[label_ix].append([file_name, str(ix * chunk_size)])
                
                meta_dict['0'] = np.array(meta_dict['0'])
                meta_dict['1'] = np.array(meta_dict['1'])
                meta_dict['-1'] = np.array(meta_dict['-1'])
                np.savez(meta_path, **meta_dict)
                
                
    if crop_data:
        data_dir = '/SSDb/sung/dataset/dongwoon'
        tr_dict = dict(np.load(os.path.join(data_dir, 'multi_class', 'train_meta_dict.npz')))
        val_dict = dict(np.load(os.path.join(data_dir, 'multi_class', 'val_meta_dict.npz')))
        
        tr_list = np.concatenate([tr_dict['0'][:,0], tr_dict['1'][:,0], tr_dict['2'][:,0]], axis=0)
        tr_list = np.unique(tr_list)

        val_list = np.concatenate([val_dict['0'][:,0], val_dict['1'][:,0], val_dict['2'][:,0]], axis=0)
        val_list = np.unique(val_list)
        
        # Crop Train Dataset
        save_dir = os.path.join(data_dir, 'publication', 'train')
        for tr_name in tqdm(tr_list):
            file_name = os.path.join(data_dir, 'train', tr_name)
            file = dict(np.load(file_name))
            
            for ix, file_ix in enumerate(file['label']):
                audio = file['audio'][(8000 * ix):(8000 * (ix+1))]
                save_name = os.path.join(save_dir, str(file_ix), '%s_id%04d.npy' %(tr_name.rstrip('.npz'), ix))
                os.makedirs(os.path.dirname(save_name), exist_ok=True)
                np.save(save_name, audio)
                
        
        # Crop Valid Dataset
        save_dir = os.path.join(data_dir, 'publication', 'val')
        for val_name in tqdm(val_list):
            file_name = os.path.join(data_dir, 'val', val_name)
            file = dict(np.load(file_name))
            
            for ix, file_ix in enumerate(file['label']):
                audio = file['audio'][(8000 * ix):(8000 * (ix+1))]
                save_name = os.path.join(save_dir, str(file_ix), '%s_id%04d.npy' %(val_name.rstrip('.npz'), ix))
                os.makedirs(os.path.dirname(save_name), exist_ok=True)
                np.save(save_name, audio)