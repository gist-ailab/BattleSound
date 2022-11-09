from glob import glob
import numpy as np
import os
from tqdm import tqdm


if __name__=='__main__':
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