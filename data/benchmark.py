import numpy as np
import os

if __name__=='__main__':
    data_dir = '/data/sung/dataset/dongwoon'
    data_type = 'event' #event, voice
    
    # Label List
    label_list = dict(np.load(os.path.join(data_dir, 'val_0.5', 'meta_dict_%s_pre.npz'%data_type)))

    positive_list = label_list['1'].tolist()
    negative_list = label_list['0'].tolist() + \
                            label_list['-1'].tolist()

    file_index = positive_list + negative_list
    label_list = [1] * len(positive_list) + [0] * len(negative_list)
    
    # Index
    for index in range(len(file_index)):
        name, ix = file_index[index]

        file = dict(np.load(os.path.join(data_dir, 'val_0.5', data_type, name)))
        signal, label = file['audio'][int(ix)], int(file['label'][int(ix)])
        print(signal)
        print(label)
        exit()
