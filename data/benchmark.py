import numpy as np

if __name__=='__main__':
    data_dir = ''
    label_list = dict(np.load(data_dir, 'val_0.5', 'meta_dict_%s_pre.npz'%option.result['data']['data_type'])))

self.positive_list = self.label_list['1'].tolist()
self.negative_list = self.label_list['0'].tolist() + \
                        self.label_list['-1'].tolist()

self.file_index = self.positive_list + self.negative_list
self.label_list = [1] * len(self.positive_list) + [0] * len(self.negative_list)




name, ix = self.file_index[index]

file = dict(np.load(os.path.join(self.option.result['data']['data_dir'], 'val_0.5', self.option.result['data']['data_type'], name)))
signal, label = file['audio'][int(ix)], int(file['label'][int(ix)])
