import numpy as np

if __name__=='__main__':
    a = '/data/sung/dataset/dongwoon/label_2.0/meta_dict_event.npz'
    a = dict(np.load(a))
    print(a) 
    
    a = '/data/sung/dataset/dongwoon/label_2.0/event/pubg_0604_010.npz'
    a = dict(np.load(a))
    print(a)