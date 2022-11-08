import torch
from glob import glob
import numpy as np
import scipy.io.wavfile
import os
from tqdm import tqdm

# https://github.com/pyannote/pyannote-audio/tree/master/tutorials/pretrained/model

# Load Model
sad = torch.hub.load('pyannote/pyannote-audio', 'sad_ami')

# number of parameters
model = sad.model_
model_parameters = filter(lambda p: p.requires_grad, model.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])


# Load Data
data_dir = '/data/sung/dataset/dongwoon'
data_type = 'voice' #event, voice

# Label List
label_list = dict(np.load(os.path.join(data_dir, 'val_0.5', 'meta_dict_%s_pre.npz'%data_type)))

positive_list = label_list['1'].tolist()
negative_list = label_list['0'].tolist() + \
                        label_list['-1'].tolist()

file_index = positive_list + negative_list
label_list = [1] * len(positive_list) + [0] * len(negative_list)

# Index
pred_list = []
label_list = []

for index in tqdm(range(len(file_index))):
    name, ix = file_index[index]

    file = dict(np.load(os.path.join(data_dir, 'val_0.5', data_type, name)))
    signal, label = file['audio'][int(ix)], int(file['label'][int(ix)])

    wav = signal.astype('int16')
    scipy.io.wavfile.write('temp/temp.wav', rate=16000, data=wav)

    test_file = {'uri': 'temp', 'audio': 'temp/temp.wav'}

    # Detect Sound
    sad_scores = sad(test_file)
    from pyannote.audio.utils.signal import Binarize
    binarize = Binarize(offset=0.9, onset=0.9, log_scale=True, 
                    min_duration_off=0.1, min_duration_on=0.1)

    # speech regions (as `pyannote.core.Timeline` instance)
    speech = binarize.apply(sad_scores, dimension=1)
    if len(speech) > 0:
        pred_list.append(1)
    else: 
        pred_list.append(0)
    
    label_list.append(label)
    

# Performance
from sklearn.metrics import accuracy_score, precision_score, recall_score
accuracy = accuracy_score(label_list, pred_list)
precision = precision_score(label_list, pred_list)
recall = recall_score(label_list, pred_list)
print('Accuracy %.4f' %accuracy)
print('Precision %.4f' %precision)
print('Recall %.4f' %recall)

# 0.8743966314059772

