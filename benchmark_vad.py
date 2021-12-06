import torch
from glob import glob
import numpy as np
import scipy.io.wavfile


# https://github.com/pyannote/pyannote-audio/tree/master/tutorials/pretrained/model

# Load Model
sad = torch.hub.load('pyannote/pyannote-audio', 'sad_ami')

# Load Data
voice_list = glob('data/sample/voice/*.npz')

ind_list_0, ind_list_1 = [], []
for ix in range(20):
    voice = dict(np.load(voice_list[ix]))
    
    for ix2 in range(len(voice['label'])):
        if voice['label'][ix2] == 0:
            ind_list_0.append((ix, ix2))
        elif voice['label'][ix2] == 1: 
            ind_list_1.append((ix, ix2))
        else:
            pass

id, ix = ind_list_0[1]
voice = dict(np.load(voice_list[id]))
wav = voice['audio'][ix]
wav = wav.astype('int16')
scipy.io.wavfile.write('temp/temp.wav', rate=16000, data=wav)

test_file = {'uri': 'temp', 'audio': 'temp/temp.wav'}

# Detect Sound
sad_scores = sad(test_file)
from pyannote.audio.utils.signal import Binarize
binarize = Binarize(offset=0.9, onset=0.9, log_scale=True, 
                    min_duration_off=0.1, min_duration_on=0.1)

# speech regions (as `pyannote.core.Timeline` instance)
speech = binarize.apply(sad_scores, dimension=1)
print(speech)