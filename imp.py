import os
from glob import glob
import numpy as np
import librosa
from scipy.io.wavfile import write
import subprocess

base = '/HDD1/sung/dataset/Battle_1st/audio'

file_list = glob(os.path.join(base, '*.wav'))

ix = 0
for file in file_list:
    if ix > 10:
        break

    wav, sr = librosa.load(file)
    time = len(wav) / sr
    if time > 40:
        print('pass')
        print(time)

        new = './sample/%s' %(os.path.basename(file))

        wav_new = np.array(wav[:(sr * 40)])
        write(new, sr, wav_new)

        ix += 1
