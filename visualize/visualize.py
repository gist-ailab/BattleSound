import librosa
import numpy as np

def condition_label(wav_array, annot, sr):
    neg_id = np.where(annot == 0)[0]
    pos_id = np.where(annot == 1)[0]

    base_signal = wav_array.copy()
    neg_signal = wav_array.copy()

    for n in neg_id:
        n_init = int(n * sr / 2)
        base_signal[n_init:int(n_init+sr/2)] = np.nan

    for p in pos_id:
        p_init = int(p * sr / 2)
        neg_signal[p_init:int(p_init+sr/2)] = np.nan

    return base_signal, neg_signal

# Audioset
file_name = './samples/audioset.mp3'
wav, sr = librosa.load(file_name)

init = sr * 16
wav = wav[init:(init + sr*10)]
t = np.array(list(range(0, len(wav)))) / sr # Time

annot= np.array([1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

b, n = condition_label(wav, annot, sr)

import matplotlib.pyplot as plt
plt.switch_backend('TkAgg')

# Plot daily and weekly resampled time series together
fig, ax = plt.subplots()

ax.plot(t, b, 'b')
ax.plot(t, n, 'r')
ax.set_xlabel('Time (s)', fontsize=14)
ax.set_ylabel('Amplitude', fontsize=14)
ax.grid(True)
ax.legend()
# plt.savefig('./test.png')