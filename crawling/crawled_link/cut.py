import scipy.io.wavfile as wavfile
import numpy as np
import time


data, rate = wavfile.read('./PUBG')




# time_cri = 0.8
#
# lag = int(time_cri * rate)
#
# len_range = int(len(data)/lag)
#
# for i in range(len_range):
#     data_seg = data[i * lag : (i+1) * lag]
#     wavfile.write('%s_%.2f_%.2f.wav' %('ex1', time_cri * i, time_cri * (i+1)), rate, data_seg)
#
#     if i > 10:
#         break
