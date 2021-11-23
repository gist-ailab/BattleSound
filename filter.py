import numpy as np
import scipy as sp
from scipy.io.wavfile import read
from scipy.io.wavfile import write     # Imported libaries such as numpy, scipy(read, write), matplotlib.pyplot
from scipy import signal
import matplotlib.pyplot as plt
import soundfile as sf
import numpy as np
import scipy

wav_file = 'pubg_clip.wav'
wav_norm, _ = sf.read(wav_file)
sr, wav = read(wav_file)
wav = wav[:, 0]
wav_norm = wav_norm[:, 0]


plt.figure()
plt.plot(wav_norm) 
plt.title('Original Signal Spectrum')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
plt.savefig('original image.png')
plt.close(1)

FourierTransformation = sp.fft.fft(wav) # Calculating the fourier transformation of the signal
scale = np.linspace(0, sr, len(wav))

# Low-pass-filter
c,d = signal.butter(5, 100/(sr/2), btype='lowpass') # ButterWorth low-filter
wav_filt = signal.lfilter(c,d,wav) # Applying the filter to the signal
wav_filt[wav_norm < 0.2] = 0

plt.figure()
plt.plot(wav_filt) # plotting the signal.
plt.title('Lowpass Filter')
plt.xlabel('Frequency(Hz)')
plt.ylabel('Amplitude')
plt.savefig('Low-pass-filter.png')
plt.close(1)