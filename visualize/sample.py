import time
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
import os
from glob import glob

def save_mel_spec(file_path, save_dir):
    event_type, base_name = file_path.split('/')[-2], file_path.split('/')[-1]
    file_id = '%s_%s' %(event_type, base_name.split('/')[0])
    os.makedirs(os.path.join(save_dir, 'melspec'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'spec'), exist_ok=True)
    
    # In Annotation
    unit_time = 0.25

    # Load wav file into numpy array
    file_dict = dict(np.load(file_path))
    
    for index, (signal, label) in enumerate(zip(file_dict['audio'], file_dict['label'])):
        sample_rate = 16000
        
        if label != 1:
            continue
        
        duration = (len(signal) / sample_rate // unit_time) * unit_time

        start_time = 0
        signal = signal[int(start_time * sample_rate):int((start_time + duration) * sample_rate)]

        pre_emphasis = 0.97
        emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

        # Framing
        frame_size = 0.025
        frame_stride = frame_size / 2

        frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
        signal_length = len(emphasized_signal)
        frame_length = int(round(frame_length))
        frame_step = int(round(frame_step))
        num_frames = 3 + (signal_length - frame_length) // frame_step # padding 2 frames
        # num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
        z = np.zeros(int(frame_length / 2))
        pad_signal = np.concatenate([z, emphasized_signal, z]) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

        indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
        num_time_bin = indices.shape[0]
        
        frames = pad_signal[indices.astype(np.int32, copy=False)]

        # Hamming Window
        frames *= np.hamming(frame_length)

        # Fourier-Transform and Power Spectrum
        NFFT = 512

        mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
        spec_frames = mag_frames**2

        weighting = np.hanning(frame_length)[:, None]
        scale = np.sum(weighting**2) * sample_rate
        spec_frames[1:-1, :] *= (2.0 / scale)
        spec_frames[(0, -1), :] /= scale

        freqs = float(sample_rate) / frame_length * np.arange(spec_frames.shape[0])
        eps = 1e-14
        spectrogram = np.log(spec_frames + eps)


        plt.gca().set_aspect('equal')
        plt.pcolor(np.transpose(spectrogram)[:,:num_time_bin]) # dim: (N, T)
        print(np.transpose(spectrogram)[:,:num_time_bin].shape)
        plt.savefig(os.path.join(save_dir, 'spec/%s_%d.png' %(file_id, index)), dpi=300)
        
        pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

        # Filter Banks - Mel
        nfilt = 41
        min_freq = 300
        max_freq = 8000

        low_freq_mel = (2595 * np.log10(1 + min_freq / 700))
        high_freq_mel = (2595 * np.log10(1 + max_freq / 700))  # Convert Hz to Mel
        mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
        hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
        bin = np.floor((NFFT + 1) * hz_points / sample_rate)

        fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
        for m in range(1, nfilt + 1):
            f_m_minus = int(bin[m - 1])   # left
            f_m = int(bin[m])             # center
            f_m_plus = int(bin[m + 1])    # right

            for k in range(f_m_minus, f_m):
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            for k in range(f_m, f_m_plus):
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
        filter_banks = np.dot(pow_frames, fbank.T)
        filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
        filter_banks = 20 * np.log10(filter_banks)  # dB

        filter_banks = filter_banks.T

        num_images = 1 + (filter_banks.shape[1] - num_time_bin) // 5

        print(num_images)
        filter_banks_plot = filter_banks[:, :num_time_bin] # dim: (N, T)
        print(filter_banks.shape)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.pcolor(filter_banks_plot)
        ax.set_aspect('equal')
        plt.savefig(os.path.join(save_dir, 'melspec/%s_%d.png' %(file_id, index)), dpi=300)


if __name__=='__main__':  
    file_list = glob('/data/sung/dataset/dongwoon/label_2.0/voice/*')
    for ix, file_path in enumerate(file_list):
        save_dir = 'imp'
        save_mel_spec(file_path, save_dir)
        
        if ix == 20:
            break
    
    file_list = glob('/data/sung/dataset/dongwoon/label_0.5/event/*')
    for ix, file_path in enumerate(file_list):
        save_dir = 'imp'
        save_mel_spec(file_path, save_dir)
        
        if ix == 20:
            break
    
        