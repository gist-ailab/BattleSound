import os
import numpy as np
import matplotlib.pyplot as plt
import cv2

def extract_mel_spec(signal, sample_rate=16000):
    # In Annotation
    unit_time = 0.25

    # Load wav file into numpy array
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
    spectrogram = np.transpose(np.log(spec_frames + eps))

    # Mel    
    nfilt = 41
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum
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
    return filter_banks

def overlay_heatmap(img, heatmap):
    heatmap = np.uint8(heatmap * 255)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    
    cam = heatmap * 0.5 + (img / 255) * 0.5
    # cam = cam / np.max(cam)
    cam = np.uint8(255 * cam)
    return cam

def draw_melspec(mel_spec, label_list, event_type, duration, axes, row, col, draw_type='mel'):
    title_size = 12
    label_size = 11

    ix = col
    ax = axes.flatten()[ix]
    
    if draw_type == 'mel':
        ax.pcolor(mel_spec[:, :-1])
    else:
        ax.imshow(mel_spec[:, :-1])
        

    if duration == 0.5:
        time_stamps = [0, 20, 40]
    elif duration == 2.0:
        time_stamps = [0, 40, 80, 120, 160]
    elif duration == 4.0:
        time_stamps = [0, 80, 160, 240, 320]
    elif duration == 8.0:
        time_stamps = [0, 160, 320, 480, 640]
    else:
        raise('Select Proper Duration')

    time_stamp_labels = np.array(time_stamps) / 80.
    ax.set_xticks(time_stamps)
    ax.set_xticklabels(time_stamp_labels)

    ylim = 41
    label_index_list = []
    label_old = 0
    label_new = []

    for ix, label in enumerate(label_list):
        if label == 1:
            if label_old != 1:
                label_new = [ix]
            else:
                label_new.append(ix)
        else:
            if label_old == 1:
                label_index_list.append(label_new)
                label_new = []
            else:
                continue

        label_old = label
    
    # for Last element
    if label_old == 1:
        label_new.append(ix+1)
        label_index_list.append(label_new)
        label_new = []

    alpha = 0.8
    for index in label_index_list:
        if len(index) == 1:
            # ax.annotate("", xy=(40 * index[0], ylim * 0.93), xytext=(40 * index[0] + 40, ylim * 0.93), arrowprops=dict(arrowstyle="<->", alpha=alpha, lw=2))
            ax.annotate("", xy=(40 * index[0], ylim * 0.07), xytext=(40 * index[0] + 40, ylim * 0.07), arrowprops=dict(arrowstyle="<->", alpha=alpha, lw=2))
            ax.annotate("", (40 * index[0], 0), xytext=(40 * index[0], 41), rotation=90, va='top', arrowprops = {'width': 0, 'headwidth': 0, 'linestyle': '--', 'alpha':alpha})
            ax.annotate("", (40 * index[0]+40, 0), xytext=(40 * index[0]+40, 41), rotation=90, va='top', arrowprops = {'width': 0, 'headwidth': 0, 'linestyle': '--', 'alpha':alpha})
        else:
            # ax.annotate("", xy=(40 * index[0], ylim * 0.93), xytext=(40 * index[-1], ylim * 0.93), arrowprops=dict(arrowstyle="<->", alpha=alpha, lw=2))
            ax.annotate("", xy=(40 * index[0], ylim * 0.07), xytext=(40 * index[-1], ylim * 0.07), arrowprops=dict(arrowstyle="<->", alpha=alpha, lw=2))
            ax.annotate("", (40 * index[0], 0), xytext=(40 * index[0], 41), rotation=90, va='top', arrowprops = {'width': 0, 'headwidth': 0, 'linestyle': '--', 'alpha':alpha})
            ax.annotate("", (40 * index[-1], 0), xytext=(40 * index[-1], 41), rotation=90, va='top', arrowprops = {'width': 0, 'headwidth': 0, 'linestyle': '--', 'alpha':alpha})
        
        ax.xaxis.set_tick_params(labelsize=label_size)
        ax.yaxis.set_tick_params(labelsize=label_size)
        
        # if ix == 0:
        #     ax.set_ylabel('Mel Filter', fontsize=y_size)
        # ax.set_xlabel('Time (s)', fontsize=x_size)
        
        if event_type == 'voice':
            letter = 'VOICE'
        elif event_type == 'event':
            letter = 'EVENT'
        
        ax.set_title('%s (%.1fs)' %(letter, duration), fontsize=title_size)
        
        
if __name__=='__main__':
    # Voice
    event_type = 'voice'
    out_voice = []
    for duration in [0.5, 2.0, 4.0, 8.0]: # 13 10 11 12
        window = int(duration / 0.5)

        source_dir = '/data/sung/dataset/dongwoon/label_0.5/%s' %event_type
        label_dict = dict(np.load('/data/sung/dataset/dongwoon/label_%.1f/meta_dict_%s.npz' %(duration, event_type)))
        source_list = label_dict['1']

        for ix, source in enumerate(source_list):
            data = dict(np.load(os.path.join(source_dir, source[0])))
            signal = data['audio'][int(source[1]):int(source[1]) + window].reshape(-1)
            label_list = data['label'][int(source[1]):int(source[1]) + window]
            mel_spec = extract_mel_spec(signal, 16000)
            
            mel_spec_rgb = np.tile(np.expand_dims(mel_spec, axis=-1), (1,1,3))
            gradcam = np.squeeze(np.load('/data/sung/dataset/dongwoon/gradcam/label_%.1f/%s/%s_%d.npy' %(duration, event_type, source[0].rstrip('.npz'), int(source[1]))))
            overlay_img = overlay_heatmap(mel_spec_rgb, gradcam)
            
            out_voice.append([mel_spec, overlay_img, label_list, event_type, duration, ix])
        
            if ix > 300:
                break

    for out in out_voice:
        mel_spec, overlay_img, label_list, event_type, duration, ix = out
        save_path = './vis_result/%s/%.1f/%d.png' %(event_type, duration, ix)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(8, 2))
        
        draw_melspec(mel_spec, label_list, event_type, duration, axes, 0, 0, draw_type='mel')
        draw_melspec(overlay_img, label_list, event_type, duration, axes, 0, 1, draw_type='heatmap')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(1)
        
    
    # Event
    event_type = 'event'
    out_voice = []
    for duration in [0.5, 2.0, 4.0, 8.0]: # 13 10 11 12
        window = int(duration / 0.5)

        source_dir = '/data/sung/dataset/dongwoon/label_0.5/%s' %event_type
        label_dict = dict(np.load('/data/sung/dataset/dongwoon/label_%.1f/meta_dict_%s.npz' %(duration, event_type)))
        source_list = label_dict['1']

        for ix, source in enumerate(source_list):
            data = dict(np.load(os.path.join(source_dir, source[0])))
            signal = data['audio'][int(source[1]):int(source[1]) + window].reshape(-1)
            label_list = data['label'][int(source[1]):int(source[1]) + window]
            mel_spec = extract_mel_spec(signal, 16000)
            
            mel_spec_rgb = np.tile(np.expand_dims(mel_spec, axis=-1), (1,1,3))
            gradcam = np.squeeze(np.load('/data/sung/dataset/dongwoon/gradcam/label_%.1f/%s/%s_%d.npy' %(duration, event_type, source[0].rstrip('.npz'), int(source[1]))))
            overlay_img = overlay_heatmap(mel_spec_rgb, gradcam)
            
            out_voice.append([mel_spec, overlay_img, label_list, event_type, duration, ix])
            
            if ix > 300:
                break
        

    for out in out_voice:
        mel_spec, overlay_img, label_list, event_type, duration, ix = out
        save_path = './vis_result/%s/%.1f/%d.png' %(event_type, duration, ix)
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(8, 2))
        
        draw_melspec(mel_spec, label_list, event_type, duration, axes, 0, 0, draw_type='mel')
        draw_melspec(overlay_img, label_list, event_type, duration, axes, 0, 1, draw_type='heatmap')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)    
        plt.close(1)