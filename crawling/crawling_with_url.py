import subprocess
import glob
import numpy as np
import pandas as pd
import os, sys
import json
from utility import Crawling_from_url
import time
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import *
from collections import defaultdict
from tqdm import tqdm
##################### Function for utility ###############################
## Encoding (type = 'audio' : mp4 files to wav audio files, type = 'video' : mp4 files to 30fps video files)
def encoding(type, fps, original, new):
    if type == 'wav':
        subprocess.call(['ffmpeg',  '-i',
                     original, '-vn', '-acodec', 'pcm_s16le', '-ar',
                     str(fps), '-ac', '1', new, '-y'])

    elif type == 'video':
        pass
        # subprocess.call(['ffmpeg', '-i',
        #                  original, '-r', str(fps), new])
    else:
        raise Exception('Input : type should be audio or video')

## Cut the mp4 files
def cut_mp4(original, new, left, right):
    # ffmpeg_extract_subclip(original, left_sec, right_sec, targetname=new)
    subprocess.call(['ffmpeg', '-ss',
                     left, '-i', original, '-t', right, new])

def label_list(class_list):
    human_voice = ["/m/09x0r", "/m/07p6fty", "/m/03qc9zr", "/m/02rtxlg, \
                   /m/01j3sz", "/m/0463cq4", "/m/07qw_06", "/m/07plz5l, \
                   /m/015lz1", "/m/02fxyj", "/m/07s2xch", "/m/07r4k75", "/m/01j423"]

    gunshot = ["/m/04zjc", "/m/02z32qm", "/m/0_1c", "/m/073cg4"]

    for c in class_list:
        c = c.replace('\"', '').replace('\'', '').strip()

        ix = -1

        if c in human_voice:
            ix = 0
            break
        elif c in gunshot:
            ix = 1
            break
        else:
            continue

    return ix

################### Code ######################
if __name__=='__main__':
    # Base
    base = '/data_2/sung/dataset/audioset'
    os.makedirs(base, exist_ok=True)

    # Extracts label
    label_csv = './data/balanced_train_segments.csv'

    with open(label_csv, 'r') as f:
        label_data = f.readlines()

    data_dict = defaultdict(list)
    for ix in range(len(label_data)):
        out = label_data[ix]
        out = out.split(',')

        url = out[0]
        if url[0] == '=':
            url = url[1:]

        time = out[1]

        class_list = out[3:]
        class_ix = label_list(class_list)

        data_dict[str(class_ix)].append([url, time])

    data_ix = np.random.choice(len(data_dict['0']), 1000, replace=False)
    data_0 = np.array(data_dict['0'])[data_ix]
    data_1 = np.array(data_dict['1'])

    # Save the videos of class 0
    os.makedirs(os.path.join(base, '0'), exist_ok=True)
    os.makedirs(os.path.join(base, '0_clip'), exist_ok=True)
    for data_ in tqdm(data_0):
        url = data_[0]
        time = int(float(data_[1].strip()))
        re = Crawling_from_url('https://www.youtube.com/watch?v=' + url, os.path.join(base, '0'), url)

        if re:
            # Clip
            original_name = os.path.join(base,'0', url+'.mp4')
            new_name = os.path.join(base,'0_clip',url+'.mp4')

            time_in = '%02d:%02d:%02d' % (time // 3600, (time % 3600) // 60, time % 60)
            diff = 10
            time_out = '%02d:%02d:%02d' % (diff // 3600, (diff % 3600) // 60, diff % 60)
            cut_mp4(original_name, new_name, time_in, time_out)


    # Save the videos of class 1
    os.makedirs(os.path.join(base, '1'), exist_ok=True)
    os.makedirs(os.path.join(base, '1_clip'), exist_ok=True)

    for data_ in tqdm(data_1):
        url = data_[0]
        time = int(float(data_[1].strip()))
        re = Crawling_from_url('https://www.youtube.com/watch?v=' + url, os.path.join(base, '1'), url)

        if re:
            # Clip
            original_name = os.path.join(base,'1', url+'.mp4')
            new_name = os.path.join(base,'1_clip',url+'.mp4')

            time_in = '%02d:%02d:%02d' % (time // 3600, (time % 3600) // 60, time % 60)
            diff = 10
            time_out = '%02d:%02d:%02d' % (diff // 3600, (diff % 3600) // 60, diff % 60)
            cut_mp4(original_name, new_name, time_in, time_out)

    exit()

    # Encoding the video into audio files (.wav)
    sr = 16000

    original = name
    folder = '/'.join(name.split('/')[:-1])

    new1_wav = folder + '/' + 'audio/' + file_name + '.wav'

    try:
        os.makedirs(folder + '/audio')
    except:
        pass
    _ = encoding('wav', sr, original, new1_wav)