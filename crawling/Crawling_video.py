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

## Using the xls files with labels, get the 'video', 'title' and 'labels'
def get_label(xls_file, crawling=True):
    def split(list_data, n_group):
        lab = []
        length = len(list_data) // n_group

        if len(list_data) % n_group != 0:
            raise Exception('Length of list_data must be the multiplication of the n_group')

        for i in range(length):
            part = list_data[:n_group]
            lab.append(part)
            list_data = list_data[n_group:]
        return lab

    df = pd.read_excel(xls_file)
    df = df.dropna(subset=['gamenum', 'url'])
    df = df.drop_duplicates(['url'], keep='first')
    df = df.replace([' ', '  ', '   '], None)


    column_name = list(df.columns)
    last_num = column_name.index('end')-1
    start_num = column_name.index('1i')

    # Selecting the target columns (game_category, url, labels)
    game_num = list(df.loc[:,'gamenum'])
    file_num = list(df.loc[:, 'num'])
    url = list(df.loc[:,'url'])
    label_list = df.iloc[:, start_num:(last_num+1)]

    label_all = []

    size_label = len(game_num)
    for i in range(size_label):
        label = label_list.iloc[i, :].dropna()
        label = split(list(label), 2)
        label_all.append(label)

    return ({'game_num' : game_num, 'url': url, 'label':label_all, 'file_num':file_num}, size_label)

## Labeling the game number into game name
def num2name(game_num):
    if game_num == 5:
        game_folder = 'clash'
    elif game_num == 4:
        game_folder = 'chulgwon'
    elif game_num == 3:
        game_folder = 'cart'
    elif game_num == 2:
        game_folder = 'over'
    elif game_num == 1:
        game_folder = 'bag'

    else:
        raise Exception('game_num must be the one of the number between 0 ~ 5')

    return game_folder

################### Code ######################
# 1. Clipping the videos using the labels
label_xls = './sample/labels1.xlsx'
get_lab, data_num = get_label(label_xls)
file_folder = './sample/'

#### Options ####
url_type = True
clipping = True
sr = 16000
#################

if clipping == True:
    for i in range(data_num):
        time.sleep(2)
        # Two types crawling from the url or pre-crawled
        if url_type == True:
            file_num = get_lab['file_num'][i]
            Crawling_from_url(get_lab['url'][i], file_folder, file_num)

        labels_i = get_lab['label'][i]
        game_i = get_lab['game_num'][i]

        original_filename = file_num

        # Segment the video using the labels
        for ix, label_i in enumerate(labels_i):
            original = file_folder + original_filename + '.mp4'
            new_filename = original_filename + '_%d'%ix + '.mp4'

            game_folder = num2name(game_i)
            new = file_folder + game_folder + '/' + new_filename

            try:
                os.makedirs(file_folder + game_folder)
            except:
                pass

            try:
                print('***************** Cutting **************')
                print(original, new)
                # cut_mp4(original, new, int(label_i[0]), int(label_i[1]))
                time_in = '%02d:%02d:%02d' %(int(label_i[0]) // 3600, (int(label_i[0]) % 3600)//60 ,int(label_i[0]) % 60)
                diff = int(label_i[1]) - int(label_i[0])
                time_out ='%02d:%02d:%02d' %(diff// 3600, (diff % 3600)//60 , diff % 60)
                cut_mp4(original, new, time_in, time_out)

            except:
                pass

# 2. Encoding (Video and Music files)
game_list = ['chulgwon', 'clash', 'cart', 'over', 'bag']
crawled_data = []
for game in game_list:
    crawled = glob.glob(file_folder + game + '/*.mp4')
    crawled_data = crawled_data + crawled

# Loop to the directory
for name in crawled_data:
    name = name.replace('\\', '/')
    file_name = name.split('/')[-1][:-4]

    # Set the original, new files
    original = name
    folder = '/'.join(name.split('/')[:-1])

    new1_wav = folder + '/' + 'audio/' + file_name + '.wav'

    try:
        os.makedirs(folder + '/audio')
    except:
        pass
    _ = encoding('wav', sr, original, new1_wav)


