import os
import time
import contextlib
import msvcrt
import glob
from scipy.io.wavfile import read
import numpy as np

with contextlib.redirect_stdout(None):
    import pygame

LABLE1 = 0
LABLE2 = 1
LABLE3 = 2
UNKOWN = 3

ann_dict = {
    0: 'VOICE',
    1: 'EFFECT',
    2: 'MIXED',
    3: 'UNKNOWN'
}

unit_time = 0.5
wait_time = 2

audio_list = sorted(glob.glob('audio\*.wav'))
ann_dir = 'annotation'

if len(audio_list) == 0:
    raise FileNotFoundError

if not os.path.isdir(ann_dir):
    os.system('mkdir ' + ann_dir)

ann_list = ['audio' + ann[10:-3] + 'wav' for ann in sorted(glob.glob('annotation\*.npy'))]

if not len(ann_list) == 0:
    print('\nPre-existing annotation files are detected')
    while True:
        print('Do you want to resume? Press [y/n]: ')
        resume_decision = msvcrt.getch()
        if resume_decision.lower() == b'y':
            audio_list = sorted(list(set(audio_list) - set(ann_list)))
            print('Start from file {}...'.format(audio_list[0][6:]))
            time.sleep(0.5)
            os.system('cls')
            break

        elif resume_decision.lower() == b'n':
            print('Start from scratch...')
            time.sleep(0.5)
            os.system('cls')
            break

        else:
            pass


for audio_dir in audio_list:

    idx = 0
    ann_list = []

    audio_name = audio_dir[6:]
    ann_name = audio_name[:-4]

    fs, audio_arr = read(audio_dir)
    n_channels = len(audio_arr.shape)
    unit_sample = int(fs * unit_time)
    max_idx = len(audio_arr) // unit_sample
    pygame.mixer.pre_init(fs, channels=n_channels)
    pygame.mixer.init()

    print('\nFile: {}\nPress [Enter] to start: '.format(audio_name))
    input()

    time.sleep(0.2)
    # for i in reversed(range(wait_time)):
    #     print('Audio plays after {} sec...'.format(i + 1), end='\r')
    #     time.sleep(1.0)

    # print('Audio plays after 0 sec...\n')
    while idx < max_idx:
        print('Audio plays...')
        sound = pygame.sndarray.make_sound(audio_arr[unit_sample*idx:unit_sample*(idx + 1)])
        sound.play()
        time.sleep(unit_time)
        command = msvcrt.getch()
        # print(command)
        if command.lower() == b'a':
            print('Labeled with VOICE')
            ann_list.append(LABLE1)
            idx += 1
            print('Recent 5 labels: {}\n'.format([ann_dict[ann] for ann in ann_list[-5:]]))

        elif command.lower() == b's':
            print('Labeled with GUN')
            ann_list.append(LABLE2)
            idx += 1
            print('Recent 5 labels: {}\n'.format([ann_dict[ann] for ann in ann_list[-5:]]))

        elif command.lower() == b'd':
            print('Labeled with MIXED')
            ann_list.append(LABLE3)
            idx += 1
            print('Recent 5 labels: {}\n'.format([ann_dict[ann] for ann in ann_list[-5:]]))

        elif command.lower() == b'f':
            print('Labeled with UNKNOWN')
            ann_list.append(UNKOWN)
            idx += 1
            print('Recent 5 labels: {}\n'.format([ann_dict[ann] for ann in ann_list[-5:]]))

        elif command.lower() == b'v':
            if idx == 0:
                print('There is no previous audio')
            else:
                print('Get back')
                del ann_list[-1]
                idx -= 1
            print('Recent 5 labels: {}\n'.format([ann_dict[ann] for ann in ann_list[-5:]]))

        elif command.lower() == b'r':
            print('Repeat')
            print('Recent 5 labels: {}\n'.format([ann_dict[ann] for ann in ann_list[-5:]]))

        elif command.lower() == b'q':
            while True:
                print('Do you really want to quit the program? Press [y/n]: ')
                quit_decision = msvcrt.getch()
                if quit_decision.lower() == b'y':
                    print('Quit program...')
                    quit()
                elif quit_decision.lower() == b'n':
                    print('Cancel...\n')
                    break
                else:
                    pass

        else:
            print('Unexpected input')
            print('Recent 5 labels: {}\n'.format([ann_dict[ann] for ann in ann_list[-5:]]))

        if idx == max_idx:
            assert len(ann_list) == max_idx
            print('File {} finished'.format(audio_name))
            np.save(ann_dir + '\\' + ann_name, np.array(ann_list))
            time.sleep(0.5)
            os.system('cls')
