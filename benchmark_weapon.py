import tensorflow as tf
import librosa
import numpy as np
import scipy.io.wavfile
from glob import glob
# https://github.com/gabemagee/gunshot_detection/tree/master/raspberry_pi
import os

AUDIO_RATE = 44100
NUMBER_OF_AUDIO_CHANNELS = 1
MINIMUM_FREQUENCY = 20
MAXIMUM_FREQUENCY = AUDIO_RATE // 2
NUMBER_OF_MELS = 128
NUMBER_OF_FFTS = NUMBER_OF_MELS * 20

def power_to_db(S, ref=1.0, amin=1e-10, top_db=80.0):
    S = np.asarray(S)
    if amin <= 0:
        pass
    if np.issubdtype(S.dtype, np.complexfloating):
        magnitude = np.abs(S)
    else:
        magnitude = S

    ref_value = np.abs(ref)

    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))
    if top_db is not None:
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)
    return log_spec

def convert_audio_to_spectrogram(data):
    spectrogram = librosa.feature.melspectrogram(y=data, sr=AUDIO_RATE,
                                                 hop_length=HOP_LENGTH,
                                                 fmin=MINIMUM_FREQUENCY,
                                                 fmax=MAXIMUM_FREQUENCY,
                                                 n_mels=NUMBER_OF_MELS,
                                                 n_fft=NUMBER_OF_FFTS)
    spectrogram = power_to_db(spectrogram)
    spectrogram = spectrogram.astype(np.float32)
    return spectrogram


# Load Model
# Loads 44100 x 1 Keras model from H5 file
interpreter_1 = tf.lite.Interpreter(model_path = "checkpoint/1D.tflite")
interpreter_1.allocate_tensors()
    
# Sets the input shape for the 44100 x 1 model
input_details_1 = interpreter_1.get_input_details()
output_details_1 = interpreter_1.get_output_details()
input_shape_1 = input_details_1[0]['shape']

# Loads 128 x 64 Keras model from H5 file
interpreter_2 = tf.lite.Interpreter(model_path = "checkpoint/128_x_64_2D.tflite")
interpreter_2.allocate_tensors()

# Gets the input shape from the 128 x 64 Keras model
input_details_2 = interpreter_2.get_input_details()
output_details_2 = interpreter_2.get_output_details()
input_shape_2 = input_details_2[0]['shape']

# Loads 128 x 128 Keras model from H5 file
interpreter_3 = tf.lite.Interpreter(model_path = "checkpoint/128_x_128_2D.tflite")
interpreter_3.allocate_tensors()

# Gets the input shape from the 128 x 128 Keras model
input_details_3 = interpreter_3.get_input_details()
output_details_3 = interpreter_3.get_output_details()
input_shape_3 = input_details_3[0]['shape']

# Load Data
data_dir = '/data/sung/dataset/dongwoon'
data_type = 'event' #event, voice

# Label List
label_list = dict(np.load(os.path.join(data_dir, 'val_0.5', 'meta_dict_%s_pre.npz'%data_type)))

positive_list = label_list['1'].tolist()
negative_list = label_list['0'].tolist() + \
                        label_list['-1'].tolist()

file_index = positive_list + negative_list
label_list = [1] * len(positive_list) + [0] * len(negative_list)


# Index
for index in range(len(file_index)):
    name, ix = file_index[index]

    file = dict(np.load(os.path.join(data_dir, 'val_0.5', data_type, name)))
    signal, label = file['audio'][int(ix)], int(file['label'][int(ix)])
    
    wav = np.zeros(32000)
    wav[:len(signal)] = signal
    wav = wav.astype('int16')
    scipy.io.wavfile.write('temp/temp.wav', rate=16000, data=wav)

    wav, sr = librosa.load('temp/temp.wav', sr=16000)
    wav_new = librosa.resample(y = wav, orig_sr=16000, target_sr = 22050)


    # Passes an audio sample of an appropriate format into the model for inference
    processed_data_1 = wav_new
    processed_data_1 = processed_data_1.reshape(input_shape_1)

    HOP_LENGTH = 345 * 2
    processed_data_2 = convert_audio_to_spectrogram(data = wav_new)
    processed_data_2 = processed_data_2.reshape(input_shape_2)
        
    HOP_LENGTH = 345
    processed_data_3 = convert_audio_to_spectrogram(data = wav_new)
    processed_data_3 = processed_data_3.reshape(input_shape_3)


    # Performs inference with the instantiated TensorFlow Lite models
    interpreter_1.set_tensor(input_details_1[0]['index'], processed_data_1)
    interpreter_1.invoke()
    probabilities_1 = interpreter_1.get_tensor(output_details_1[0]['index'])

    interpreter_2.set_tensor(input_details_2[0]['index'], processed_data_2)
    interpreter_2.invoke()
    probabilities_2 = interpreter_2.get_tensor(output_details_2[0]['index'])

    interpreter_3.set_tensor(input_details_3[0]['index'], processed_data_3)
    interpreter_3.invoke()
    probabilities_3 = interpreter_3.get_tensor(output_details_3[0]['index'])

    print("The 44100 x 1 model-predicted probability values: " + str(probabilities_1[0]))
    print("The 128 x 64 model-predicted probability values: " + str(probabilities_2[0]))
    print("The 128 x 128 model-predicted probability values: " + str(probabilities_3[0]))
    print(label)