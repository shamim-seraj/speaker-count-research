import os
import librosa.display
import matplotlib.pyplot as plt
from os import path
import numpy as np
import librosa.display
from PIL import Image
from scipy import signal


def generate(filename, size, sr, save_path):
    x, fs = librosa.load(filename, sr=sr)
    X = librosa.stft(x)
    Xdb = librosa.amplitude_to_db(abs(X))
    k = plt.figure(figsize=size)
    fig_name = os.path.basename(filename)
    # print("only file name:" + fig_name)
    fig_name = fig_name[:-4]
    librosa.display.specshow(Xdb, sr=fs)
    generated_file = fig_name + '.png'
    print("Generated File: ", generated_file)
    k.savefig(os.path.join(
        save_path,
        generated_file))


def generateV3(filename, size, sr, save_path):
    scale, sr = librosa.load(filename)
    FRAME_SIZE = 2048
    HOP_SIZE = 512
    S_scale = librosa.stft(scale, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
    Y_scale = np.abs(S_scale) ** 2
    Y_log_scale = librosa.power_to_db(Y_scale)
    k = plt.figure(figsize=size, dpi=200)
    plt.axis('off')
    librosa.display.specshow(Y_log_scale, sr=sr, hop_length=HOP_SIZE, x_axis='time', y_axis='log')
    file_name = os.path.basename(filename)[:-4] + '.png'
    k.tight_layout(pad=0)
    k.savefig(os.path.join(
        save_path,
        file_name))


def generate_spectrogramV2(filename, size, save_path):
    # Load sample audio file
    scale, sr = librosa.load(filename)

    # Size of the FFT, which will also be used as the window length
    FRAME_SIZE = 512

    # Step or stride between windows. If the step is smaller than the window length, the windows will overlap
    HOP_SIZE = 128

    # Calculate the spectrogram as the square of the complex magnitude of the STFT
    spectrogram = np.abs(
        librosa.stft(scale, n_fft=FRAME_SIZE, hop_length=HOP_SIZE, win_length=FRAME_SIZE, window='hann')) ** 2
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    k = plt.figure(figsize=size, dpi=50)
    plt.axis('off')
    librosa.display.specshow(spectrogram_db, sr=sr, hop_length=HOP_SIZE)
    file_name = os.path.basename(filename)[:-4] + '.png'
    k.tight_layout(pad=0)
    k.savefig(os.path.join(
        save_path,
        file_name))


def generate_spectrogramV3(filename, size, save_path):
    scale, sr = librosa.load(filename)
    FRAME_SIZE = 512
    HOP_SIZE = 128
    spectrogram = np.abs(
        librosa.stft(scale, n_fft=FRAME_SIZE, hop_length=HOP_SIZE, win_length=FRAME_SIZE, window='hann')) ** 2
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    k = plt.figure(figsize=size, dpi=50)
    data = librosa.display.specshow(spectrogram_db, sr=sr, hop_length=HOP_SIZE)
    file_name = os.path.basename(filename)[:-4] + '.png'

    cm = plt.get_cmap('viridis')
    img = Image.fromarray(data)
    img.save(os.path.join(save_path, file_name))

    '''
    k = plt.figure(figsize=size, dpi=50)
    plt.axis('off')
    lib_img = librosa.display.specshow(spectrogram_db, sr=sr, hop_length=HOP_SIZE)  
    k.tight_layout(pad=0)
    # k.savefig(os.path.join(save_path,file_name))
    plt.imsave(os.path.join(save_path, file_name), lib_img, format='png')
    '''


if __name__ == '__main__':
    clip_path = r'/home/shuvornb/Desktop/NIJ-AI-SMS/testdata/exported/v2/3speakers'
    files_path = os.listdir(clip_path)
    i = 1
    k = 0
    for j in files_path:
        print("Count: ", i)
        i = i + 1

        if path.exists('/home/shuvornb/Desktop/NIJ-AI-SMS/testdata/exported/v2/3speakers_spec/' + j[:-4] + '.png'):
            k = k + 1
            continue
        generate_spectrogramV3(clip_path + '/' + j, (2, 1),
                               '/home/shuvornb/Desktop/NIJ-AI-SMS/testdata/exported/v2/3speakers_spec')
