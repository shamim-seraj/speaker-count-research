import os
from os import path
import librosa.display
import librosa.display
import matplotlib.pyplot as plt
import numpy as np


def generate_spectrogram(filename, size, save_path):
    scale, sr = librosa.load(filename)
    FRAME_SIZE = 512
    HOP_SIZE = 128
    spectrogram = np.abs(
        librosa.stft(scale, n_fft=FRAME_SIZE, hop_length=HOP_SIZE, win_length=FRAME_SIZE, window='hann')) ** 2
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    plt.axis('off')
    plot = plt.figure(figsize=size, dpi=50)
    librosa.display.specshow(spectrogram_db, sr=sr, hop_length=HOP_SIZE)
    file_name = os.path.basename(filename)[:-4] + '.png'
    plot.tight_layout(pad=0)
    plot.savefig(os.path.join(save_path, file_name))
    plt.close()


if __name__ == '__main__':
    clip_path = '/home/shuvornb/Desktop/NIJ-AI-SMS/testdata/exported/v2/2speakers'
    files_path = os.listdir(clip_path)
    i = 1
    for j in files_path:
        print("Count: ", i)
        i = i + 1
        if path.exists('/home/shuvornb/Desktop/NIJ-AI-SMS/testdata/exported/v2/2speakers_spec/' + j[:-4] + '.png'):
            continue
        generate_spectrogram(clip_path + '/' + j, (2, 1), '/home/shuvornb/Desktop/NIJ-AI-SMS/testdata/exported/v2/2speakers_spec')
