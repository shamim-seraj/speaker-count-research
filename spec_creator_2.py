import os
import librosa.display
import matplotlib.pyplot as plt
from os import path
import numpy as np
import librosa.display


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


if __name__ == '__main__':
    clip_path = r'/home/shuvornb/Desktop/NIJ-AI-SMS/testdata/exported/v2/2speakers'
    files_path = os.listdir(clip_path)
    i = 1
    k = 0
    for j in files_path:
        print("Count: ", i)
        i = i + 1

        if path.exists('/home/shuvornb/Desktop/NIJ-AI-SMS/testdata/exported/v2/2speakers_spec/' + j[:-4] + '.png'):
            k = k + 1
            continue
        generate_spectrogramV2(clip_path + '/' + j, (2, 1), '/home/shuvornb/Desktop/NIJ-AI-SMS/testdata/exported/v2/2speakers_spec')
        if i-k == 500:
            break
