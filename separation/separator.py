import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from dotenv import load_dotenv

import constants
import models
import audio

class Separator():
    def __init__(self, model_path, device) -> None:
        pass

    def load_file(self, file, sample_rate):
        wave, sr = audio.load(file, mono=False)

        if sr != sample_rate:
            wave = audio.resample(wave, sr, sample_rate)

    def separate(self, wave, window_length, hop_length, patch_length):

        if wave.ndim == 1:
            wave = wave[np.newaxis, :]

        old_len = wave.shape[1]
        factor = patch_length * hop_length
        new_len = math.ceil(old_len / factor) * factor - 1
        diff = new_len - wave.shape[1]
        wave = np.pad(wave, ((0, 0), (0, diff)), mode='constant')

        magnitude, phase = audio.to_mag_phase(wave, window_length, hop_length)

        # TODO: pre_magnitude

        pre_wave = audio.to_wave(pre_magnitude, phase, window_length, hop_length)

        if pre_wave.shape[0] == 1:
            pre_wave = pre_wave[0]

        return pre_wave

def main():
    load_dotenv()
    MODEL_DIR = os.getenv('MODEL_DIR')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    separator = Separator(MODEL_DIR, device)
    separator.separate_from_file('/home/jljl1337/dataset/musdb18wav/test/Forkupines - Semantics/mixture.wav', './output/')

if __name__ == "__main__":
    main()