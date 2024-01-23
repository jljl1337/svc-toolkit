import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from numpy import random
from torch.utils.data import Dataset

# from separation.audio import duration
# from audio import duration, load, pre_stft, stft
import audio
# import separation.constants as constants
import constants

class MusicDataset(Dataset):
    def __init__(self, root) -> None:
        self.chunk_length = 20
        self.margin_length = 0.5

        self.root = root
        self.chunk_list = []

        for dir in tqdm(os.listdir(root)):
            vocal_path = os.path.join(root, dir, 'vocals.wav')
            accompaniment_path = os.path.join(root, dir, 'accompaniment.wav')
            mixture_path = os.path.join(root, dir, 'mixture.wav')
            length = duration(mixture_path)
            n_chunks = min(100, int((length - self.margin_length) // self.chunk_length))
            for i in range(n_chunks):
                start = i * (self.chunk_length - self.margin_length)
                self.chunk_list.append((mixture_path, vocal_path,
                                        accompaniment_path, start,
                                        self.chunk_length))
                continue

                post_stft_list = []
                for path in [mixture_path, vocal_path, accompaniment_path]:
                    audio_data, sample_rate = load(path, offset=start, duration=length)
                    audio_data = pre_stft(audio_data, sample_rate, constants.SAMPLE_RATE)
                    post_stft = stft(audio_data, constants.FRAME_LENGTH, constants.FRAME_STEP)
                    post_stft = abs(post_stft)
                    post_stft = post_stft.transpose(2, 1, 0)
                    post_stft_list.append(post_stft)

                start_max = post_stft_list[0].shape[2] - constants.T
                start = random.randint(low = 1, high = start_max)
                for i in range(len(post_stft_list)):
                    post_stft_list[i] = post_stft_list[i][:, :constants.F, start: start + constants.T]

                self.chunk_list.append(post_stft_list)


        
    def __len__(self):
        return len(self.chunk_list)

    def __getitem__(self, index):
        index %= len(self.chunk_list)
        # return self.chunk_list[index]

        chunk_item = self.chunk_list[index]
        path_list = chunk_item[0:3]
        start, length = chunk_item[3:5]
        post_stft_list = []
        for path in path_list:
            audio_data, sample_rate = load(path, offset = start, duration = length)
            audio_data = pre_stft(audio_data, sample_rate, constants.SAMPLE_RATE)
            post_stft = stft(audio_data, constants.FRAME_LENGTH, constants.FRAME_STEP)
            post_stft = abs(post_stft)
            post_stft = post_stft.transpose(2, 1, 0)
            post_stft_list.append(post_stft)

        start = random.randint(low=1,
                               high=(post_stft_list[0].shape[2] - constants.T))
        for i in range(len(post_stft_list)):
            post_stft_list[i] = post_stft_list[i][:, :constants.F, start: start + constants.T]

        return (*post_stft_list, )

class MagnitudeDataset(Dataset):
    def __init__(self, csv_path, expand_factor, win_length, hop_length, patch_length) -> None:
        self.magnitudes = []
        self.expanded_magnitudes = []

        df = pd.read_csv(csv_path)

        for index, row in tqdm(df.iterrows(), total=len(df)):
            # Load audio
            mixture_path = row['mixture_path']
            stem_path = row['stem_path']
            mixture_wave, mixture_sr = audio.load(mixture_path)
            stem_wave, _stem_sr = audio.load(stem_path)

            # Save magnitude
            tmp = audio.mix_stem_to_mag_phase(mixture_wave, stem_wave, constants.WIN_LENGTH, constants.HOP_LENGTH)
            mix_magnitude, stem_magnitude = tmp[0], tmp[2]
            self.magnitudes.append((mix_magnitude, stem_magnitude))

            # Expand dataset by duration of each song
            duration = mixture_wave.shape[0] / mixture_sr
            weight = int(duration // expand_factor + 1)
            index = len(self.magnitudes) - 1
            self.expanded_magnitudes.extend([index] * weight)

    def __len__(self):
        return len(self.expanded_magnitudes)
    
    def __getitem__(self, index):
        actual_index = self.expanded_magnitudes[index]
        mix_magnitude, stem_magnitude = self.magnitudes[actual_index]
        start = random.randint(0, mix_magnitude.shape[1] - constants.PATCH_LENGTH + 1)

        mix_magnitude = mix_magnitude[: -1, start: start + constants.PATCH_LENGTH, np.newaxis]
        stem_magnitude = stem_magnitude[: -1, start: start + constants.PATCH_LENGTH, np.newaxis]

        mix_tensor = torch.from_numpy(mix_magnitude).permute(2, 0, 1)
        stem_tensor = torch.from_numpy(stem_magnitude).permute(2, 0, 1)

        return mix_tensor, stem_tensor

if __name__ == "__main__":
    print('test')
    dataset = MagnitudeDataset('train.csv', 30)