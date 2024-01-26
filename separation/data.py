import os
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed

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

class MagnitudeDataset(Dataset):
    def __init__(self, csv_path, expand_factor, win_length, hop_length, patch_length) -> None:
        self.win_length = win_length
        self.hop_length = hop_length
        self.patch_length = patch_length
        self.expand_factor = expand_factor

        df = pd.read_csv(csv_path)

        self.magnitudes = [None] * len(df)
        self.expanded_magnitudes = []

        with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            futures = [executor.submit(self.load_magnitude, index, row) for index, row in df.iterrows()]

            for future in tqdm(as_completed(futures), total=len(df)):
                index, mix_magnitude, stem_magnitude, weight = future.result()
                self.magnitudes[index] = (mix_magnitude, stem_magnitude)

                self.expanded_magnitudes.extend([index] * weight)

            self.expanded_magnitudes = sorted(self.expanded_magnitudes)

    def load_magnitude(self, index, row):
        # Load audio
        mixture_path = row['mixture_path']
        stem_path = row['stem_path']
        mixture_wave, mixture_sr = audio.load(mixture_path)
        stem_wave, _stem_sr = audio.load(stem_path)

        # Save magnitude
        tmp = audio.mix_stem_to_mag_phase(mixture_wave, stem_wave, self.win_length, self.hop_length)
        mix_magnitude, stem_magnitude = tmp[0], tmp[2]

        # Expand dataset by duration of each song
        duration = mixture_wave.shape[0] / mixture_sr
        weight = int(duration // self.expand_factor + 1)

        return index, mix_magnitude, stem_magnitude, weight

    def __len__(self):
        return len(self.expanded_magnitudes)
    
    def __getitem__(self, index):
        actual_index = self.expanded_magnitudes[index]
        mix_magnitude, stem_magnitude = self.magnitudes[actual_index]
        start = random.randint(0, mix_magnitude.shape[1] - self.patch_length + 1)

        mix_magnitude = mix_magnitude[: -1, start: start + self.patch_length, np.newaxis]
        stem_magnitude = stem_magnitude[: -1, start: start + self.patch_length, np.newaxis]

        mix_tensor = torch.from_numpy(mix_magnitude).permute(2, 0, 1)
        stem_tensor = torch.from_numpy(stem_magnitude).permute(2, 0, 1)

        return mix_tensor, stem_tensor

if __name__ == "__main__":
    print('test')
    dataset = MagnitudeDataset('train.csv', 30, 1024, 768, 128)