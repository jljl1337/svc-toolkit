import math
from typing import Iterable
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import numpy as np
from pandas import read_csv
from torch.utils.data import Dataset
from tqdm import tqdm

from separation.audio import load, to_magnitude, to_mag_phase, pad_wave
from separation.constants import CSV_MIXTURE_PATH_COLUMN, CSV_STEM_PATH_COLUMN, NYQUIST, ZERO, NEGLECT_FREQUENCY_OPTIONS

class MagnitudeRandomDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        expand_factor: float,
        win_length: int,
        hop_length: int,
        patch_length: int,
        neglect_frequency: str,
        sample_rate: int,
    ):
        # Validate neglect_frequency
        if neglect_frequency not in NEGLECT_FREQUENCY_OPTIONS:
            raise ValueError(f'Invalid neglect_frequency: {self.neglect_frequency}')

        # Save parameters
        self.win_length = win_length
        self.hop_length = hop_length
        self.patch_length = patch_length
        self.expand_factor = expand_factor
        self.neglect_frequency = neglect_frequency
        self.sample_rate = sample_rate

        # Load CSV
        df = read_csv(csv_path)

        # Set up magnitudes
        self.magnitudes = [None] * len(df)
        self.expanded_magnitudes = []

        # Load magnitudes in parallel
        with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            # Submit tasks
            futures = [executor.submit(self.load_magnitude, index, row) for index, row in df.iterrows()]

            # For each finished task
            for future in tqdm(as_completed(futures), total=len(df)):
                index, mix_magnitude, stem_magnitude, weight = future.result()

                # Save magnitude
                self.magnitudes[index] = (mix_magnitude, stem_magnitude)
                self.expanded_magnitudes.extend([index] * weight)

            # Sort expanded magnitudes since they are not executed in order
            self.expanded_magnitudes = sorted(self.expanded_magnitudes)

    def load_magnitude(self, index, row):
        # Load audio
        mixture_path = row[CSV_MIXTURE_PATH_COLUMN]
        stem_path = row[CSV_STEM_PATH_COLUMN]
        mixture_wave, _mixture_sr = load(mixture_path, sr=self.sample_rate)
        stem_wave, _stem_sr = load(stem_path, sr=self.sample_rate)

        # Get magnitude
        mix_magnitude = to_magnitude(mixture_wave, self.win_length, self.hop_length)
        stem_magnitude = to_magnitude(stem_wave, self.win_length, self.hop_length)

        # Normalize magnitude
        mix_magnitude_max = mix_magnitude.max()
        mix_magnitude /= mix_magnitude_max
        stem_magnitude /= mix_magnitude_max

        # Neglect frequency to match model input
        if self.neglect_frequency == NYQUIST:
            mix_magnitude = mix_magnitude[: -1]
            stem_magnitude = stem_magnitude[: -1]
        elif self.neglect_frequency == ZERO:
            mix_magnitude = mix_magnitude[1:]
            stem_magnitude = stem_magnitude[1:]

        # Expand dataset by duration of each song
        duration = mixture_wave.shape[0] / self.sample_rate
        weight = math.ceil(duration / self.expand_factor)

        return index, mix_magnitude, stem_magnitude, weight

    def __len__(self):
        return len(self.expanded_magnitudes)
    
    def __getitem__(self, index):
        # Get a magnitude from the expanded list
        actual_index = self.expanded_magnitudes[index]
        mix_magnitude, stem_magnitude = self.magnitudes[actual_index]

        # Randomly select a patch
        start = np.random.randint(0, mix_magnitude.shape[1] - self.patch_length + 1)
        mix_magnitude = mix_magnitude[np.newaxis, :, start: start + self.patch_length]
        stem_magnitude = stem_magnitude[np.newaxis, :, start: start + self.patch_length]

        # Convert to tensor
        mix_tensor = torch.from_numpy(mix_magnitude)
        stem_tensor = torch.from_numpy(stem_magnitude)

        return mix_tensor, stem_tensor

class MagnitudeDataset(Dataset):
    def __init__(
        self,
        paths: Iterable[str],
        expand_factor: float,
        win_length: int,
        hop_length: int,
        patch_length: int,
        neglect_frequency: str,
        sample_rate: int,
    ):
        # Validate neglect_frequency
        if neglect_frequency not in NEGLECT_FREQUENCY_OPTIONS:
            raise ValueError(f'Invalid neglect_frequency: {self.neglect_frequency}')

        # Save parameters
        self.win_length = win_length
        self.hop_length = hop_length
        self.patch_length = patch_length
        self.expand_factor = expand_factor
        self.neglect_frequency = neglect_frequency
        self.sample_rate = sample_rate

        # Set up magnitudes
        self.magnitude_list = [None] * len(paths)
        self.phase_list = [None] * len(paths)
        self.magnitude_max_list = [None] * len(paths)
        self.old_length_list = [None] * len(paths)
        self.index_channel_patch_list = []

        # Load magnitudes in parallel
        with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            # Submit tasks
            futures = [executor.submit(self._load_magnitude, index, path) for index, path in enumerate(paths)]

            # For each finished task
            for future in tqdm(as_completed(futures), total=len(paths)):
                index, magnitude, phase, magnitude_max, channel_num, patch_num, old_length = future.result()

                # Save magnitude, phase, and old length
                self.magnitude_list[index] = magnitude
                self.phase_list[index] = phase
                self.magnitude_max_list[index] = magnitude_max
                self.old_length_list[index] = old_length
                
                # Expand dataset by channel and patch number
                song_index_channel_patch_list = [(index, channel, patch)
                                                 for channel in range(channel_num)
                                                 for patch in range(patch_num)]
                self.index_channel_patch_list.extend(song_index_channel_patch_list)

        # Sort index_channel_patch_list
        self.index_channel_patch_list = sorted(self.index_channel_patch_list)

    def _load_magnitude(self, index: int, path: str):
        # Load audio
        wave, _sr = load(path, sr=self.sample_rate, mono=False)

        # Pad wave
        padded_wave = pad_wave(wave, self.win_length, self.hop_length)

        # Get magnitude
        magnitude, phase = to_mag_phase(padded_wave, self.win_length, self.hop_length)

        # Normalize magnitude
        magnitude_max = magnitude.max()
        magnitude /= magnitude_max

        old_length = wave.shape[-1]
        channel_num = magnitude.shape[0]
        patch_num = magnitude.shape[-1] / self.patch_length

        # Check if patch_num is integer
        if patch_num != int(patch_num):
            raise ValueError(f'Patch length {self.patch_length} is not compatible with magnitude shape {magnitude.shape}')

        patch_num = int(patch_num)

        return index, magnitude, phase, magnitude_max, channel_num, patch_num, old_length

    def __len__(self):
        return len(self.index_channel_patch_list)
    
    def __getitem__(self, index):
        # Get a magnitude from the expanded list
        actual_index, channel, patch = self.index_channel_patch_list[index]
        magnitude = self.magnitude_list[actual_index]

        # Select the patch
        start = patch * self.patch_length
        end = start + self.patch_length
        magnitude = magnitude[np.newaxis, channel, :, start: end]

        # Neglect frequency to match model input
        if self.neglect_frequency == NYQUIST:
            magnitude = magnitude[:, : -1]
        elif self.neglect_frequency == ZERO:
            magnitude = magnitude[:, 1:]

        # Convert to tensor
        tensor = torch.from_numpy(magnitude)

        return tensor

if __name__ == "__main__":
    print('test')
    dataset = MagnitudeRandomDataset('train.csv', 30, 1024, 768, 128)