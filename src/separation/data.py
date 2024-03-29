import math
from typing import Iterable
from multiprocessing import cpu_count
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
import numpy as np
import pytorch_lightning as pl
from pandas import read_csv
from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.trainer.states import TrainerFn
from tqdm import tqdm

from separation.audio import load, to_magnitude, to_mag_phase, to_wave, pad_wave
from separation.constants import CSV_MIXTURE_PATH_COLUMN, CSV_STEM_PATH_COLUMN, NYQUIST, ZERO, NEGLECT_FREQUENCY_OPTIONS

class MagnitudeRandomDataset(Dataset):
    def __init__(
        self,
        mixture_path_list: list[str],
        stem_path_list: list[str],
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

        # Check if the length of the lists are equal
        if not len(mixture_path_list) == len(stem_path_list):
            raise ValueError('Length of mixture and stem lists must be equal')

        self.song_num = len(mixture_path_list)

        # Save parameters
        self.win_length = win_length
        self.hop_length = hop_length
        self.patch_length = patch_length
        self.expand_factor = expand_factor
        self.neglect_frequency = neglect_frequency
        self.sample_rate = sample_rate

        # Set up magnitudes
        self.magnitudes = [None] * self.song_num
        self.expanded_magnitudes = []

        # Load magnitudes in parallel
        with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            # Submit tasks
            zipped = zip(range(self.song_num), mixture_path_list, stem_path_list)
            futures = [executor.submit(self.load_magnitude, index, mixture_path, stem_path) for index, mixture_path, stem_path in zipped]

            # For each finished task
            for future in tqdm(as_completed(futures), total=self.song_num):
                index, mix_magnitude, stem_magnitude, weight = future.result()

                # Save magnitude
                self.magnitudes[index] = (mix_magnitude, stem_magnitude)
                self.expanded_magnitudes.extend([index] * weight)

            # Sort expanded magnitudes since they are not executed in order
            self.expanded_magnitudes = sorted(self.expanded_magnitudes)

    def load_magnitude(self, index, mixture_path, stem_path):
        # Load audio
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
        self.paths = paths
        self.win_length = win_length
        self.hop_length = hop_length
        self.patch_length = patch_length
        self.neglect_frequency = neglect_frequency
        self.sample_rate = sample_rate

        self._setup()

    def _setup(self):
        # Set up magnitudes
        self.magnitude_list = [None] * len(self.paths)
        self.phase_list = [None] * len(self.paths)
        self.magnitude_max_list = [None] * len(self.paths)
        self.old_length_list = [None] * len(self.paths)
        self.index_channel_patch_list = []

        # Load magnitudes in parallel
        with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
            # Submit tasks
            futures = [executor.submit(self._load_magnitude, index, path) for index, path in enumerate(self.paths)]

            # For each finished task
            for future in tqdm(as_completed(futures), total=len(self.paths)):
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

        self.need_setup = False

    def _load_magnitude(self, index: int, path: str) -> tuple[int, np.ndarray, np.ndarray, float, int, int, int]:
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

    def get_wave_list(self, mask: np.ndarray, invert=False):
        if self.need_setup:
            self._setup()

        if mask.shape != (len(self.index_channel_patch_list), self.win_length / 2, self.patch_length):
            raise ValueError(f'Invalid mask shape: {mask.shape}')

        # Invert mask if needed
        if invert:
            mask = 1 - mask

        max_workers = min(cpu_count(), 4)

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._get_song_wave, index, mask) for index in range(len(self.paths))]

            result_list = [future.result() for future in tqdm(as_completed(futures), total=len(self.paths))]

            self.index_channel_patch_list = None
            self.need_setup = True

            return sorted(result_list)

    def _get_song_wave(self, index: int, mask: np.ndarray):
        for i in (range(len(self.index_channel_patch_list))):
            actual_index, channel, patch = self.index_channel_patch_list[i]
            
            if index != actual_index:
                continue
            
            start = patch * self.patch_length
            end = start + self.patch_length

            if self.neglect_frequency == NYQUIST:
                # result_magnitude[index][channel, :-1, start: end] = mask[i]
                self.magnitude_list[index][channel, :-1, start: end] = mask[i]
            elif self.neglect_frequency == ZERO:
                # result_magnitude[index][channel, 1:, start: end] = mask[i]
                self.magnitude_list[index][channel, 1:, start: end] = mask[i]

        self.magnitude_list[index] *= self.magnitude_max_list[index]

        # wave = to_wave(result_magnitude[i], self.phase_list[i], self.win_length, self.hop_length)
        wave = to_wave(self.magnitude_list[index], self.phase_list[index], self.win_length, self.hop_length)

        old_length = self.old_length_list[index]
        wave = wave[:, :old_length]

        self.magnitude_list[index] = None
        self.phase_list[index] = None
        self.magnitude_max_list[index] = None
        self.old_length_list[index] = None

        return index, wave

class MagnitudeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_csv: str = None,
        val_csv: str = None,
        predict_path_list: Iterable[str] = None,
        expand_factor: float = None,
        win_length: int = None,
        hop_length: int = None,
        patch_length: int = None,
        neglect_frequency: str = None,
        sample_rate: int = None,
        batch_size: int = None,
        loader_num_workers: int = None,
    ):
        super().__init__()

        self.train_csv = train_csv
        self.val_csv = val_csv
        self.predict_path_list = predict_path_list
        self.expand_factor = expand_factor
        self.win_length = win_length
        self.hop_length = hop_length
        self.patch_length = patch_length
        self.neglect_frequency = neglect_frequency
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        self.loader_num_workers = loader_num_workers

        self.train_dataset = None
        self.val_dataset = None
        self.predict_dataset = None

        self.dataset_kwargs = {
            'expand_factor': expand_factor,
            'win_length': win_length,
            'hop_length': hop_length,
            'patch_length': patch_length,
            'neglect_frequency': neglect_frequency,
            'sample_rate': sample_rate
        }

    def setup(self, stage: str = None):
        if stage == TrainerFn.FITTING and self.train_dataset is None and self.val_dataset is None:
            print('Loading training datasets')
            df_train = read_csv(self.train_csv)
            self.train_dataset = MagnitudeRandomDataset(
                df_train[CSV_MIXTURE_PATH_COLUMN].tolist(),
                df_train[CSV_STEM_PATH_COLUMN].tolist(),
                **self._dataset_kwargs(fit=True)
            )

            print('Loading validation datasets')
            df_val = read_csv(self.val_csv)
            self.val_dataset = MagnitudeRandomDataset(
                df_val[CSV_MIXTURE_PATH_COLUMN].tolist(),
                df_val[CSV_STEM_PATH_COLUMN].tolist(),
                **self._dataset_kwargs(fit=True)
            )

        elif stage == TrainerFn.PREDICTING and self.predict_dataset is None:
            print('Loading prediction datasets')
            self.predict_dataset = MagnitudeDataset(
                self.predict_path_list,
                **self._dataset_kwargs(fit=False)
            )

    def _dataset_kwargs(self, fit: bool):
        kwargs_dict = {
            'win_length': self.win_length,
            'hop_length': self.hop_length,
            'patch_length': self.patch_length,
            'neglect_frequency': self.neglect_frequency,
            'sample_rate': self.sample_rate
        }

        if fit:
            kwargs_dict['expand_factor'] = self.expand_factor

        return kwargs_dict

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            **self._loader_kwargs(shuffle=True),
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            **self._loader_kwargs(shuffle=False),
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            **self._loader_kwargs(shuffle=False),
        )

    def _loader_kwargs(self, shuffle: bool):
        return {
            'batch_size': self.batch_size,
            'num_workers': self.loader_num_workers,
            'persistent_workers': True,
            'pin_memory': True,
            'shuffle': shuffle,
        }

    def get_wave_list(self, mask: np.ndarray, invert=False):
        self.predict_dataset.get_wave_list(mask, invert)

if __name__ == "__main__":
    print('test')
    dataset = MagnitudeRandomDataset('train.csv', 30, 1024, 768, 128)