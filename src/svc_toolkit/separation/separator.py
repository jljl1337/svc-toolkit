import os
import math

import numpy as np
import torch

from svc_toolkit.utility.functions import load_yaml
from svc_toolkit.separation import utility, models, audio, constants

class SeparatorFactory():
    def __init__(self) -> None:
        pass

    def create(self, model_dir, device, precision):
        return Separator(model_dir, device, precision)

class Separator():
    def __init__(self, model_dir, device, precision, last=False) -> None:
        if last:
            model_path = utility.get_last_checkpoint_path(model_dir)
        else:
            model_path = utility.get_best_checkpoint_path(model_dir)

        hparams_path = os.path.join(model_dir, 'hparams.yaml')
        config_path = os.path.join(model_dir, 'config.yml')

        config = load_yaml(config_path)
        self.sample_rate = config['sample_rate']
        self.window_length = config['win_length']
        self.hop_length = config['hop_length']
        self.patch_length = config['patch_length']
        self.neglect_frequency = config['neglect_frequency']

        self.model = models.UNetLightning.load_from_checkpoint(model_path, map_location=device, hparams_file=hparams_path)
        self.model.eval()
        self.device = device
        self.precision = precision

    def load_file(self, file):
        wave, _sr = audio.load(file, sr=self.sample_rate, mono=False)
        return wave

    def separate(self, wave, invert=False, emit=None):
        # Convert to 2D array if mono for convenience
        if wave.ndim == 1:
            wave = wave[np.newaxis, :]

        # Pad to fit segment length
        old_len = wave.shape[1]
        factor = self.patch_length * self.hop_length
        new_len = math.ceil((old_len + 1) / factor) * factor - 1
        diff = new_len - wave.shape[1]
        wave = np.pad(wave, ((0, 0), (0, diff)), mode='constant')

        # Separate spectrogram to magnitude and phase
        magnitude, phase = audio.to_mag_phase(wave, self.window_length, self.hop_length)

        # Normalize magnitude
        magnitude_max = magnitude.max()
        magnitude /= magnitude_max

        # Calculate segment number
        segment_num = magnitude.shape[-1] // self.patch_length
        total_segments = segment_num * magnitude.shape[0]

        for channel in range(magnitude.shape[0]):
            for segment_index in range(segment_num):
                # Extract segment
                start = segment_index * self.patch_length
                end = start + self.patch_length
                segment = magnitude[np.newaxis, channel, :, start: end]


                if self.neglect_frequency == constants.NYQUIST:
                    segment = segment[:, : -1]
                elif self.neglect_frequency == constants.ZERO:
                    segment = segment[:, 1:]

                segment_tensor = torch.from_numpy(segment)
                segment_tensor = torch.unsqueeze(segment_tensor, 0).to(self.device)

                # Predict mask
                with torch.no_grad():
                    if self.precision == 'bf16':
                        with torch.autocast(device_type=str(self.device), dtype=torch.bfloat16):
                            mask = self.model(segment_tensor)
                    elif self.precision == '32':
                        mask = self.model(segment_tensor)

                # Invert mask if needed
                if invert:
                    mask = 1 - mask

                # Apply mask
                masked = segment_tensor * mask

                # Save masked segment
                if self.neglect_frequency == constants.NYQUIST:
                    magnitude[channel, :-1, start: end] = masked.squeeze().cpu().numpy()
                elif self.neglect_frequency == constants.ZERO:
                    magnitude[channel, 1:, start: end] = masked.squeeze().cpu().numpy()

                # Update progress
                if emit is not None:
                    progress = (channel * segment_num + segment_index + 1) / total_segments * 100
                    emit(progress)

        # Denormalize magnitude
        magnitude *= magnitude_max

        # Reconstruct wave
        pre_wave = audio.to_wave(magnitude, phase, self.window_length, self.hop_length)

        # Remove padding
        pre_wave = pre_wave[:, :old_len]

        # Convert to 1D array if mono
        if pre_wave.shape[0] == 1:
            pre_wave = pre_wave[0]

        return pre_wave, self.sample_rate

    def separate_file(self, file, output_path, invert=False, emit=None):
        wave = self.load_file(file)
        new_wave, sample_rate = self.separate(wave, invert=invert, emit=emit)
        audio.save(output_path, new_wave.T, sample_rate)
