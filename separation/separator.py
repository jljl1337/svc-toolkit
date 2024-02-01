import os
import math
import numpy as np
import torch

import utility
import models
import audio

class Separator():
    def __init__(self, model_dir, device) -> None:
        model_path = ''
        for file in os.listdir(model_dir):
            if file.startswith('best') and file.endswith('.ckpt'):
                model_path = os.path.join(model_dir, file)
                break

        hparams_path = os.path.join(model_dir, 'hparams.yaml')
        config_path = os.path.join(model_dir, 'config.yml')

        config = utility.load_yaml(config_path)
        self.sample_rate = config['sample_rate']
        self.window_length = config['win_length']
        self.hop_length = config['hop_length']
        self.patch_length = config['patch_length']

        self.model = models.UNetLightning.load_from_checkpoint(model_path, map_location=device, hparams_file=hparams_path)
        self.model.eval()
        self.device = device

    def load_file(self, file):
        wave, _sr = audio.load(file, sr=self.sample_rate, mono=False)
        return wave

    def separate(self, wave):
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

        for channel in range(magnitude.shape[0]):
            for segment_index in range(segment_num):
                # Extract segment
                start = segment_index * self.patch_length
                end = start + self.patch_length
                segment = magnitude[np.newaxis, channel, :-1, start: end]
                segment_tensor = torch.from_numpy(segment)
                segment_tensor = torch.unsqueeze(segment_tensor, 0).to(self.device)

                # Predict mask
                with torch.no_grad():
                    mask = self.model(segment_tensor)

                # Apply mask
                masked = segment_tensor * mask
                # masked = segment_tensor * (1 - mask)

                # Save masked segment
                magnitude[channel, :-1, start: end] = masked.squeeze().cpu().numpy()

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

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    separator = Separator('/home/jljl1337/git/singing-voice-conversion-gui/model/all_deeper/20240131_041926', device)
    wave = separator.load_file('/home/jljl1337/dataset/musdb18hq/test/Al James - Schoolboy Facination/mixture.wav')
    new_wave, sample_rate = separator.separate(wave)
    audio.save('test123.wav', new_wave.T, sample_rate)

if __name__ == "__main__":
    main()