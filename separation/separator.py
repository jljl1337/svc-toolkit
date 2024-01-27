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
        self.model = models.UNetLightning.load_from_checkpoint(model_path)
        self.model.to(device)
        self.model.eval()
        self.device = device

    def load_file(self, file, sample_rate):
        wave, sr = audio.load(file, mono=False)

        if sr != sample_rate:
            wave = audio.resample(wave, sr, sample_rate)

        return wave

    def separate(self, wave, window_length, hop_length, patch_length):
        # Convert to 2D array if mono for convenience
        if wave.ndim == 1:
            wave = wave[np.newaxis, :]

        # Pad to fit segment length
        old_len = wave.shape[1]
        factor = patch_length * hop_length
        new_len = math.ceil((old_len + 1) / factor) * factor - 1
        diff = new_len - wave.shape[1]
        wave = np.pad(wave, ((0, 0), (0, diff)), mode='constant')

        # Separate spectrogram to magnitude and phase
        magnitude, phase = audio.to_mag_phase(wave, window_length, hop_length)

        # Normalize magnitude
        magnitude_max = magnitude.max()
        magnitude /= magnitude_max

        # Calculate segment number
        segment_num = magnitude.shape[-1] // patch_length

        for channel in range(magnitude.shape[0]):
            for segment_index in range(segment_num):
                # Extract segment
                start = segment_index * patch_length
                end = start + patch_length
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
        pre_wave = audio.to_wave(magnitude, phase, window_length, hop_length)

        # Remove padding
        pre_wave = pre_wave[:, :old_len]

        # Convert to 1D array if mono
        if pre_wave.shape[0] == 1:
            pre_wave = pre_wave[0]

        return pre_wave

def main():
    load_dotenv()
    MODEL_DIR = os.getenv('MODEL_DIR')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    separator = Separator('model/test4096/20240127_010925/best-epoch=296.ckpt', device)
    wave = separator.load_file('/home/jljl1337/dataset/musdb18hq/test/Al James - Schoolboy Facination/mixture.wav', constants.SAMPLE_RATE)
    new_wave = separator.separate(wave, 4096, 1024, 512)
    audio.save('testlast.wav', new_wave.T, constants.SAMPLE_RATE)

if __name__ == "__main__":
    main()