from librosa import load
import numpy as np
import soundfile as sf

class MixerFactory:
    def __init__(self) -> None:
        pass

    def create(self):
        return Mixer()

class Mixer:
    def __init__(self) -> None:
        pass

    def mix(self, source_1_path: str, source_2_path: str, output_path: str, source_1_ratio: float, normalize: bool = False):
        wave_1, sr1 = load(source_1_path, mono=False, sr=None)
        wave_2, sr2 = load(source_2_path, mono=False, sr=None)

        if sr1 != sr2:
            raise ValueError('Sampling rate mismatch')
        
        if wave_1.shape != wave_2.shape:
            raise ValueError('Shape mismatch')

        if source_1_ratio < 0 or source_1_ratio > 1:
            raise ValueError('Ratio must be between 0 and 1')
        
        # Mix
        mixed_wave = wave_1 * source_1_ratio + wave_2 * (1 - source_1_ratio)

        # Normalize if necessary
        if normalize:
            mixed_wave /= np.max(np.abs(mixed_wave))

        # Save
        sf.write(output_path, mixed_wave.T, sr1)
