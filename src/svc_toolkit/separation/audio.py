import math

import numpy as np
import librosa
import soundfile as sf

def load(path, sr=None, mono=True, offset=0, duration=None) -> np.ndarray:
    return librosa.load(path, sr=sr, mono=mono, offset=offset, duration=duration)

def save(path, data: np.ndarray, sample_rate: int) -> None:
    sf.write(path, data, sample_rate)

def to_mag_phase(wave: np.ndarray, win_length: int, hop_length: int) -> tuple[np.ndarray, np.ndarray]:
    spectrogram = librosa.stft(wave, n_fft=win_length, hop_length=hop_length)
    return librosa.magphase(spectrogram)

def to_magnitude(wave: np.ndarray, win_length: int, hop_length: int) -> np.ndarray:
    return to_mag_phase(wave, win_length, hop_length)[0]

def to_wave(magnitude: np.ndarray, phase: np.ndarray, win_length: int, hop_length: int) -> np.ndarray:
    spectrogram = magnitude * phase
    return librosa.istft(spectrogram, win_length=win_length, hop_length=hop_length)

def pad_wave(wave: np.ndarray, hop_length: int, patch_length: int) -> np.ndarray:
    factor = patch_length * hop_length
    new_len = math.ceil((wave.shape[1] + 1) / factor) * factor - 1
    diff = new_len - wave.shape[1]
    return np.pad(wave, ((0, 0), (0, diff)), mode='constant')