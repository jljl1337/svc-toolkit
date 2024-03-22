import math

import numpy as np
import librosa
import soundfile as sf

def load(path, sr=None, mono=True, offset=0, duration=None):
    return librosa.load(path, sr=sr, mono=mono, offset=offset, duration=duration)

def save(path, data, sample_rate):
    sf.write(path, data, sample_rate)

def to_mag_phase(wave, win_length, hop_length):
    spectrogram = librosa.stft(wave, n_fft=win_length, hop_length=hop_length)
    return librosa.magphase(spectrogram)

def to_magnitude(wave, win_length, hop_length):
    return to_mag_phase(wave, win_length, hop_length)[0]

def to_wave(magnitude, phase, win_length, hop_length):
    spectrogram = magnitude * phase
    return librosa.istft(spectrogram, win_length=win_length, hop_length=hop_length)

def pad_wave(wave, hop_length, patch_length):
    factor = patch_length * hop_length
    new_len = math.ceil((wave.shape[1] + 1) / factor) * factor - 1
    diff = new_len - wave.shape[1]
    return np.pad(wave, ((0, 0), (0, diff)), mode='constant')