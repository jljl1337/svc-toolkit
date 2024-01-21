import torch
import torchaudio
import numpy as np
import librosa
import soundfile as sf
from scipy.signal.windows import hann

def load(path, sr=None, offset=0, duration=None):
    return librosa.load(path, sr=sr, offset=offset, duration=duration)

def save(path, data, sample_rate):
    sf.write(path, data, sample_rate)

def duration(path):
    return sf.info(path).duration

def resample(data, source_sr, target_sr):
    data = torchaudio.transforms.Resample(source_sr, target_sr)(data)
    return data

def to_mag_phase(wave, win_length, hop_length):
    spectrogram = librosa.stft(wave, n_fft=win_length, hop_length=hop_length)
    return librosa.magphase(spectrogram)

# TODO: Too high level?
def mix_stem_to_mag_phase(mix_wave, stem_wave, win_length, hop_length):
    mix_magnitude, mix_phase = to_mag_phase(mix_wave, win_length, hop_length)
    stem_magnitude, stem_phase = to_mag_phase(stem_wave, win_length, hop_length)

    mix_mag_max = mix_magnitude.max()
    mix_magnitude /= mix_mag_max
    stem_magnitude /= mix_mag_max

    return mix_magnitude, mix_phase, stem_magnitude, stem_phase, mix_mag_max

def to_wave(magnitude, phase, win_length, hop_length):
    spectrogram = magnitude * phase
    return librosa.istft(spectrogram, win_length=win_length, hop_length=hop_length)

# TODO: Too high level?
def stem_to_wave(magnitude, phase, win_length, hop_length, mag_max):
    magnitude *= mag_max
    return to_wave(magnitude, phase, win_length, hop_length)