import torch
import torchaudio
import numpy as np
import librosa
import soundfile as sf
from scipy.signal.windows import hann

def load(path, offset=0, duration=-1):
    sr = sf.info(path).samplerate
    data, sr = sf.read(path, start=int(offset*sr), frames=int(duration*sr),
                       dtype='float32')
    return data, sr

def save(path, data, sample_rate):
    sf.write(path, data, sample_rate)

def duration(path):
    return sf.info(path).duration

def resample(data, source_sr, target_sr):
    data = torchaudio.transforms.Resample(source_sr, target_sr)(data)
    return data

def to_mag_phase(mix_wave, stem_wave, win_length, hop_length):
    mix_spectrogram = librosa.stft(mix_wave, n_fft=win_length, hop_length=hop_length)
    stem_spectrogram = librosa.stft(stem_wave, n_fft=win_length, hop_length=hop_length)
    mix_magnitude, mix_phase = librosa.magphase(mix_spectrogram)
    stem_magnitude, stem_phase = librosa.magphase(stem_spectrogram)

    mix_mag_max = mix_magnitude.max()
    mix_magnitude /= mix_mag_max
    stem_magnitude /= mix_mag_max

    return mix_magnitude, mix_phase, stem_magnitude, stem_phase

def to_wave(magnitude, phase, mix_spectrogram, win_length, hop_length):
    spectrogram = magnitude * phase
    spectrogram *= mix_spectrogram.max()
    return librosa.istft(spectrogram, win_length=win_length, hop_length=hop_length)