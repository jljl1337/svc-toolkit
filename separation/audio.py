import torch
import torchaudio
import numpy as np
import librosa
import soundfile as sf
from scipy.signal.windows import hann

# import separation.contants as contants
import constants

def load(path, offset=0, duration=-1):
    sr = sf.info(path).samplerate
    data, sr = sf.read(path, start=int(offset*sr), frames=int(duration*sr),
                       dtype='float32')
    return data, sr

def save(path, data, sr=constants.SAMPLE_RATE):
    sf.write(path, data, sr)
    print('save')

def duration(path):
    return sf.info(path).duration

def resample(data, source_sr, target_sr=constants.SAMPLE_RATE):
    data = torchaudio.transforms.Resample(source_sr, target_sr)(data)
    return data

def pre_stft(data, sample_rate):
    data = data.T
    if sample_rate != constants.SAMPLE_RATE:
        data = resample(data, sample_rate)
    channel = data.shape[0]
    if channel == 1:
        data = torch.cat((data, data), dim=0)
    elif channel > 2:
        data = data[:2, :]
    return data.T
    
def stft(data, inverse=False, length=None):
    data = np.asfortranarray(data)
    channel = data.shape[-1]
    window = hann(constants.FRAME_LENGTH, False)
    output = []

    for i in range(channel):
        if not inverse:
            transformed = data[:, i]
            transformed = librosa.core.stft(
                transformed,
                hop_length=constants.FRAME_STEP,
                window=window,
                center=False,
                n_fft=constants.FRAME_LENGTH,
            )
            transformed = np.expand_dims(transformed.T, axis=2)
        else:
            transformed = data[:, :, i].T
            transformed = librosa.core.istft(
                transformed,
                hop_length=constants.FRAME_STEP,
                window=window,
                center=False,
                win_length=None,
                length=length,
            )
            transformed = np.expand_dims(transformed.T, axis=1)
        output.append(transformed)

    if channel == 1:
        return output[0]
    return np.concatenate(output, axis=1 if inverse else 2)