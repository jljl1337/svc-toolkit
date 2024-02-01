import librosa
import soundfile as sf

def load(path, sr=None, mono=True, offset=0, duration=None):
    return librosa.load(path, sr=sr, mono=mono, offset=offset, duration=duration)

def save(path, data, sample_rate):
    sf.write(path, data, sample_rate)

def to_mag_phase(wave, win_length, hop_length):
    spectrogram = librosa.stft(wave, n_fft=win_length, hop_length=hop_length)
    return librosa.magphase(spectrogram)

def to_wave(magnitude, phase, win_length, hop_length):
    spectrogram = magnitude * phase
    return librosa.istft(spectrogram, win_length=win_length, hop_length=hop_length)