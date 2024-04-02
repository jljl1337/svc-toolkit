import os
import numpy as np

from separation.audio import load, save, to_mag_phase, to_magnitude, to_wave, pad_wave

CURRENT_DIR = os.path.dirname(__file__)

def test_load():
    # Load an audio file
    data, sample_rate = load(os.path.join(CURRENT_DIR, '../source.wav'))

    # Check if the data is np.ndarray
    assert isinstance(data, np.ndarray)

    # Check if the sample rate is int
    assert isinstance(sample_rate, int)

def test_save():
    # Generate a random audio file
    data = np.random.randn(44100)
    save(os.path.join(CURRENT_DIR, 'test_save.wav'), data, 44100)

    # Check if the file is created
    assert os.path.exists(os.path.join(CURRENT_DIR, 'test_save.wav'))

    os.remove(os.path.join(CURRENT_DIR, 'test_save.wav'))

def test_to_mag_phase():
    # Generate a random audio file
    data = np.random.randn(44100)

    # Convert the audio file to magnitude and phase
    mag, phase = to_mag_phase(data, 2048, 512)

    # Check if the shape of magnitude is correct
    assert mag.shape == (1025, 87)

    # Check if the shape of phase is correct
    assert phase.shape == (1025, 87)

def test_to_magnitude():
    # Generate a random audio file
    data = np.random.randn(44100)

    # Convert the audio file to magnitude
    mag = to_magnitude(data, 2048, 512)

    # Check if the shape of magnitude is correct
    assert mag.shape == (1025, 87)

def test_to_wave():
    # Generate a random audio file
    data = np.random.randn(45055)

    # Convert the audio file to magnitude and phase
    mag, phase = to_mag_phase(data, 2048, 512)

    # Convert the magnitude and phase back to audio
    wave = to_wave(mag, phase, 2048, 512)

    # Check if the audio is 1D
    assert wave.ndim == 1

def test_pad_wave():
    # Generate a random audio file
    data = np.random.randn(1, 44100)

    # Pad the audio file
    padded_data = pad_wave(data, 512, 8)

    # Check if the shape of the padded audio is correct
    assert padded_data.shape == (1, 45055)