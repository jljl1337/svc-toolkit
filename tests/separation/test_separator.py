import os

import numpy as np

from svc_toolkit.separation.separator import Separator
from svc_toolkit.separation.constants import ZERO, NYQUIST

CURRENT_DIR = os.path.dirname(__file__)

class MockModel:
    def __call__(self, tensor):
        return tensor

class MockSeparator(Separator):
    def __init__(self, *args, **kwargs) -> None:
        self.model = MockModel()

def test_separator_load_file():
    separator = MockSeparator()

    separator.sample_rate = 44100

    wave = separator.load_file(os.path.join(CURRENT_DIR, '../source.wav'))

    assert isinstance(wave, np.ndarray)

def test_separator_separate():
    separator = MockSeparator()

    separator.sample_rate = 44100
    separator.window_length = 2048
    separator.hop_length = 512
    separator.patch_length = 128
    separator.neglect_frequency = ZERO
    separator.device = 'cpu'
    separator.precision = 'bf16'

    wave = np.ones((441000,))

    progress = None

    def emit(x):
        nonlocal progress
        progress = x

    magnitude, sr = separator.separate(wave, invert=True, emit=emit)

    assert isinstance(magnitude, np.ndarray)
    assert isinstance(sr, int)

    assert magnitude.shape == (441000,)
    assert progress is not None

    separator.neglect_frequency = NYQUIST
    separator.precision = '32'
    magnitude, sr = separator.separate(wave, invert=True, emit=emit)

def test_separator_separate_file():
    separator = MockSeparator()

    separator.sample_rate = 44100
    separator.window_length = 2048
    separator.hop_length = 512
    separator.patch_length = 128
    separator.neglect_frequency = ZERO
    separator.device = 'cpu'
    separator.precision = 'bf16'

    output_path = os.path.join(CURRENT_DIR, 'output.wav')

    separator.separate_file(os.path.join(CURRENT_DIR, '../source.wav'), output_path, invert=True)

    assert os.path.exists(output_path)

    os.remove(output_path)