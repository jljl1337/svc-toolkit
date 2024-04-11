import numpy as np
import soundfile as sf
from librosa import load

from so_vits_svc_fork.inference.main import infer

class ConverterFactory:
    def __init__(self) -> None:
        pass

    def create(self, model_path: str, config_path: str, device: str):
        return Converter(model_path, config_path, device)

class Converter:
    def __init__(self, model_path: str, config_path: str, device: str):
        self.model_path = model_path
        self.config_path = config_path
        self.device = device

    def convert(self, input_path: str, output_path: str, speaker: int, **kwargs):
        infer(
            input_path=input_path,
            output_path=output_path,
            model_path=self.model_path,
            config_path=self.config_path,
            speaker=speaker,
            device=self.device,
            **kwargs
        )

        input_wave, _input_sr = load(input_path, mono=False, sr=None)

        # Get number of channels
        num_channels = input_wave.shape[0]

        # Load the output file
        output_wave, _output_sr = load(output_path, mono=False, sr=None)

        # Set a ndarray with zeros to the same shape as the input
        new_output_wave = np.zeros_like(input_wave)

        # Save the output file with the same number of channels
        for i in range(num_channels):
            new_output_wave[i] = output_wave

        # Save the output file with the same number of channels
        sf.write(output_path, new_output_wave.T, _output_sr)
