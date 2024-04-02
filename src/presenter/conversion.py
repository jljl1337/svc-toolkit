from widget.conversion import VoiceConversionWidget
from conversion.voice_converter import VoiceConverterFactory
from presenter.common import get_available_device

class VoiceConversionPresenter:
    def __init__(self, view: VoiceConversionWidget, model_factory: VoiceConverterFactory):
        self.view = view
        self.model_factory = model_factory

        self.view.set_device_list(get_available_device())
        self.view.set_conversion_function(self.convert)

    def convert(
        self, 
        model_path: str, 
        config_path: str, 
        device: str, 
        input_path: str,
        output_path: str,
        speaker: int,
        **kwargs
    ):
        converter = self.model_factory.create(model_path, config_path, device)
        converter.convert(input_path, output_path, speaker, **kwargs)