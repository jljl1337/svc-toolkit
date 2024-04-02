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