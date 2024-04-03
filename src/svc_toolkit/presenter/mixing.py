from svc_toolkit.widget.mixing import MixingWidget
from svc_toolkit.conversion.mixer import MixerFactory

class MixingPresenter:
    def __init__(self, view: MixingWidget, mixer_factory: MixerFactory):
        self.view = view
        self.mixer_factory = mixer_factory

        self.view.set_mixer_function(self.mix)

    def mix(
        self,
        source_1_path: str,
        source_2_path: str,
        output_path: str,
        source_1_ratio: float,
        **kwargs
    ):
        mixer = self.mixer_factory.create()
        mixer.mix(source_1_path, source_2_path, output_path, source_1_ratio, **kwargs)