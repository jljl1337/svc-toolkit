import os

from widget.vocal_separation import VocalSeparationWidget
from separation.separator import SeparatorFactory
from presenter.common import get_available_device

VOCAL_FILE_NAME = 'vocal.wav'
NON_VOCAL_FILE_NAME = 'instrument.wav'

class VocalSeparationPresenter:
    def __init__(self, view: VocalSeparationWidget, model_factory: SeparatorFactory):
        self.view = view
        self.model_factory = model_factory

        self.view.set_model_list(self._get_model_list())
        self.view.set_device_list(get_available_device())
        self.view.set_separation_function(self.start_separation)

    def _get_model_list(self):
        return [('Small', 'model/small')]

    def start_separation(self, emit, file, output_dir, vocal, non_vocal, model_dir, device, precision):
        separator = self.model_factory.create(model_dir, device, precision)
        vocal_file_path = os.path.join(output_dir, VOCAL_FILE_NAME)
        non_vocal_file_path = os.path.join(output_dir, NON_VOCAL_FILE_NAME)

        fn = emit
        fn2 = emit

        if vocal and non_vocal:
            fn = lambda x: emit(x / 2)
            fn2 = lambda x: emit(50 + x / 2)

        if vocal:
            separator.separate_file(file, vocal_file_path, emit=fn)
        if non_vocal:
            separator.separate_file(file, non_vocal_file_path, invert=True, emit=fn2)
