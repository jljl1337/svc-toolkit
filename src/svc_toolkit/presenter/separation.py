import os

from huggingface_hub import hf_hub_download

from svc_toolkit.utility.functions import load_yaml
from svc_toolkit.widget.separation import SeparationWidget
from svc_toolkit.separation.separator import SeparatorFactory
from svc_toolkit.presenter.common import get_available_device

VOCAL_FILE_NAME = 'vocal.wav'
NON_VOCAL_FILE_NAME = 'instrument.wav'

class SeparationPresenter:
    def __init__(self, view: SeparationWidget, model_factory: SeparatorFactory):
        self.view = view
        self.model_factory = model_factory
        self.manifest = self._get_manifest()

        self.view.set_model_list(self._get_model_list(self.manifest))
        self.view.set_device_list(get_available_device())
        self.view.set_separation_function(self.start_separation)

    def _get_manifest(self):
        manifest_path = os.path.join(os.path.dirname(__file__), '../models/manifest.yml')
        manifest = load_yaml(manifest_path)

        return manifest

    def _get_model_list(self, manifest):
        return [(model_name, model_name) for model_name in manifest['models']]

    def start_separation(self, emit, file, output_dir, vocal, non_vocal, model, device, precision):
        model_dir = self._get_model_dir(model)
        self._download_model_if_needed(model, model_dir)

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

    def _get_model_dir(self, model: str):
        return os.path.join(os.path.dirname(__file__), '../models', self.manifest['models'][model]['subfolder'])

    def _download_model_if_needed(self, model, model_dir):
        if os.path.exists(model_dir):
            return

        repo_id = self.manifest['models'][model]['repo_id']
        local_dir = os.path.dirname(model_dir)

        for asset in self.manifest['models'][model]['assets']:
            hf_hub_download(
                repo_id,
                self.manifest['models'][model]['assets'][asset],
                subfolder=self.manifest['models'][model]['subfolder'],
                local_dir=local_dir,
                local_dir_use_symlinks=False
            )

