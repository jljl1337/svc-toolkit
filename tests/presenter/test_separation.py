import os
import shutil

from svc_toolkit.presenter.separation import SeparationPresenter

# Create a VocalSeparationWidget dummy class
class SeparationWidgetDummy:
    def set_model_list(self, model_list):
        self.model_list = model_list

    def set_device_list(self, device_list):
        self.device_list = device_list

    def set_separation_function(self, separation_function):
        self.separation_function = separation_function

# Create a Separator dummy class
class SeparatorDummy:
    def separate_file(self, file, output_path, invert=False, emit=lambda x: x):
        self.file = file
        self.output_path = output_path
        self.invert = invert
        self.emit = emit

# Create a SeparatorFactory dummy class
class SeparatorFactoryDummy:
    separator = SeparatorDummy()

    def create(self, model_dir, device, precision):
        self.model_dir = model_dir
        self.device = device
        self.precision = precision

        return self.separator

def test_separation_presenter_get_manifest():
    # Create a VocalSeparationWidget dummy object
    view = SeparationWidgetDummy()
    # Create a SeparatorFactory dummy object
    model_factory = SeparatorFactoryDummy()
    # Create a VocalSeparationPresenter object
    presenter = SeparationPresenter(view, model_factory)

    # Check if can load the manifest
    assert 'models' in presenter._get_manifest()

def test_separation_presenter_get_model_list():
    # Create a VocalSeparationWidget dummy object
    view = SeparationWidgetDummy()
    # Create a SeparatorFactory dummy object
    model_factory = SeparatorFactoryDummy()
    # Create a VocalSeparationPresenter object
    presenter = SeparationPresenter(view, model_factory)

    # Check if number of keys in the manifest is equal to the number of models
    dummy_manifest = {'models': {'model1': {}, 'model2': {}}}
    assert len(presenter._get_model_list(dummy_manifest)) == len(dummy_manifest['models'])

# Test for the VocalSeparationPresenter class constructor with dummy class
def test_separation_presenter_constructor():
    # Create a VocalSeparationWidget dummy object
    view = SeparationWidgetDummy()
    # Create a SeparatorFactory dummy object
    model_factory = SeparatorFactoryDummy()
    # Create a VocalSeparationPresenter object
    presenter = SeparationPresenter(view, model_factory)

    # Check if the model list is set
    assert len(view.model_list) > 0
    # Check if the device list is set
    assert view.device_list[-1] == ('CPU', 'cpu')
    # Check if the separation function is set
    assert view.separation_function == presenter.start_separation
    # Check if the model factory is set
    assert presenter.model_factory == model_factory
    # Check if the manifest is set
    assert type(presenter.manifest) == dict

def test_separation_presenter_get_model_dir():
    # Create a VocalSeparationWidget dummy object
    view = SeparationWidgetDummy()
    # Create a SeparatorFactory dummy object
    model_factory = SeparatorFactoryDummy()
    # Create a VocalSeparationPresenter object
    presenter = SeparationPresenter(view, model_factory)

    # Create a dummy model
    model = 'dummy_model'
    # Set the manifest
    presenter.manifest = {'models': {'dummy_model': {'subfolder': 'dummy_subfolder'}}}

    # Check if the model directory is set correctly
    assert presenter._get_model_dir(model).endswith('dummy_subfolder')

def test_separation_presenter_download_model_if_needed():
    # Create a VocalSeparationWidget dummy object
    view = SeparationWidgetDummy()
    # Create a SeparatorFactory dummy object
    model_factory = SeparatorFactoryDummy()
    # Create a VocalSeparationPresenter object
    presenter = SeparationPresenter(view, model_factory)

    # Create a dummy model
    model = 'Test Model'
    # Create a dummy model directory
    model_dir = os.path.join(os.path.dirname(__file__), 'test_model')
    # Set the manifest
    presenter.manifest = {
        'models': {
            'Test Model': {
                'repo_id': 'jljl1337/svc-toolkit-test',
                'subfolder': 'test_model',
                'assets': {
                    'model': 'model.tmp',
                    'config': 'config.tmp'
                }
            }
        }
    }

    # Remove the model directory if it exists
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)

    # Call the download_model_if_needed function
    presenter._download_model_if_needed(model, model_dir)

    # Check if the model directory is created
    assert os.path.exists(model_dir)
    # Check if the files are downloaded
    assert os.path.exists(os.path.join(model_dir, 'model.tmp'))
    assert os.path.exists(os.path.join(model_dir, 'config.tmp'))

    # Get model file last modified time
    model_file_time = os.path.getmtime(os.path.join(model_dir, 'model.tmp'))

    # Call again the download_model_if_needed function
    presenter._download_model_if_needed(model, model_dir)

    # Check if the model file is not downloaded again
    assert model_file_time == os.path.getmtime(os.path.join(model_dir, 'model.tmp'))

    # Remove the model directory
    shutil.rmtree(model_dir)
    
def test_vocal_separation_presenter_start_separation():
    # Create a VocalSeparationWidget dummy object
    view = SeparationWidgetDummy()
    # Create a SeparatorFactory dummy object
    model_factory = SeparatorFactoryDummy()
    # Create a VocalSeparationPresenter object
    presenter = SeparationPresenter(view, model_factory)

    # Create a dummy emit function
    def emit(value):
        return value

    # Create a dummy file
    file = 'dummy_file'
    # Create a dummy output directory
    output_dir = 'dummy_output_dir'
    # Create a dummy vocal flag
    vocal = True
    # Create a dummy non-vocal flag
    non_vocal = True
    # Create a dummy model
    model = 'Test'
    # Create a dummy device
    device = 'dummy_device'
    # Create a dummy precision
    precision = 'dummy_precision'
    # Set the manifest
    presenter.manifest = {
        'models': {
            'Test': {
                'repo_id': 'jljl1337/svc-toolkit-test',
                'subfolder': 'test_model',
                'assets': {
                    'model': 'model.tmp',
                    'config': 'config.tmp'
                }
            }
        }
    }

    # Call the start_separation function
    presenter.start_separation(emit, file, output_dir, vocal, non_vocal, model, device, precision)

    # Check if the model directory is set correctly
    assert model_factory.model_dir.endswith('test_model')
    # Check if the device is set correctly
    assert model_factory.device == device
    # Check if the file is set correctly
    assert model_factory.separator.file == file

    # Remove the model directory
    print(os.path.dirname(__file__))
    shutil.rmtree(os.path.join(os.path.dirname(__file__), '../models', model_factory.model_dir))
