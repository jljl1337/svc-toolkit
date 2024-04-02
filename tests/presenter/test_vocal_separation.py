from presenter.separation import VocalSeparationPresenter

# Create a VocalSeparationWidget dummy class
class VocalSeparationWidgetDummy:
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

    def create(self, model, device):
        self.model = model
        self.device = device

        return self.separator

# Test for the VocalSeparationPresenter class constructor with dummy class
def test_vocal_separation_presenter_constructor():
    vocal_separation_widget = VocalSeparationWidgetDummy()
    separator_factory = SeparatorFactoryDummy()
    vocal_separation_presenter = VocalSeparationPresenter(vocal_separation_widget, separator_factory)

    assert vocal_separation_widget.model_list == vocal_separation_presenter._get_model_list()
    assert vocal_separation_widget.device_list[-1] == ('CPU', 'cpu')
    assert vocal_separation_widget.separation_function == vocal_separation_presenter.start_separation

# Test for start_separation method with vocal only
def test_vocal_separation_presenter_start_separation():
    vocal_separation_widget = VocalSeparationWidgetDummy()
    separator_factory = SeparatorFactoryDummy()
    vocal_separation_presenter = VocalSeparationPresenter(vocal_separation_widget, separator_factory)
    vocal_separation_presenter.start_separation(lambda x: x, 'file', 'output_dir', True, False, 'model', 'device')

    assert separator_factory.model == 'model'
    assert separator_factory.device == 'device'

    assert separator_factory.separator.file == 'file'
    assert separator_factory.separator.output_path == 'output_dir/vocal.wav'
    assert separator_factory.separator.invert == False
    assert separator_factory.separator.emit(100) == 100
    assert separator_factory.separator.emit(50) == 50

# Test for start_separation method with non vocal only
def test_vocal_separation_presenter_start_separation_non_vocal():
    vocal_separation_widget = VocalSeparationWidgetDummy()
    separator_factory = SeparatorFactoryDummy()
    vocal_separation_presenter = VocalSeparationPresenter(vocal_separation_widget, separator_factory)
    vocal_separation_presenter.start_separation(lambda x: x, 'file', 'output_dir', False, True, 'model', 'device')

    assert separator_factory.model == 'model'
    assert separator_factory.device == 'device'

    assert separator_factory.separator.file == 'file'
    assert separator_factory.separator.output_path == 'output_dir/instrument.wav'
    assert separator_factory.separator.invert == True
    assert separator_factory.separator.emit(100) == 100
    assert separator_factory.separator.emit(50) == 50

# Test for start_separation method with vocal and non vocal
def test_vocal_separation_presenter_start_separation_vocal_non_vocal():
    vocal_separation_widget = VocalSeparationWidgetDummy()
    separator_factory = SeparatorFactoryDummy()
    vocal_separation_presenter = VocalSeparationPresenter(vocal_separation_widget, separator_factory)
    vocal_separation_presenter.start_separation(lambda x: x, 'file', 'output_dir', True, True, 'model', 'device')

    assert separator_factory.model == 'model'
    assert separator_factory.device == 'device'

    assert separator_factory.separator.file == 'file'
    assert separator_factory.separator.output_path == 'output_dir/instrument.wav'
    assert separator_factory.separator.invert == True
    assert separator_factory.separator.emit(100) == 100
    assert separator_factory.separator.emit(50) == 75