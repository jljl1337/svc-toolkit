import os
import shutil

from presenter.conversion import ConversionPresenter

class ConversionWidgetDummy:
    def set_device_list(self, devices):
        self.devices = devices

    def set_conversion_function(self, function):
        self.function = function

class ConverterDummy:
    input_path = None
    output_path = None
    speaker = None
    kwargs = None

    def convert(self, input_path, output_path, speaker, **kwargs):
        ConverterDummy.input_path = input_path
        ConverterDummy.output_path = output_path
        ConverterDummy.speaker = speaker
        ConverterDummy.kwargs = kwargs

class ConverterFactoryDummy:
    def create(self, model_path, config_path, device):
        self.model_path = model_path
        self.config_path = config_path
        self.device = device

        return ConverterDummy()

def test_conversion_presenter_constructor():
    # Create a ConversionWidget dummy object
    view = ConversionWidgetDummy()
    # Create a ConverterFactory dummy object
    model_factory = ConverterFactoryDummy()
    # Create a ConversionPresenter object
    presenter = ConversionPresenter(view, model_factory)

    # Check if the view is set
    assert presenter.view == view
    # Check if the model factory is set
    assert presenter.model_factory == model_factory

    # Check if the device list is set
    assert hasattr(view, 'devices')
    # Check if the conversion function is set
    assert view.function == presenter.convert

def test_conversion_presenter_convert():
    # Create a ConversionWidget dummy object
    view = ConversionWidgetDummy()
    # Create a ConverterFactory dummy object
    model_factory = ConverterFactoryDummy()
    # Create a ConversionPresenter object
    presenter = ConversionPresenter(view, model_factory)

    # Check if the converter is created with the correct parameters
    presenter.convert(
        'model_path',
        'config_path',
        'device',
        'input_path',
        'output_path',
        0,
        key1='value1',
    )
    assert model_factory.model_path == 'model_path'
    assert model_factory.config_path == 'config_path'
    assert model_factory.device == 'device'

    assert ConverterDummy.input_path == 'input_path'
    assert ConverterDummy.output_path == 'output_path'
    assert ConverterDummy.speaker == 0
    assert ConverterDummy.kwargs == {'key1': 'value1'}
