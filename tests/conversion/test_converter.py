from conversion.converter import ConverterFactory, Converter

def test_converter_factory_create():
    # Create a ConverterFactory object
    factory = ConverterFactory()
    # Create a Converter object
    converter = factory.create('model_path', 'config_path', 'device')

    # Check if the model path is set
    assert converter.model_path == 'model_path'
    # Check if the config path is set
    assert converter.config_path == 'config_path'
    # Check if the device is set
    assert converter.device == 'device'

def test_converter_constructor():
    # Create a Converter object
    converter = Converter('model_path', 'config_path', 'device')

    # Check if the model path is set
    assert converter.model_path == 'model_path'
    # Check if the config path is set
    assert converter.config_path == 'config_path'
    # Check if the device is set
    assert converter.device == 'device'