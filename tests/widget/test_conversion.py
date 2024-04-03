import os

from svc_toolkit.widget.conversion import ConversionThread, ConversionWidget

def test_conversion_thread_run(qtbot):
    def test_function():
        pass

    conversion_thread = ConversionThread(test_function, {'key1': 'value1'})

    conversion_thread.run()

    assert conversion_thread.isRunning() == False

def test_conversion_widget_constructor(qtbot):
    conversion_widget = ConversionWidget()
    qtbot.addWidget(conversion_widget)

    assert conversion_widget.model_widget.label.text() == 'Model: Not chosen'

def test_conversion_widget_toggle_advanced_settings(qtbot):
    conversion_widget = ConversionWidget()
    qtbot.addWidget(conversion_widget)

    assert conversion_widget.advanced_settings_widget.isHidden() == True

    conversion_widget._show_advanced_settings()

    assert conversion_widget.advanced_settings_widget.isHidden() == False

    conversion_widget._close_advanced_settings()

    assert conversion_widget.advanced_settings_widget.isHidden() == True

def test_conversion_widget_set_speaker_list(qtbot):
    current_dir = os.path.dirname(__file__)

    conversion_widget = ConversionWidget()
    qtbot.addWidget(conversion_widget)

    conversion_widget._set_speaker_list(os.path.join(current_dir, 'test_config.json'))

    assert conversion_widget.speaker_dropdown.dropdown.currentText() == 'Test Speaker'

def test_conversion_widget_set_device_list(qtbot):
    conversion_widget = ConversionWidget()
    qtbot.addWidget(conversion_widget)

    conversion_widget.set_device_list([('CPU', 'cpu'), ('GPU', 'gpu')])

    assert conversion_widget.device_dropdown.dropdown.currentText() == 'CPU'

def test_conversion_widget_set_conversion_function(qtbot):
    conversion_widget = ConversionWidget()
    qtbot.addWidget(conversion_widget)

    def test_function():
        pass

    conversion_widget.set_conversion_function(test_function)

    assert conversion_widget.conversion_function == test_function

def test_conversion_widget_start_conversion(qtbot):
    conversion_widget = ConversionWidget()
    qtbot.addWidget(conversion_widget)

    conversion_widget.conversion_function = lambda: None
    conversion_widget._conversion_end = lambda: None
    conversion_widget.input_file_widget.get_file = lambda: 'input.wav'
    conversion_widget.output_file_widget.get_file = lambda: 'output.wav'
    conversion_widget.model_widget.get_file = lambda: 'test_model.pth'
    conversion_widget.config_widget.get_file = lambda: 'test_config.json'

    conversion_widget.start_conversion()

    assert conversion_widget.start_button.isEnabled() == True