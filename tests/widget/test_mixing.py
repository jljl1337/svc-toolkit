import time

from widget.mixing import MixingWidget

def test_mixing_widget_constructor(qtbot):
    mixing_widget = MixingWidget()
    qtbot.addWidget(mixing_widget)

    assert mixing_widget.source_1_file_widget.label.text() == 'Source 1 File: Not chosen'

def test_mixing_widget_set_mixer_function(qtbot):
    mixing_widget = MixingWidget()
    qtbot.addWidget(mixing_widget)

    def test_function():
        pass

    mixing_widget.set_mixer_function(test_function)

    assert mixing_widget.mixer_function == test_function

def test_mixing_widget_start_mixing(qtbot):
    mixing_widget = MixingWidget()
    qtbot.addWidget(mixing_widget)

    def test_function(*args, **kwargs):
        pass

    mixing_widget.set_mixer_function(test_function)
    mixing_widget.mixing_end = lambda: None
    mixing_widget.source_1_file_widget.get_file = lambda: 'source1.wav'
    mixing_widget.source_2_file_widget.get_file = lambda: 'source2.wav'
    mixing_widget.save_file_widget.get_file = lambda: 'output.wav'
    mixing_widget.source_1_ratio_slider.get_value = lambda: 0.5
    mixing_widget.normalize_checkbox.get_checked = lambda: True

    mixing_widget.start_mixing()

    assert mixing_widget.start_button.isEnabled() == False

    # Wait for the thread to finish
    time.sleep(0.5)