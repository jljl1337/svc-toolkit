from vc_toolkit.widget.common import FileWidget, SaveFileWidget, DirectoryWidget, CheckboxWidget, DropdownWidget, DropdownWidget, SliderWidget, FloatSliderWidget

def test_file_widget(qtbot):
    file_widget = FileWidget('Test File')
    qtbot.addWidget(file_widget)
    assert file_widget.label.text() == 'Test File: Not chosen'

    assert file_widget.get_file() is None

def test_save_file_widget(qtbot):
    save_file_widget = SaveFileWidget('Test File', 'WAV (*.wav)')
    qtbot.addWidget(save_file_widget)
    assert save_file_widget.label.text() == 'Test File: Not chosen'

    assert save_file_widget.get_file() is None

def test_directory_widget(qtbot):
    directory_widget = DirectoryWidget('Test Directory')
    qtbot.addWidget(directory_widget)
    assert directory_widget.label.text() == 'Test Directory: Not chosen'

    assert directory_widget.get_directory() is None

def test_checkbox_widget(qtbot):
    update = None

    def on_change(checked):
        nonlocal update
        update = checked

    checkbox_widget = CheckboxWidget('Test Checkbox', on_change=on_change, default_checked=True)
    qtbot.addWidget(checkbox_widget)

    assert checkbox_widget.get_checked() == True

    checkbox_widget.checkbox.setChecked(False)

    assert update == False

def test_dropdown_widget(qtbot):
    dropdown_widget = DropdownWidget('Test Dropdown', [('A', 'a'), ('B', 'b')])
    qtbot.addWidget(dropdown_widget)
    assert dropdown_widget.dropdown.currentText() == 'A'

    assert dropdown_widget.get_data() == 'a'

    dropdown_widget.set_options([('C', 'c'), ('D', 'd')])

    assert dropdown_widget.dropdown.currentText() == 'C'

def test_slider_widget(qtbot):
    slider_widget = SliderWidget('Test Slider', -10, 10, 0, tick_interval=5)
    qtbot.addWidget(slider_widget)
    assert slider_widget.slider.value() == 0

    assert slider_widget.get_value() == 0

    slider_widget.slider.setValue(5)

    assert slider_widget.get_value() == 5

def test_float_slider_widget(qtbot):
    slider_widget = FloatSliderWidget('Test Slider', -10, 10, 0)
    qtbot.addWidget(slider_widget)
    assert slider_widget.slider.value() == 0

    slider_widget.slider.setValue(100)

    assert slider_widget.get_value() == 1
