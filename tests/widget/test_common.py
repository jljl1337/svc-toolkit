from widget.common import FileWidget, SaveFileWidget, DropdownWidget, CheckboxWidget, DropdownWidget

def test_file_widget(qtbot):
    file_widget = FileWidget('Test File')
    qtbot.addWidget(file_widget)
    assert file_widget.label.text() == 'Test File: Not chosen'

    assert file_widget.get_file() is None

