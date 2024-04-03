import time

from widget.separation import SeparationThread, SeparationWidget

def test_separation_thread_run(qtbot):
    def test_function():
        pass

    separation_thread = SeparationThread(test_function, {'key1': 'value1'})

    separation_thread.run()

    assert separation_thread.isRunning() == False

def test_separation_widget_constructor(qtbot):
    separation_widget = SeparationWidget()
    qtbot.addWidget(separation_widget)

    assert separation_widget.file_widget.label.text() == 'Input File: Not chosen'

def test_separation_widget_set_model_list(qtbot):
    separation_widget = SeparationWidget()
    qtbot.addWidget(separation_widget)

    separation_widget.set_model_list([('model 1', 'model1'), ('model 2', 'model2')])

    assert separation_widget.model_dropdown.dropdown.currentText() == 'model 1'

def test_separation_widget_set_device_list(qtbot):
    separation_widget = SeparationWidget()
    qtbot.addWidget(separation_widget)

    separation_widget.set_device_list([('CPU', 'cpu'), ('GPU', 'gpu')])

    assert separation_widget.device_dropdown.dropdown.currentText() == 'CPU'

def test_separation_widget_set_separation_function(qtbot):
    separation_widget = SeparationWidget()
    qtbot.addWidget(separation_widget)

    def test_function():
        pass

    separation_widget.set_separation_function(test_function)

    assert separation_widget.separation_function == test_function

def test_separation_widget_update_progress(qtbot):
    separation_widget = SeparationWidget()
    qtbot.addWidget(separation_widget)

    separation_widget.update_progress(50)

    assert separation_widget.progress_bar.value() == 50

def test_separation_widget_start_separation(qtbot):
    separation_widget = SeparationWidget()
    qtbot.addWidget(separation_widget)

    def test_function():
        pass

    separation_widget.set_separation_function(test_function)
    separation_widget.update_progress = lambda x: None
    separation_widget.file_widget.get_file = lambda: 'file'
    separation_widget.dir_widget.get_directory = lambda: 'dir'
    separation_widget.vocal_checkbox.get_checked = lambda: True

    separation_widget.start_separation()

    assert separation_widget.start_button.isEnabled() == False

    time.sleep(0.5)