from widget.training import TrainingThread, PreprocessThread, TrainingWidget

def test_training_thread_run(qtbot):
    def test_function():
        pass

    training_thread = TrainingThread(test_function, 'config_file', 'model_output_dir')

    training_thread.run()

    assert training_thread.isRunning() == False

def test_preprocess_thread_run(qtbot):
    def test_function():
        pass

    preprocess_thread = PreprocessThread(test_function, 'dataset_dir', 'output_dir', 'split')

    preprocess_thread.run()

    assert preprocess_thread.isRunning() == False

def test_training_widget_constructor(qtbot):
    training_widget = TrainingWidget()
    qtbot.addWidget(training_widget)

    assert training_widget.dataset_dir_widget.label.text() == 'Dataset Directory: Not chosen'

def test_training_widget_long_process_end(qtbot):
    training_widget = TrainingWidget()
    qtbot.addWidget(training_widget)

    training_widget._long_process_end()

    assert training_widget.overlayHidden == True

def test_training_widget_long_process_start(qtbot):
    training_widget = TrainingWidget()
    qtbot.addWidget(training_widget)

    training_widget._long_process_start()

    assert training_widget.overlayHidden == False

def test_training_widget_start_preprocess(qtbot):
    training_widget = TrainingWidget()
    qtbot.addWidget(training_widget)

    training_widget._preprocess = lambda: None
    training_widget._preprocess_end = lambda: None
    training_widget.dataset_dir_widget.get_directory = lambda: 'dataset_dir'
    training_widget.Output_dir_widget.get_directory = lambda: 'output_dir'
    training_widget.split_checkbox.get_checked = lambda: True

    training_widget._start_preprocess()

    assert training_widget.overlayHidden == False

def test_training_widget_start_train(qtbot):
    training_widget = TrainingWidget()
    qtbot.addWidget(training_widget)

    training_widget._train = lambda: None
    training_widget._train_end = lambda: None
    training_widget.model_output_dir_widget.get_directory = lambda: 'model_output_dir'
    training_widget.config_file_widget.get_file = lambda: 'config_file'

    training_widget._start_train()

    assert training_widget.overlayHidden == False

def test_training_widget_set_preprocess_function(qtbot):
    training_widget = TrainingWidget()
    qtbot.addWidget(training_widget)

    def test_function():
        pass

    training_widget.set_preprocess_function(test_function)

    assert training_widget._preprocess == test_function

def test_training_widget_set_train_function(qtbot):
    training_widget = TrainingWidget()
    qtbot.addWidget(training_widget)

    def test_function():
        pass

    training_widget.set_train_function(test_function)

    assert training_widget._train == test_function