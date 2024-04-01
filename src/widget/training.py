import os

from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QGroupBox
from PySide6.QtCore import Qt, QThread, QSize
from PySide6.QtGui import QMovie

from .common import FileWidget, DirectoryWidget, CheckboxWidget, info_message_box, error_message_box

class PreprocessThread(QThread):
    def __init__(self, preprocess_function, dataset_dir, output_dir, split=False):
        super().__init__()
        self.preprocess_function = preprocess_function
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.split = split

    def run(self):
        try:
            self.error_message = None
            self.preprocess_function(self.dataset_dir, self.output_dir, self.split)

        except Exception as e:
            self.error_message = str(e)

class TrainingThread(QThread):
    def __init__(self, train_function, config_file, model_output_dir):
        super().__init__()
        self.train_function = train_function
        self.config_file = config_file
        self.model_output_dir = model_output_dir

    def run(self):
        self.train_function(self.config_file, self.model_output_dir)
        try:
            self.error_message = None
            self.train_function(self.model_output_dir, self.config_file)

        except Exception as e:
            self.error_message = str(e)

class TrainingWidget(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)

        self.preprocess_group_box = QGroupBox('Preprocess')
        self.preprocess_layout = QVBoxLayout(self.preprocess_group_box)

        self.dataset_dir_widget = DirectoryWidget('Dataset Directory')
        self.Output_dir_widget = DirectoryWidget('Output Directory')
        self.split_checkbox = CheckboxWidget('Split Dataset')
        self.start_preprocess_button = QPushButton('Start Preprocess')
        self.start_preprocess_button.clicked.connect(self._start_preprocess)

        self.preprocess_layout.addWidget(self.dataset_dir_widget)
        self.preprocess_layout.addWidget(self.Output_dir_widget)
        self.preprocess_layout.addWidget(self.split_checkbox)
        self.preprocess_layout.addWidget(self.start_preprocess_button)

        self.train_group_box = QGroupBox('Train')
        self.train_layout = QVBoxLayout(self.train_group_box)
        
        self.model_output_dir_widget = DirectoryWidget('Model Output Directory')
        self.config_file_widget = FileWidget('Config File')
        self.start_train_button = QPushButton('Start Train')
        self.start_train_button.clicked.connect(self._start_train)

        self.train_layout.addWidget(self.model_output_dir_widget)
        self.train_layout.addWidget(self.config_file_widget)
        self.train_layout.addWidget(self.start_train_button)

        self.loading_label = QLabel()
        self.loading_movie = QMovie(os.path.join(os.path.dirname(__file__), '../../img/loading.gif'))
        self.loading_movie.setScaledSize(QSize(165, 30))
        self.loading_label.setMovie(self.loading_movie)
        self.loading_label.hide()

        layout.addWidget(self.preprocess_group_box)
        layout.addWidget(self.train_group_box)
        layout.addWidget(self.loading_label, alignment=Qt.AlignCenter)

    def _preprocess_end(self):
        self.loading_label.hide()
        self.loading_movie.stop()
        self.start_preprocess_button.setEnabled(True)
        self.start_train_button.setEnabled(True)

        if self.preprocess_thread.error_message:
            error_message_box(self.preprocess_thread.error_message)
        else:
            info_message_box('Preprocessing is done')

    def _train_end(self):
        self.loading_label.hide()
        self.loading_movie.stop()
        self.start_preprocess_button.setEnabled(True)
        self.start_train_button.setEnabled(True)

        if self.train_thread.error_message:
            error_message_box(self.train_thread.error_message)
        else:
            info_message_box('Training is done')

    def _start_preprocess(self):
        dataset_dir = self.dataset_dir_widget.get_directory()
        output_dir = self.Output_dir_widget.get_directory()
        split = self.split_checkbox.get_checked()

        if not dataset_dir or not output_dir:
            error_message_box('Please select dataset directory and output directory')
            return

        self.preprocess_thread = PreprocessThread(
            self._preprocess,
            dataset_dir=dataset_dir,
            output_dir=output_dir,
            split=split
        )
        self.preprocess_thread.finished.connect(self._preprocess_end)
        self.preprocess_thread.start()

        self.loading_label.show()
        self.loading_movie.start()
        self.start_preprocess_button.setEnabled(False)
        self.start_train_button.setEnabled(False)

    def _start_train(self):
        model_output_dir = self.model_output_dir_widget.get_directory()
        config_file = self.config_file_widget.get_file()

        if not model_output_dir or not config_file:
            error_message_box('Please select model output directory and config file')
            return

        self.train_thread = TrainingThread(
            self._train,
            config_file=config_file,
            model_output_dir=model_output_dir
        )
        self.train_thread.finished.connect(self._train_end)
        self.train_thread.start()

        self.loading_label.show()
        self.loading_movie.start()
        self.start_preprocess_button.setEnabled(False)
        self.start_train_button.setEnabled(False)

    def set_preprocess_function(self, preprocess_function):
        self._preprocess = preprocess_function

    def set_train_function(self, train_function):
        self._train = train_function




