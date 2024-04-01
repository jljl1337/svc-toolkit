import os

from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QGroupBox
from PySide6.QtCore import Qt, QThread, QSize
from PySide6.QtGui import QMovie

from widget.common import SliderWidget, FloatSliderWidget, FileWidget, SaveFileWidget, DropdownWidget, CheckboxWidget, DropdownWidget, info_message_box, error_message_box

class TrainingWidget(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)

        self.preprocess_group_box = QGroupBox('Preprocess')
        self.preprocess_layout = QVBoxLayout(self.preprocess_group_box)

        self.dataset_dir_widget = FileWidget('Dataset Directory')
        self.Output_dir_widget = FileWidget('Output Directory')
        self.start_preprocess_button = QPushButton('Start Preprocess')
        # self.start_preprocess_button.clicked.connect(self.start_preprocess)

        self.preprocess_layout.addWidget(self.dataset_dir_widget)
        self.preprocess_layout.addWidget(self.Output_dir_widget)
        self.preprocess_layout.addWidget(self.start_preprocess_button)

        self.train_group_box = QGroupBox('Train')
        self.train_layout = QVBoxLayout(self.train_group_box)
        
        self.model_file_widget = FileWidget('Model File')
        self.config_file_widget = FileWidget('Config File')
        self.start_train_button = QPushButton('Start Train')
        # self.start_train_button.clicked.connect(self.start_train)

        self.train_layout.addWidget(self.model_file_widget)
        self.train_layout.addWidget(self.config_file_widget)
        self.train_layout.addWidget(self.start_train_button)

        self.loading_label = QLabel()
        self.loading_movie = QMovie(os.path.join(os.path.dirname(__file__), '../../img/loading.gif'))
        self.loading_movie.setScaledSize(QSize(165, 30))
        self.loading_label.setMovie(self.loading_movie)
        self.loading_label.hide()

        layout.addWidget(self.preprocess_group_box)
        layout.addWidget(self.train_group_box)
        layout.addWidget(self.loading_label)





