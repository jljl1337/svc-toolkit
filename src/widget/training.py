from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QComboBox
from PySide6.QtCore import Qt

class TrainingWidget(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)

        self.dataset_layout = QHBoxLayout()
        self.dataset_button = QPushButton('Choose Dataset Directory')
        self.dataset_button.clicked.connect(self.choose_dataset)
        self.dataset_label = QLabel('No dataset directory chosen')
        self.dataset_layout.addWidget(self.dataset_label)
        self.dataset_layout.addWidget(self.dataset_button)

        self.device_dropdown = QComboBox()
        self.device_dropdown.addItem('Device 1')
        self.device_dropdown.addItem('Device 2')
        # Add more devices as needed

        layout.addLayout(self.dataset_layout)
        layout.addWidget(self.device_dropdown)

        self.setLayout(layout)

    def choose_dataset(self):
        dataset_dir = QFileDialog.getExistingDirectory(self, 'Choose Dataset Directory')
        if dataset_dir:
            self.dataset_label.setText(dataset_dir)
