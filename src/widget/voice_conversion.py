from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QGroupBox, QSlider, QComboBox
from PySide6.QtCore import Qt

class VoiceConversionWidget(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)

        self.file_layout = QHBoxLayout()
        self.file_button = QPushButton('Choose File for Conversion')
        self.file_button.clicked.connect(self.choose_file)
        self.file_label = QLabel('No file chosen for conversion')
        self.file_layout.addWidget(self.file_label)
        self.file_layout.addWidget(self.file_button)

        self.model_layout = QHBoxLayout()
        self.model_button = QPushButton('Choose Model')
        self.model_button.clicked.connect(self.choose_model)
        self.model_label = QLabel('No model chosen')
        self.model_layout.addWidget(self.model_label)
        self.model_layout.addWidget(self.model_button)

        self.config_layout = QHBoxLayout()
        self.config_button = QPushButton('Choose Config')
        self.config_button.clicked.connect(self.choose_config)
        self.config_label = QLabel('No config chosen')
        self.config_layout.addWidget(self.config_label)
        self.config_layout.addWidget(self.config_button)

        self.preset_dropdown = QComboBox()
        self.preset_dropdown.addItem('Preset 1')
        self.preset_dropdown.addItem('Preset 2')
        # Add more presets as needed

        self.device_dropdown = QComboBox()
        self.device_dropdown.addItem('Device 1')
        self.device_dropdown.addItem('Device 2')
        # Add more devices as needed

        self.start_button = QPushButton('Start Conversion')
        # Connect to your conversion function

        self.advanced_settings = QGroupBox('Advanced Settings')
        self.advanced_settings_layout = QVBoxLayout(self.advanced_settings)
        self.slider1 = QSlider(Qt.Horizontal)
        self.slider1_label = QLabel('Slider 1')
        self.advanced_settings_layout.addWidget(self.slider1_label)
        self.advanced_settings_layout.addWidget(self.slider1)
        self.slider2 = QSlider(Qt.Horizontal)
        self.slider2_label = QLabel('Slider 2')
        self.advanced_settings_layout.addWidget(self.slider2_label)
        self.advanced_settings_layout.addWidget(self.slider2)

        self.advanced_settings.hide()

        self.advanced_settings_widget = QWidget()
        self.advanced_settings_widget_layout = QVBoxLayout(self.advanced_settings_widget)
        self.advanced_settings_widget_layout.addWidget(self.advanced_settings)
        self.advanced_settings_widget.setFixedHeight(200)

        self.show_hide_button = QPushButton('Show Advanced Settings')
        self.show_hide_button.clicked.connect(self.toggle_advanced_settings)

        layout.addLayout(self.file_layout)
        layout.addLayout(self.model_layout)
        layout.addLayout(self.config_layout)
        layout.addWidget(self.preset_dropdown)
        layout.addWidget(self.device_dropdown)
        layout.addWidget(self.start_button)
        layout.addWidget(self.show_hide_button)
        layout.addWidget(self.advanced_settings_widget)

        self.setLayout(layout)

    def choose_model(self):
        model_name, _ = QFileDialog.getOpenFileName(self, 'Choose Model')
        if model_name:
            self.model_label.setText(model_name)

    def choose_config(self):
        config_name, _ = QFileDialog.getOpenFileName(self, 'Choose Config')
        if config_name:
            self.config_label.setText(config_name)

    def choose_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Choose File for Conversion')
        if file_name:
            self.file_label.setText(file_name)

    def toggle_advanced_settings(self):
        if self.advanced_settings.isHidden():
            self.advanced_settings.show()
            self.show_hide_button.setText('Hide Advanced Settings')
        else:
            self.advanced_settings.hide()
            self.show_hide_button.setText('Show Advanced Settings')
