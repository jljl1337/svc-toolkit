from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QGroupBox
from PySide6.QtCore import QThread, Signal

from svc_toolkit.widget.common import info_message_box, error_message_box, FileWidget, SaveFileWidget, CheckboxWidget, FloatSliderWidget

class MixingThread(QThread):
    def __init__(self, mixer_function, kwargs):
        super().__init__()
        self.mixer_function = mixer_function
        self.kwargs = kwargs

    def run(self):
        try:
            self.error_message = None
            self.mixer_function(**self.kwargs)

        except Exception as e:
            self.error_message = str(e)

class MixingWidget(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)

        self.file_group_box = QGroupBox('Files')
        self.file_layout = QVBoxLayout(self.file_group_box)

        self.source_1_file_widget = FileWidget('Source 1 File')
        self.source_2_file_widget = FileWidget('Source 2 File')
        self.save_file_widget = SaveFileWidget('Output File', 'WAV (*.wav)')
        
        self.file_layout.addWidget(self.source_1_file_widget)
        self.file_layout.addWidget(self.source_2_file_widget)
        self.file_layout.addWidget(self.save_file_widget)

        self.option_group_box = QGroupBox('Options')
        self.option_layout = QVBoxLayout(self.option_group_box)

        self.source_1_ratio_slider = FloatSliderWidget('Source 1 Ratio', 0.01, 0.99, 0.5)
        self.normalize_checkbox = CheckboxWidget('Normalize', default_checked=True)

        self.option_layout.addWidget(self.source_1_ratio_slider)
        self.option_layout.addWidget(self.normalize_checkbox)

        self.start_button = QPushButton('Start Mixing')
        self.start_button.clicked.connect(self.start_mixing)

        layout.addWidget(self.file_group_box)
        layout.addWidget(self.option_group_box)
        layout.addWidget(self.start_button)

    def set_mixer_function(self, mixer_function):
        self.mixer_function = mixer_function

    def mixing_end(self):
        self.start_button.setEnabled(True)

        error_message = self.mixing_thread.error_message

        if error_message:
            error_message_box(error_message)
        else:
            info_message_box('Mixing completed')

    def start_mixing(self):
        error_message = ''

        if not self.source_1_file_widget.get_file():
            error_message += 'Source 1 file is not selected.\n'
        if not self.source_2_file_widget.get_file():
            error_message += 'Source 2 file is not selected.\n'
        if not self.save_file_widget.get_file():
            error_message += 'Output file is not selected.\n'

        if error_message:
            error_message_box(error_message)
            return

        self.mixing_thread = MixingThread(self.mixer_function, {
            'source_1_path': self.source_1_file_widget.get_file(),
            'source_2_path': self.source_2_file_widget.get_file(),
            'output_path': self.save_file_widget.get_file(),
            'source_1_ratio': self.source_1_ratio_slider.get_value(),
            'normalize': self.normalize_checkbox.get_checked()
        })

        self.mixing_thread.finished.connect(self.mixing_end)
        self.mixing_thread.start()

        self.start_button.setEnabled(False)