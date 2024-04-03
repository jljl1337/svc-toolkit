from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QProgressBar, QGroupBox
from PySide6.QtCore import QThread, Signal

from svc_toolkit.widget.common import info_message_box, error_message_box, FileWidget, DirectoryWidget, CheckboxWidget, DropdownWidget

class SeparationThread(QThread):
    progress_signal = Signal(int)

    def __init__(self, separation_function, kwargs):
        super().__init__()
        self.separation_function = separation_function
        self.kwargs = kwargs

    def progress_signal_connect(self, slot):
        self.progress_signal.connect(slot)

    def run(self):
        try:
            self.error_message = None
            self.separation_function(self.progress_signal.emit, **self.kwargs)

        except Exception as e:
            self.error_message = str(e)

class SeparationWidget(QWidget):
    def __init__(self):
        super().__init__()

        layout = QVBoxLayout(self)

        self.file_group_box = QGroupBox('Files')
        self.file_layout = QVBoxLayout(self.file_group_box)

        self.file_widget = FileWidget('Input File')
        self.dir_widget = DirectoryWidget('Output Directory')
        
        self.file_layout.addWidget(self.file_widget)
        self.file_layout.addWidget(self.dir_widget)

        self.option_group_box = QGroupBox('Options')
        self.option_layout = QVBoxLayout(self.option_group_box)

        self.vocal_checkbox = CheckboxWidget('Output a vocal-only file (vocal.wav)')
        self.non_vocal_checkbox = CheckboxWidget('Output a non-vocal file (instrument.wav)')
        self.model_dropdown = DropdownWidget('Model', [])

        self.option_layout.addWidget(self.vocal_checkbox)
        self.option_layout.addWidget(self.non_vocal_checkbox)
        self.option_layout.addWidget(self.model_dropdown)

        self.setting_group_box = QGroupBox('Settings')
        self.setting_layout = QVBoxLayout(self.setting_group_box)

        self.device_dropdown = DropdownWidget('Device', [])
        self.precision_dropdown = DropdownWidget('Precision', [('BFloat16', 'bf16'), ('Float32', '32')])

        self.setting_layout.addWidget(self.device_dropdown)
        self.setting_layout.addWidget(self.precision_dropdown)

        self.start_button = QPushButton('Start Separation')
        self.start_button.clicked.connect(self.start_separation)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        layout.addWidget(self.file_group_box)
        layout.addWidget(self.option_group_box)
        layout.addWidget(self.setting_group_box)
        layout.addWidget(self.start_button)
        layout.addWidget(self.progress_bar)

    def set_model_list(self, model_list):
        self.model_dropdown.set_options(model_list)

    def set_device_list(self, device_list):
        self.device_dropdown.set_options(device_list)

    def set_separation_function(self, separation_function):
        self.separation_function = separation_function

    def update_progress(self, value):
        self.progress_bar.setValue(value)

    def separation_end(self):
        self.start_button.setEnabled(True)
        self.progress_bar.setValue(0)

        if self.separation_thread.error_message is None:
            info_message_box('Separation finished.')
        else:
            error_message_box(self.separation_thread.error_message)

    def start_separation(self):
        error_message = ''

        if self.file_widget.get_file() is None:
            error_message += 'No file chosen.\n'
        if self.dir_widget.get_directory() is None:
            error_message += 'No output directory chosen.\n'
        if not self.vocal_checkbox.get_checked() and not self.non_vocal_checkbox.get_checked():
            error_message += 'Choose at least one output option.\n'

        if error_message != '':
            error_message_box(error_message)
            return
        
        kwargs = {
            'file': self.file_widget.get_file(),
            'output_dir': self.dir_widget.get_directory(),
            'vocal': self.vocal_checkbox.get_checked(),
            'non_vocal': self.non_vocal_checkbox.get_checked(),
            'model': self.model_dropdown.get_data(),
            'device': self.device_dropdown.get_data(),
            'precision': self.precision_dropdown.get_data()
        }

        self.separation_thread = SeparationThread(self.separation_function, kwargs)
        self.separation_thread.progress_signal_connect(self.update_progress)
        self.separation_thread.finished.connect(self.separation_end)
        self.separation_thread.start()

        self.start_button.setEnabled(False)

