import json

from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QGroupBox
from PySide6.QtCore import QThread
from pyside6_utils.widgets import OverlayWidget

from svc_toolkit.widget.common import SliderWidget, FloatSliderWidget, FileWidget, SaveFileWidget, DropdownWidget, CheckboxWidget, DropdownWidget, info_message_box, error_message_box
from svc_toolkit.widget.loading_overlay import LoadingOverlayWidget

class ConversionThread(QThread):
    def __init__(self, conversion_function, kwargs):
        super().__init__()
        self.conversion_function = conversion_function
        self.kwargs = kwargs

    def run(self):
        try:
            self.error_message = None
            self.conversion_function(**self.kwargs)

        except Exception as e:
            self.error_message = str(e)

class ConversionWidget(OverlayWidget):
    def __init__(self):
        super().__init__(parent=None)

        layout = QVBoxLayout(self)

        self.file_group_box = QGroupBox('Files')
        self.file_layout = QVBoxLayout(self.file_group_box)

        self.input_file_widget = FileWidget('Input File')
        self.output_file_widget = SaveFileWidget('Output File', 'WAV (*.wav)')
        self.model_widget = FileWidget('Model')
        self.config_widget = FileWidget('Config', on_change=self._set_speaker_list)

        self.file_layout.addWidget(self.input_file_widget)
        self.file_layout.addWidget(self.output_file_widget)
        self.file_layout.addWidget(self.model_widget)
        self.file_layout.addWidget(self.config_widget)

        self.option_group_box = QGroupBox('Options')
        self.option_layout = QVBoxLayout(self.option_group_box)

        self.speaker_dropdown = DropdownWidget('Speaker', [])
        self.pitch_slider = SliderWidget('Pitch (12 = 1 octave)', -36, 36, 0, tick_interval=12)
        self.auto_predict_checkbox = CheckboxWidget('Auto Predict F0')
        self.device_dropdown = DropdownWidget('Device', [])

        self.option_layout.addWidget(self.speaker_dropdown)
        self.option_layout.addWidget(self.pitch_slider)
        self.option_layout.addWidget(self.auto_predict_checkbox)
        self.option_layout.addWidget(self.device_dropdown)

        self.advance_settings_show_button = QPushButton('Show Advanced Settings')
        self.advance_settings_show_button.clicked.connect(self._show_advanced_settings)

        self.start_button = QPushButton('Start Conversion')
        self.start_button.clicked.connect(self.start_conversion)

        self.loading_overlay = LoadingOverlayWidget(self)
        self.set_overlay_widget(self.loading_overlay)
        self.set_overlay_hidden(True)

        layout.addWidget(self.file_group_box)
        layout.addWidget(self.option_group_box)
        layout.addWidget(self.advance_settings_show_button)
        layout.addWidget(self.start_button)
        
        # Advanced Settings Window
        self.advanced_settings_widget = QWidget()
        self.advanced_settings_widget.setWindowTitle('Advanced Settings')
        self.advanced_settings_widget.setFixedWidth(600)
        self.advanced_settings_layout = QVBoxLayout(self.advanced_settings_widget)
        
        self.f0_prediction_method_dropdown = DropdownWidget('F0 Prediction Method', [('Crepe', 'crepe'), ('Crepe-tiny', 'crepe-tiny'), ('Parselmouth', 'parselmouth'), ('Dio', 'dio'), ('Harvest', 'harvest')])
        self.silence_threshold_slider = FloatSliderWidget('Silence Threshold', -60, 0, -35)
        self.cluster_infer_ratio_slider = FloatSliderWidget('Cluster Infer Ratio', 0, 1, 0)
        self.noise_scale_slider = FloatSliderWidget('Noise Scale', 0, 1, 0.4)
        self.pad_seconds_slider = FloatSliderWidget('Pad Seconds', 0, 1, 0.1)
        self.chunk_seconds_slider = FloatSliderWidget('Chunk Seconds', 0, 3, 0.5)
        self.max_chunk_seconds_slider = SliderWidget('Max Chunk Seconds', 0, 240, 40, tick_interval=20)
        self.absolute_threshold_checkbox = CheckboxWidget('Absolute Threshold', default_checked=True)
        self.advance_settings_close_button = QPushButton('Close Advanced Settings')
        self.advance_settings_close_button.clicked.connect(self._close_advanced_settings)

        self.advanced_settings_layout.addWidget(self.f0_prediction_method_dropdown)
        self.advanced_settings_layout.addWidget(self.silence_threshold_slider)
        self.advanced_settings_layout.addWidget(self.cluster_infer_ratio_slider)
        self.advanced_settings_layout.addWidget(self.noise_scale_slider)
        self.advanced_settings_layout.addWidget(self.pad_seconds_slider)
        self.advanced_settings_layout.addWidget(self.chunk_seconds_slider)
        self.advanced_settings_layout.addWidget(self.max_chunk_seconds_slider)
        self.advanced_settings_layout.addWidget(self.absolute_threshold_checkbox)
        self.advanced_settings_layout.addWidget(self.advance_settings_close_button)

    def _show_advanced_settings(self):
        self.advanced_settings_widget.show()

    def _close_advanced_settings(self):
        self.advanced_settings_widget.hide()

    def _set_speaker_list(self, config_path: str):
        try:
            # Read the config json file and set the speaker list
            with open(config_path) as f:
                config = json.load(f)
                self.speaker_dropdown.set_options(list(config['spk'].items()))
        except Exception as e:
            self.speaker_dropdown.set_options([])
            error_message_box(str(e))

    def set_device_list(self, device_list):
        self.device_dropdown.set_options(device_list)

    def set_conversion_function(self, conversion_function):
        self.conversion_function = conversion_function

    def _conversion_end(self):
        self.loading_overlay.stop_movie()
        self.set_overlay_hidden(True)

        if self.conversion_thread.error_message is None:
            info_message_box('Conversion finished.')
        else:
            error_message_box(self.conversion_thread.error_message)

    def start_conversion(self):
        error_message = ''

        # if self.input_file_widget.get_file() is None:
        #     error_message += 'Input file is not chosen.\n'
        # if self.output_file_widget.get_file() is None:
        #     error_message += 'Output file is not chosen.\n'
        # if self.model_widget.get_file() is None:
        #     error_message += 'Model file is not chosen.\n'
        # if self.config_widget.get_file() is None:
        #     error_message += 'Config file is not chosen.\n'

        if error_message != '':
            error_message_box(error_message)
            return

        kwargs = {
            'model_path': self.model_widget.get_file(),
            'config_path': self.config_widget.get_file(),
            'device': self.device_dropdown.get_data(),
            'input_path': self.input_file_widget.get_file(),
            'output_path': self.output_file_widget.get_file(),
            'speaker': self.speaker_dropdown.get_data(),
            'transpose': self.pitch_slider.get_value(),
            'auto_predict_f0': self.auto_predict_checkbox.get_checked(),
            'f0_method': self.f0_prediction_method_dropdown.get_data(),
            'db_thresh': self.silence_threshold_slider.get_value(),
            'cluster_infer_ratio': self.cluster_infer_ratio_slider.get_value(),
            'noise_scale': self.noise_scale_slider.get_value(),
            'pad_seconds': self.pad_seconds_slider.get_value(),
            'chunk_seconds': self.chunk_seconds_slider.get_value(),
            'max_chunk_seconds': self.max_chunk_seconds_slider.get_value(),
            'absolute_thresh': self.absolute_threshold_checkbox.get_checked(),
        }

        self.conversion_thread = ConversionThread(self.conversion_function, kwargs)
        self.conversion_thread.finished.connect(self._conversion_end)
        self.conversion_thread.start()

        self.set_overlay_hidden(False)
        self.loading_overlay.start_movie()
