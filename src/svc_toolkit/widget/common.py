import os
from typing import Callable

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QCheckBox, QComboBox, QMessageBox, QSlider, QLineEdit
from PySide6.QtCore import Qt
from PySide6.QtGui import QIntValidator, QDoubleValidator

def info_message_box(message: str) -> None:
    message_box = QMessageBox()
    message_box.setIcon(QMessageBox.Icon.Information)
    message_box.setWindowTitle('Info')
    message_box.setText(message)
    message_box.exec()

def error_message_box(message: str) -> None:
    message_box = QMessageBox()
    message_box.setIcon(QMessageBox.Icon.Critical)
    message_box.setWindowTitle('Error')
    message_box.setText(message)
    message_box.exec()

class FileWidget(QWidget):
    def __init__(self, name: str, on_change: Callable = None) -> None:
        super().__init__()

        layout = QHBoxLayout(self)
        self.name = name
        self.file_name = None
        self.on_change = on_change

        self.label = QLabel(f'{name}: Not chosen')
        self.button = QPushButton('Choose File')
        self.button.clicked.connect(self._choose_file)
        layout.addWidget(self.label, alignment=Qt.AlignLeft)
        layout.addWidget(self.button)

        self.setFixedHeight(self.sizeHint().height())

    def _choose_file(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(self, 'Choose File')
        if file_name:
            self.file_name = file_name
            self.label.setText(f'{self.name}: {os.path.basename(file_name)}')
            self.label.setToolTip(file_name)

            if self.on_change:
                self.on_change(file_name)

    def get_file(self) -> str:
        return self.file_name


# Save as file widget
class SaveFileWidget(QWidget):
    def __init__(self, name: str, file_types: str) -> None:
        super().__init__()

        layout = QHBoxLayout(self)
        self.name = name
        self.file_name = None

        self.label = QLabel(f'{name}: Not chosen')
        self.button = QPushButton('Choose File')
        self.button.clicked.connect(self._choose_file)
        layout.addWidget(self.label, alignment=Qt.AlignLeft)
        layout.addWidget(self.button)

        self.file_types = file_types

        self.setFixedHeight(self.sizeHint().height())

    def _choose_file(self) -> None:
        file_name, _ = QFileDialog.getSaveFileName(self, 'Choose File', '~/', self.file_types)
        if file_name:
            self.file_name = file_name
            self.label.setText(f'{self.name}: {os.path.basename(file_name)}')
            self.label.setToolTip(file_name)

    def get_file(self) -> str:
        return self.file_name

class DirectoryWidget(QWidget):
    def __init__(self, name: str) -> None:
        super().__init__()

        layout = QHBoxLayout(self)
        self.name = name
        self.dir_name = None

        self.label = QLabel(f'{name}: Not chosen')
        self.button = QPushButton('Choose Directory')
        self.button.clicked.connect(self._choose_directory)
        layout.addWidget(self.label, alignment=Qt.AlignLeft)
        layout.addWidget(self.button)

        self.setFixedHeight(self.sizeHint().height())

    def _choose_directory(self) -> None:
        dir_name = QFileDialog.getExistingDirectory(self, 'Choose Output Directory')
        if dir_name:
            self.dir_name = dir_name
            self.label.setText(f'{self.name}: {os.path.basename(dir_name)}')
            self.label.setToolTip(dir_name)

    def get_directory(self) -> str:
        return self.dir_name

class CheckboxWidget(QWidget):
    def __init__(self, name: str, on_change: Callable = None, default_checked: bool = False) -> None:
        super().__init__()

        layout = QHBoxLayout(self)
        self.name = name
        self.on_change = on_change

        self.label = QLabel(f'{name}:')
        self.checkbox = QCheckBox()

        if default_checked:
            self.checkbox.setChecked(True)

        self.checkbox.stateChanged.connect(self._on_checkbox_changed)

        layout.addWidget(self.label, alignment=Qt.AlignLeft)
        layout.addWidget(self.checkbox, alignment=Qt.AlignCenter)

        self.setFixedHeight(self.sizeHint().height())

    def _on_checkbox_changed(self, state: Qt.CheckState) -> None:
        if self.on_change:
            self.on_change(state == Qt.CheckState.Checked)

    def get_checked(self) -> bool:
        return self.checkbox.isChecked()

class DropdownWidget(QWidget):
    def __init__(self, name: str, options: list) -> None:
        super().__init__()

        layout = QHBoxLayout(self)
        self.name = name

        self.label = QLabel(f'{name}:')
        self.dropdown = QComboBox()

        for option in options:
            self.dropdown.addItem(*option)

        layout.addWidget(self.label, alignment=Qt.AlignLeft)
        layout.addWidget(self.dropdown)

        self.setFixedHeight(self.sizeHint().height())

    def set_options(self, options: list) -> None:
        self.dropdown.clear()
        for option in options:
            self.dropdown.addItem(*option)

    def get_data(self) -> str:
        return self.dropdown.currentData()

# Slider widget with label, and a text box to show the value, and the slider and text box are connected
class SliderWidget(QWidget):
    def __init__(self, name: str, min_value: int, max_value: int, default_value: int, tick_interval: int = None) -> None:
        super().__init__()

        layout = QHBoxLayout(self)
        self.name = name

        self.label = QLabel(f'{name}:')
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(min_value, max_value)
        self.slider.setValue(default_value)

        if tick_interval:
            self.slider.setTickPosition(QSlider.TickPosition.TicksBelow)
            self.slider.setTickInterval(tick_interval)

        self.textbox = QLineEdit(str(default_value))
        self.textbox.setFixedWidth(50)
        self.textbox.setValidator(QIntValidator(min_value, max_value))

        self.slider.valueChanged.connect(self.update_textbox)
        self.textbox.textChanged.connect(self.update_slider)

        # Create a QHBoxLayout for the slider and the line edit
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.textbox)

        layout.addWidget(self.label)
        layout.addLayout(slider_layout)

        layout.setStretch(0, 1)
        layout.setStretch(1, 1)

        self.setFixedHeight(self.sizeHint().height())

    def update_textbox(self, value: int) -> None:
        self.textbox.setText(str(value))

    def update_slider(self, text: str) -> None:
        if text:
            self.slider.setValue(float(text))

    def get_value(self) -> int:
        return self.slider.value()

class FloatSliderWidget(QWidget):
    def __init__(self, name: str, min_value: float, max_value: float, default_value: float, decimals: int = 2) -> None:
        super().__init__()

        layout = QHBoxLayout(self)
        self.name = name
        self.multiplier = 10 ** decimals

        self.label = QLabel(f'{name}:')
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(min_value * self.multiplier, max_value * self.multiplier)
        self.slider.setValue(default_value * self.multiplier)

        self.textbox = QLineEdit(str(default_value))
        self.textbox.setFixedWidth(50)
        self.textbox.setValidator(QDoubleValidator(min_value, max_value, decimals))

        self.slider.valueChanged.connect(self.update_textbox)
        self.textbox.textChanged.connect(self.update_slider)

        # Create a QHBoxLayout for the slider and the line edit
        slider_layout = QHBoxLayout()
        slider_layout.addWidget(self.slider)
        slider_layout.addWidget(self.textbox)

        layout.addWidget(self.label)
        layout.addLayout(slider_layout)

        layout.setStretch(0, 1)
        layout.setStretch(1, 1)

        self.setFixedHeight(self.sizeHint().height())

    def update_textbox(self, value: int) -> None:
        self.textbox.setText(str(value / self.multiplier))

    def update_slider(self, text: str) -> None:
        if text:
            self.slider.setValue(float(text) * self.multiplier)

    def get_value(self) -> float:
        return self.slider.value() / self.multiplier