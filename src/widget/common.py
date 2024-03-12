import os

from PySide6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QCheckBox, QComboBox, QMessageBox
from PySide6.QtCore import Qt

def info_message_box(message):
    message_box = QMessageBox()
    message_box.setIcon(QMessageBox.Icon.Information)
    message_box.setWindowTitle('Info')
    message_box.setText(message)
    message_box.exec()

def error_message_box(message):
    message_box = QMessageBox()
    message_box.setIcon(QMessageBox.Icon.Critical)
    message_box.setWindowTitle('Error')
    message_box.setText(message)
    message_box.exec()

class FileWidget(QWidget):
    def __init__(self, name):
        super().__init__()

        layout = QHBoxLayout(self)
        self.name = name
        self.file_name = None

        self.label = QLabel(f'{name}: Not chosen')
        self.button = QPushButton('Choose File')
        self.button.clicked.connect(self._choose_file)
        layout.addWidget(self.label, alignment=Qt.AlignLeft)
        layout.addWidget(self.button)

        self.setLayout(layout)
        self.setFixedHeight(self.sizeHint().height())

    def _choose_file(self):
        file_name, _ = QFileDialog.getOpenFileName(self, 'Choose File')
        if file_name:
            self.file_name = file_name
            self.label.setText(f'{self.name}: {os.path.basename(file_name)}')
            self.label.setToolTip(file_name)

    def get_file(self):
        return self.file_name

class DirectoryWidget(QWidget):
    def __init__(self, name):
        super().__init__()

        layout = QHBoxLayout(self)
        self.name = name
        self.dir_name = None

        self.label = QLabel(f'{name}: Not chosen')
        self.button = QPushButton('Choose Directory')
        self.button.clicked.connect(self._choose_directory)
        layout.addWidget(self.label, alignment=Qt.AlignLeft)
        layout.addWidget(self.button)

        self.setLayout(layout)
        self.setFixedHeight(self.sizeHint().height())

    def _choose_directory(self):
        dir_name = QFileDialog.getExistingDirectory(self, 'Choose Output Directory')
        if dir_name:
            self.dir_name = dir_name
            self.label.setText(f'{self.name}: {os.path.basename(dir_name)}')
            self.label.setToolTip(dir_name)

    def get_directory(self):
        return self.dir_name

class CheckboxWidget(QWidget):
    def __init__(self, name):
        super().__init__()

        layout = QHBoxLayout(self)
        self.name = name

        self.label = QLabel(f'{name}:')
        self.checkbox = QCheckBox()

        layout.addWidget(self.label, alignment=Qt.AlignLeft)
        layout.addWidget(self.checkbox, alignment=Qt.AlignCenter)

        self.setLayout(layout)
        self.setFixedHeight(self.sizeHint().height())

    def get_checked(self):
        return self.checkbox.isChecked()

class DropdownWidget(QWidget):
    def __init__(self, name, options):
        super().__init__()

        layout = QHBoxLayout(self)
        self.name = name

        self.label = QLabel(f'{name}:')
        self.dropdown = QComboBox()

        for option in options:
            self.dropdown.addItem(option)

        layout.addWidget(self.label, alignment=Qt.AlignLeft)
        layout.addWidget(self.dropdown)

        self.setLayout(layout)
        self.setFixedHeight(self.sizeHint().height())

    def set_options(self, options):
        self.dropdown.clear()
        for option in options:
            self.dropdown.addItem(*option)

    def get_data(self):
        return self.dropdown.currentData()