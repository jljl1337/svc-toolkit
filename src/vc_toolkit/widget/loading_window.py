from PySide6.QtWidgets import QMainWindow, QHBoxLayout, QLabel, QWidget
from PySide6.QtCore import Qt

class LoadingWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('Loading...')
        self.setFixedSize(300, 100)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)

        # Create a QWidget, set its layout and set it as central widget
        widget = QWidget()

        # Create a QHBoxLayout
        layout = QHBoxLayout(widget)

        # Add stretchable space before the label
        layout.addStretch()

        # Create the label
        label = QLabel('Loading...')

        # Set the stylesheet to increase the font size
        label.setStyleSheet("font-size: 20px;")

        # Add the label to the layout
        layout.addWidget(label)

        # Add stretchable space after the label
        layout.addStretch()
        self.setCentralWidget(widget)