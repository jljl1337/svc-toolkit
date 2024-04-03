import os

from PySide6.QtWidgets import QWidget, QLabel, QVBoxLayout
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QMovie

class LoadingOverlayWidget(QWidget):
    def __init__(self, parent=None):
        super(LoadingOverlayWidget, self).__init__(parent)

        self.loading_label = QLabel(self)
        self.loading_movie = QMovie(os.path.join(os.path.dirname(__file__), '../img/loading.gif'))
        self.loading_movie.setScaledSize(QSize(165, 30))
        self.loading_label.setMovie(self.loading_movie)

        layout = QVBoxLayout(self)
        layout.addWidget(self.loading_label, alignment=Qt.AlignCenter)

    def start_movie(self):
        self.loading_movie.start()

    def stop_movie(self):
        self.loading_movie.stop()
