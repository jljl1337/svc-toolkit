import time

from PySide6.QtGui import QMovie

from vc_toolkit.widget.loading_overlay import LoadingOverlayWidget

def test_loading_overlay(qtbot):
    loading_overlay = LoadingOverlayWidget()
    loading_overlay.show()
    qtbot.addWidget(loading_overlay)

    assert loading_overlay.loading_movie.state() == QMovie.MovieState.NotRunning

    loading_overlay.start_movie()
    time.sleep(1)

    assert loading_overlay.loading_movie.state() == QMovie.MovieState.Running

    loading_overlay.stop_movie()

    assert loading_overlay.loading_movie.state() == QMovie.MovieState.NotRunning