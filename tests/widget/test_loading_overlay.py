from PySide6.QtGui import QMovie

from widget.loading_overlay import LoadingOverlayWidget

def test_loading_overlay(qtbot):
    loading_overlay = LoadingOverlayWidget()
    qtbot.addWidget(loading_overlay)

    assert loading_overlay.loading_label.movie().state() == QMovie.MovieState.NotRunning

    loading_overlay.start_movie()

    assert loading_overlay.loading_label.movie().state() == QMovie.MovieState.Running

    loading_overlay.stop_movie()

    assert loading_overlay.loading_label.movie().state() == QMovie.MovieState.NotRunning