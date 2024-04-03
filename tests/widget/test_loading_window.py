from svc_toolkit.widget.loading_window import LoadingWindow

def test_loading_window(qtbot):
    loading_window = LoadingWindow()
    qtbot.addWidget(loading_window)

    assert loading_window.windowTitle() == 'Loading...'