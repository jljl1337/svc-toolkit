from PySide6.QtWidgets import QWidget

from widget.main_window import MainWindow

def test_main_window(qtbot):
    dummy_tab = QWidget()
    tab_list = [(dummy_tab, 'Dummy')]

    main_window = MainWindow(tab_list)

    qtbot.addWidget(main_window)

    assert main_window.tab_widget.count() == 1

    assert main_window.tab_widget.tabText(0) == 'Dummy'