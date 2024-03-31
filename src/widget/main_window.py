from PySide6.QtWidgets import QMainWindow, QTabWidget

class MainWindow(QMainWindow):
    def __init__(self, tab_list):
        super().__init__()

        self.tab_widget = QTabWidget()

        for tab, name in tab_list:
            self.tab_widget.addTab(tab, name)

        # self.tab_widget.setCurrentIndex(len(tab_list) // 2)
        self.tab_widget.setCurrentIndex(1)

        self.setWindowTitle('Voice Conversion Toolkit')

        self.setCentralWidget(self.tab_widget)

        # self.setFixedSize(self.centralWidget().sizeHint())
        self.setFixedSize(1000, 700)