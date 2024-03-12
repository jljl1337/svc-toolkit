from PySide6.QtWidgets import QApplication

from widget.main_window import MainWindow
from widget.vocal_separation import VocalSeparationWidget
from widget.voice_conversion import VoiceConversionWidget
from widget.training import TrainingWidget
from separation.separator import SeparatorFactory
from presenter.vocal_separation import VocalSeparationPresenter

def main():
    app = QApplication([])
    vocal_separation_widget = VocalSeparationWidget()
    voice_conversion_widget = VoiceConversionWidget()
    training_widget = TrainingWidget()
    separator_factory = SeparatorFactory()
    vocal_separation_presenter = VocalSeparationPresenter(vocal_separation_widget, separator_factory)
    tab_list = [
        (vocal_separation_widget, 'Separation'),
        (voice_conversion_widget, 'Conversion'),
        (training_widget, 'Training')
    ]
    window = MainWindow(tab_list)
    window.show()
    app.exec()

if __name__ == '__main__':
    main()