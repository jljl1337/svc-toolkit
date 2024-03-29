from PySide6.QtWidgets import QApplication
from widget.loading_window import LoadingWindow

def main():
    app = QApplication([])

    # Show loading window before importing other modules and creating main window
    loading_window = LoadingWindow()
    loading_window.show()
    app.processEvents()

    from widget.main_window import MainWindow
    from widget.vocal_separation import VocalSeparationWidget
    from widget.voice_conversion import VoiceConversionWidget
    from widget.training import TrainingWidget
    from separation.separator import SeparatorFactory
    from presenter.vocal_separation import VocalSeparationPresenter

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

    # Close loading window after main window is ready
    loading_window.close()

    # Create main window
    window = MainWindow(tab_list)
    window.show()

    exit(app.exec())

if __name__ == '__main__':
    main()