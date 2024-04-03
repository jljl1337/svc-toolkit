import os

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon

from svc_toolkit.widget.loading_window import LoadingWindow

def main():
    app = QApplication([])

    app.setWindowIcon(QIcon(os.path.join(os.path.dirname(__file__), 'img/icon.png')))

    # Show loading window before importing other modules and creating main window
    loading_window = LoadingWindow()
    loading_window.show()
    app.processEvents()

    from svc_toolkit.widget.main_window import MainWindow
    from svc_toolkit.widget.separation import SeparationWidget
    from svc_toolkit.widget.training import TrainingWidget
    from svc_toolkit.widget.conversion import ConversionWidget
    from svc_toolkit.widget.mixing import MixingWidget
    from svc_toolkit.separation.separator import SeparatorFactory
    from svc_toolkit.conversion.converter_trainer import ConverterTrainerFactory
    from svc_toolkit.conversion.converter import ConverterFactory
    from svc_toolkit.conversion.mixer import MixerFactory
    from svc_toolkit.presenter.separation import SeparationPresenter
    from svc_toolkit.presenter.training import TrainingPresenter
    from svc_toolkit.presenter.conversion import ConversionPresenter
    from svc_toolkit.presenter.mixing import MixingPresenter

    vocal_separation_widget = SeparationWidget()
    separator_factory = SeparatorFactory()
    vocal_separation_presenter = SeparationPresenter(vocal_separation_widget, separator_factory)

    training_widget = TrainingWidget()
    trainer_factory = ConverterTrainerFactory()
    trainer_presenter = TrainingPresenter(training_widget, trainer_factory)

    voice_conversion_widget = ConversionWidget()
    converter_factory = ConverterFactory()
    vocal_conversion_presenter = ConversionPresenter(voice_conversion_widget, converter_factory)

    mixing_widget = MixingWidget()
    mixer_factory = MixerFactory()
    mixer_presenter = MixingPresenter(mixing_widget, mixer_factory)

    tab_list = [
        (vocal_separation_widget, 'Separation'),
        (training_widget, 'Training'),
        (voice_conversion_widget, 'Conversion'),
        (mixing_widget, 'Mixing'),
    ]

    # Close loading window after main window is ready
    loading_window.close()

    # Create main window
    window = MainWindow(tab_list)
    window.show()

    exit(app.exec())

if __name__ == '__main__':
    main()