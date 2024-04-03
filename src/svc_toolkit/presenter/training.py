from svc_toolkit.widget.training import TrainingWidget
from svc_toolkit.conversion.converter_trainer import ConverterTrainerFactory

class TrainingPresenter:
    def __init__(self, view: TrainingWidget, trainer_factory: ConverterTrainerFactory):
        self.view = view
        self.trainer = trainer_factory.create()

        self.view.set_preprocess_function(self.preprocess)
        self.view.set_train_function(self.train)

    def preprocess(self, dataset_dir: str, output_dir: str, split: bool):
        self.trainer.preprocess(dataset_dir, output_dir, split)

    def train(self, config_path: str, model_path: str):
        self.trainer.train(config_path, model_path)