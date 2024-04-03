from svc_toolkit.presenter.training import TrainingPresenter

class TrainingWidgetDummy:
    def set_preprocess_function(self, function):
        self.preprocess = function

    def set_train_function(self, function):
        self.train = function

class ConverterTrainerDummy:
    dataset_dir = None
    output_dir = None
    split = None

    def preprocess(self, dataset_dir, output_dir, split):
        ConverterTrainerDummy.dataset_dir = dataset_dir
        ConverterTrainerDummy.output_dir = output_dir
        ConverterTrainerDummy.split = split

    config_path = None
    model_path = None

    def train(self, config_path, model_path):
        ConverterTrainerDummy.config_path = config_path
        ConverterTrainerDummy.model_path = model_path

class ConverterTrainerFactoryDummy:
    def create(self):
        return ConverterTrainerDummy()
    
def test_training_presenter_constructor():
    # Create a TrainingWidget dummy object
    view = TrainingWidgetDummy()
    # Create a ConverterTrainerFactory dummy object
    trainer_factory = ConverterTrainerFactoryDummy()
    # Create a TrainingPresenter object
    presenter = TrainingPresenter(view, trainer_factory)

    # Check if the view is set
    assert presenter.view == view

    # Check if the preprocess function is set
    assert view.preprocess == presenter.preprocess
    # Check if the train function is set
    assert view.train == presenter.train

def test_training_presenter_preprocess():
    # Create a TrainingWidget dummy object
    view = TrainingWidgetDummy()
    # Create a ConverterTrainerFactory dummy object
    trainer_factory = ConverterTrainerFactoryDummy()
    # Create a TrainingPresenter object
    presenter = TrainingPresenter(view, trainer_factory)

    # Check if the trainer preprocess is called with the correct parameters
    presenter.preprocess('dataset_dir', 'output_dir', True)

    assert ConverterTrainerDummy.dataset_dir == 'dataset_dir'
    assert ConverterTrainerDummy.output_dir == 'output_dir'
    assert ConverterTrainerDummy.split == True

def test_training_presenter_train():
    # Create a TrainingWidget dummy object
    view = TrainingWidgetDummy()
    # Create a ConverterTrainerFactory dummy object
    trainer_factory = ConverterTrainerFactoryDummy()
    # Create a TrainingPresenter object
    presenter = TrainingPresenter(view, trainer_factory)

    # Check if the trainer train is called with the correct parameters
    presenter.train('config_path', 'model_path')

    assert ConverterTrainerDummy.config_path == 'config_path'
    assert ConverterTrainerDummy.model_path == 'model_path'