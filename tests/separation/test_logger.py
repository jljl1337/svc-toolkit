import os
import shutil

import pandas as pd

from svc_toolkit.separation.logger import MyLogger

CURRENT_DIR = os.path.dirname(__file__)

def test_logger_constructor():
    save_dir = os.path.join(CURRENT_DIR, 'save_dir')

    logger = MyLogger(save_dir, '')
    assert logger.dir == save_dir

    old_dir = os.path.join(CURRENT_DIR, 'old_dir')
    os.mkdir(old_dir)
    df = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss'])
    df.to_csv(os.path.join(old_dir, 'loss.csv'), index=False)

    logger = MyLogger(save_dir, old_dir)

    assert logger.dir == save_dir

    shutil.rmtree(old_dir)

    shutil.rmtree(save_dir)

def test_logger_properties():
    save_dir = os.path.join(CURRENT_DIR, 'save_dir')

    logger = MyLogger(save_dir, '')
    assert logger.name == 'MyLogger'
    assert logger.version == '0.1'

    shutil.rmtree(save_dir)

def test_logger_log_hyperparams():
    class DummyExperimentWriter:
        def __init__(self):
            self.hparams = None

        def log_hparams(self, hparams):
            self.hparams = hparams

        def save(self):
            pass

    save_dir = os.path.join(CURRENT_DIR, 'save_dir')

    logger = MyLogger(save_dir, '')
    logger.experiment = DummyExperimentWriter()

    logger.log_hyperparams({
        'a': 1,
        'b': 2
    })

    assert logger.experiment.hparams == {
        'a': 1,
        'b': 2
    }

    shutil.rmtree(save_dir)

def test_logger_log_metrics():
    save_dir = os.path.join(CURRENT_DIR, 'save_dir')

    logger = MyLogger(save_dir, '')

    logger.log_metrics({
        'epoch': 1,
        'train_loss': 0.1
    }, 1)

    logger.log_metrics({
        'epoch': 1,
        'val_loss': 0.2
    }, 1)


    assert logger.df['epoch'].values == [1]
    assert logger.df['train_loss'].values == [0.1]
    assert logger.df['val_loss'].values == [0.2]

    shutil.rmtree(save_dir)

def test_logger_save():
    save_dir = os.path.join(CURRENT_DIR, 'save_dir')

    logger = MyLogger(save_dir, '')

    logger.log_metrics({
        'epoch': 1,
        'train_loss': 0.1
    }, 1)

    logger.log_metrics({
        'epoch': 1,
        'val_loss': 0.2
    }, 1)

    logger.save()

    assert os.path.exists(os.path.join(save_dir, 'loss.csv'))
    assert os.path.exists(os.path.join(save_dir, 'loss.png'))

    shutil.rmtree(save_dir)