import os

import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.trainer.states import TrainerFn
from torch.utils.data import DataLoader

from svc_toolkit.separation.data import MagnitudeRandomDataset, MagnitudeDataModule
from svc_toolkit.separation.constants import NeglectFrequency, CSVColumns

CURRENT_DIR = os.path.dirname(__file__)

def test_load_magnitude_random_dataset():
    # Create a MagnitudeRandomDataset object
    dataset = MagnitudeRandomDataset([], [], 10, 10, 10, 10, NeglectFrequency.NYQUIST, 10)

    test_audio_path = os.path.join(CURRENT_DIR, '../source.wav')
    
    index, mix_mag, stem_mag, weight = dataset.load_magnitude(0, test_audio_path, test_audio_path)

    assert index == 0
    assert mix_mag.shape == stem_mag.shape
    assert weight == 2

    dataset = MagnitudeRandomDataset([], [], 10, 10, 10, 10, NeglectFrequency.ZERO, 10)
    index, mix_mag_zero, stem_mag_zero, weight = dataset.load_magnitude(0, test_audio_path, test_audio_path)

    assert index == 0
    assert mix_mag_zero.shape == stem_mag_zero.shape
    assert weight == 2

    assert mix_mag.shape == mix_mag_zero.shape
    assert stem_mag.shape == stem_mag_zero.shape

    assert not (mix_mag == mix_mag_zero).all()

def test_random_dataset_constructor():
    test_audio_path = os.path.join(CURRENT_DIR, '../source.wav')

    try:
        exception_str = ''
        # Create a MagnitudeRandomDataset object
        dataset = MagnitudeRandomDataset([test_audio_path], [test_audio_path], 10, 10, 10, 10, 'random string', 10)
    except Exception as e:
        exception_str = str(e)

    assert exception_str.startswith('Invalid neglect_frequency')

    try:
        exception_str = ''
        # Create a MagnitudeRandomDataset object
        dataset = MagnitudeRandomDataset([test_audio_path], [], 10, 10, 10, 10, NeglectFrequency.NYQUIST, 10)
    except Exception as e:
        exception_str = str(e)

    assert exception_str == 'Length of mixture and stem lists must be equal'

    # Create a MagnitudeRandomDataset object
    dataset = MagnitudeRandomDataset([test_audio_path], [test_audio_path], 10, 10, 10, 10, NeglectFrequency.NYQUIST, 10)

    assert dataset.win_length == 10
    assert dataset.hop_length == 10
    assert dataset.patch_length == 10
    assert dataset.expand_factor == 10
    assert dataset.neglect_frequency == NeglectFrequency.NYQUIST
    assert dataset.sample_rate == 10

    assert len(dataset.expanded_magnitudes) == 2
    assert len(dataset.magnitudes) == 1

def test_random_dataset_len():
    test_audio_path = os.path.join(CURRENT_DIR, '../source.wav')

    # Create a MagnitudeRandomDataset object
    dataset = MagnitudeRandomDataset([test_audio_path], [test_audio_path], 10, 10, 10, 10, NeglectFrequency.NYQUIST, 10)

    assert len(dataset) == 2

def test_random_dataset_get_item():
    pl.seed_everything(56615230)

    test_audio_path = os.path.join(CURRENT_DIR, '../source.wav')

    # Create a MagnitudeRandomDataset object
    dataset = MagnitudeRandomDataset([test_audio_path], [test_audio_path], 10, 10, 10, 10, NeglectFrequency.NYQUIST, 10)

    mix_mag, stem_mag = dataset[0]

    assert mix_mag.shape == stem_mag.shape

    mix_mag_new, stem_mag_new = dataset[0]

    assert mix_mag_new.shape == stem_mag_new.shape

    assert torch.equal(mix_mag, mix_mag_new) == False

def test_magnitude_data_module_constructor():
    # Create a MagnitudeDataModule object
    data_module = MagnitudeDataModule('', '', 1, 2, 3, 4, 5, 6, 7, 8)

    assert data_module.train_csv == ''
    assert data_module.val_csv == ''
    assert data_module.expand_factor == 1
    assert data_module.win_length == 2
    assert data_module.hop_length == 3
    assert data_module.patch_length == 4
    assert data_module.neglect_frequency == 5
    assert data_module.sample_rate == 6
    assert data_module.batch_size == 7
    assert data_module.loader_num_workers == 8

    assert data_module.train_dataset is None
    assert data_module.val_dataset is None

    assert isinstance(data_module.dataset_kwargs, dict)
    
def test_magnitude_data_module_setup():
    # Create a empty dataframe with columns
    df = pd.DataFrame(columns=[CSVColumns.MIXTURE_PATH, CSVColumns.STEM_PATH])
    df_path = os.path.join(CURRENT_DIR, 'test.csv')
    df.to_csv(df_path, index=False)

    # Create a MagnitudeDataModule object
    data_module = MagnitudeDataModule(df_path, df_path, 1, 2, 3, 4, NeglectFrequency.NYQUIST, 6, 7, 8)

    data_module.setup(TrainerFn.FITTING)

    assert data_module.train_dataset is not None
    assert data_module.val_dataset is not None

    os.remove(df_path)

def test_magnitude_data_module_loader_kwargs():
    # Create a MagnitudeDataModule object
    data_module = MagnitudeDataModule('', '', 1, 2, 3, 4, NeglectFrequency.NYQUIST, 6, 7, 8)

    loader_kwargs = data_module._loader_kwargs(shuffle=True)

    assert loader_kwargs['batch_size'] == 7
    assert loader_kwargs['num_workers'] == 8
    assert loader_kwargs['shuffle'] == True

    loader_kwargs = data_module._loader_kwargs(shuffle=False)

    assert loader_kwargs['batch_size'] == 7
    assert loader_kwargs['num_workers'] == 8
    assert loader_kwargs['shuffle'] == False

def test_magnitude_data_module_dataloaders():
    # Create a dataframe with only one row
    df = pd.DataFrame(columns=[CSVColumns.MIXTURE_PATH, CSVColumns.STEM_PATH])
    source_path = os.path.join(CURRENT_DIR, '../source.wav')
    df.loc[0] = [source_path, source_path]
    df_path = os.path.join(CURRENT_DIR, 'test.csv')
    df.to_csv(df_path, index=False)

    # Create a MagnitudeDataModule object
    data_module = MagnitudeDataModule(df_path, df_path, 1, 2, 3, 4, NeglectFrequency.NYQUIST, 6, 7, 8)

    data_module.setup(TrainerFn.FITTING)

    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()

    assert isinstance(train_loader, DataLoader)
    assert isinstance(val_loader, DataLoader)

    os.remove(df_path)