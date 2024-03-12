import os
from argparse import ArgumentParser
from datetime import datetime

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from separation import constants
from separation import utility
from separation.data import MagnitudeDataset
from separation.logger import MyLogger
from separation.models import UNetLightning

def main():
    parser = ArgumentParser()
    parser.add_argument('-t', '--train_csv', type=str, default='./train.csv')
    parser.add_argument('-v', '--val_csv', type=str, default='./val.csv')
    parser.add_argument('-e', '--experiment', type=str, default='exp')
    parser.add_argument('-m', '--model_dir', type=str, default='./model/')
    parser.add_argument('-c', '--config', type=str, default='./config.yml')
    args = parser.parse_args()

    config = utility.load_yaml(args.config)

    # If resuming, use the old config
    resume_path = config['resume_path']
    if resume_path != '':
        config_old = utility.load_yaml(os.path.join(args.resume, 'config.yml'))

        # Only use the epochs from the new config
        new_epochs = config['epochs']
        config = config_old
        config['epochs'] = new_epochs

    # Create directory for this experiment
    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.model_dir, args.experiment, date_time)
    os.makedirs(save_dir, exist_ok=True)

    # Save config
    utility.save_yaml(config, os.path.join(save_dir, 'config.yml'))

    # Load config
    sample_rate = config['sample_rate']
    batch_size = config['batch_size']
    epochs = config['epochs']
    loader_num_workers = config['loader_num_workers']

    win_length = config['win_length']
    hop_length = config['hop_length']
    patch_length = config['patch_length']
    expand_factor = config['expand_factor']

    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    deeper = config['deeper']

    # Set seed
    pl.seed_everything(constants.SEED, workers=True)

    # Load dataset
    dataset_kwargs = {
        'expand_factor': expand_factor,
        'win_length': win_length,
        'hop_length': hop_length,
        'patch_length': patch_length,
        'sample_rate': sample_rate
    }
    dataset_train = MagnitudeDataset(args.train_csv, **dataset_kwargs)
    dataset_val = MagnitudeDataset(args.val_csv, **dataset_kwargs)

    loader_train_kwargs = {
        'batch_size': batch_size,
        'num_workers': loader_num_workers,
        'pin_memory': True,
        'shuffle': True,
    }
    loader_val_kwargs = loader_train_kwargs.copy()
    loader_val_kwargs['shuffle'] = False

    loader_train = DataLoader(dataset_train, **loader_train_kwargs)
    loader_val = DataLoader(dataset_val, **loader_val_kwargs)

    # Train model
    model = UNetLightning(lr=learning_rate, weight_decay=weight_decay, deeper=deeper)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='min')
    model_checkpoint_best = ModelCheckpoint(monitor='val_loss', save_top_k=1, 
                                            mode='min', filename='best-{epoch}',
                                            dirpath=save_dir)
    model_checkpoint_last = ModelCheckpoint(filename='last-{epoch}', dirpath=save_dir)
    logger = MyLogger(save_dir, resume_path)

    trainer = pl.Trainer(max_epochs=epochs, callbacks=[model_checkpoint_best, model_checkpoint_last], logger=logger)

    if resume_path != '':
        model_path = os.path.join(resume_path, 'last.ckpt')
        trainer.fit(model, loader_train, loader_val, ckpt_path=model_path)
    else:
        trainer.fit(model, loader_train, loader_val)
    
if __name__ == '__main__':
    main()