import os
from argparse import ArgumentParser
from datetime import datetime

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from svc_toolkit.utility.functions import load_yaml, save_yaml
from svc_toolkit.separation import constants
from svc_toolkit.separation import utility
from svc_toolkit.separation.data import MagnitudeDataModule
from svc_toolkit.separation.logger import MyLogger
from svc_toolkit.separation.models import UNetLightning

def main():
    parser = ArgumentParser()
    parser.add_argument('-t', '--train_csv', type=str, required=True)
    parser.add_argument('-v', '--val_csv', type=str, required=True)
    parser.add_argument('-e', '--experiment', type=str, default='exp')
    parser.add_argument('-m', '--model_dir', type=str, default='./model/')
    parser.add_argument('-c', '--config', type=str, default='./config.yml')
    args = parser.parse_args()

    config = load_yaml(args.config)

    # If resuming, use the old config
    resume_path = config['resume_path']
    if resume_path != '':
        config_old = load_yaml(os.path.join(resume_path, 'config.yml'))

        # Only use the epochs from the new config
        new_epochs = config['epochs']
        config = config_old
        config['epochs'] = new_epochs

    # Create directory for this experiment
    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    model_dir = os.path.join(args.model_dir, args.experiment, date_time)
    os.makedirs(model_dir, exist_ok=True)

    # Save config
    save_yaml(config, os.path.join(model_dir, 'config.yml'))

    # Save song lists
    utility.save_song_list(args.train_csv, model_dir, 'train_songs.csv')
    utility.save_song_list(args.val_csv, model_dir, 'val_songs.csv')

    # Load config
    sample_rate = config['sample_rate']
    batch_size = config['batch_size']
    epochs = config['epochs']
    loader_num_workers = config['loader_num_workers']
    deterministic = config['deterministic']

    win_length = config['win_length']
    hop_length = config['hop_length']
    patch_length = config['patch_length']
    expand_factor = config['expand_factor']
    neglect_frequency = config['neglect_frequency']

    learning_rate = config['learning_rate']
    weight_decay = config['weight_decay']
    optimizer = config['optimizer']
    deeper = config['deeper']

    # Set seed
    pl.seed_everything(constants.SEED, workers=True)

    # Load dataset
    data_module = MagnitudeDataModule(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        win_length=win_length,
        hop_length=hop_length,
        patch_length=patch_length,
        expand_factor=expand_factor,
        neglect_frequency=neglect_frequency,
        sample_rate=sample_rate,
        batch_size=batch_size,
        loader_num_workers=loader_num_workers
    )

    # Train model
    model = UNetLightning(lr=learning_rate, weight_decay=weight_decay, deeper=deeper, optimizer=optimizer)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='min')
    model_checkpoint_best = ModelCheckpoint(monitor='val_loss', save_top_k=1, 
                                            mode='min', filename='best-{epoch}',
                                            dirpath=model_dir)
    model_checkpoint_last = ModelCheckpoint(filename='last-{epoch}', dirpath=model_dir)
    callbacks=[model_checkpoint_best, model_checkpoint_last]
    logger = MyLogger(model_dir, resume_path)

    trainer = pl.Trainer(max_epochs=epochs, callbacks=callbacks, logger=logger,
                         devices=[0], deterministic=deterministic, precision='bf16-mixed')

    if resume_path != '':
        model_path = utility.get_last_checkpoint_path(resume_path)
        trainer.fit(model, datamodule=data_module, ckpt_path=model_path)
    else:
        trainer.fit(model, datamodule=data_module)
    
if __name__ == '__main__':
    main()