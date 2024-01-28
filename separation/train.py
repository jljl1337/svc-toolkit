import os
from argparse import ArgumentParser
from datetime import datetime

import pytorch_lightning as pl
import yaml
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import constants
from data import MagnitudeDataset
from logger import MyLogger
from models import UNetLightning

def main():
    parser = ArgumentParser()
    parser.add_argument('-t', '--train_csv', type=str, default='./train.csv')
    parser.add_argument('-v', '--val_csv', type=str, default='./val.csv')
    parser.add_argument('-e', '--experiment', type=str, default='exp')
    parser.add_argument('-m', '--model_dir', type=str, default='./model/')
    parser.add_argument('-c', '--config', type=str, default='./config.yml')
    parser.add_argument('-r', '--resume', type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)

    # If resuming, use the old config
    if args.resume is not None:
        with open(os.path.join(args.resume, 'config.yml')) as file:
            config_old = yaml.safe_load(file)

        # Only use the epochs from the new config
        new_epochs = config['EPOCHS']
        config = config_old
        config['EPOCHS'] = new_epochs

    # Create directory for this experiment
    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(args.model_dir, args.experiment, date_time)
    os.makedirs(save_dir, exist_ok=True)

    # Save config
    with open(os.path.join(save_dir, 'config.yml'), 'w') as file:
        yaml.safe_dump(config, file, sort_keys=False)

    # Load config
    SAMPLE_RATE = config['SAMPLE_RATE']
    BATCH_SIZE = config['BATCH_SIZE']
    EPOCHS = config['EPOCHS']
    LOADER_NUM_WORKERS = config['LOADER_NUM_WORKERS']

    WIN_LENGTH = config['WIN_LENGTH']
    HOP_LENGTH = config['HOP_LENGTH']
    PATCH_LENGTH = config['PATCH_LENGTH']
    EXPAND_FACTOR = config['EXPAND_FACTOR']

    LEARNING_RATE = config['LEARNING_RATE']
    WEIGHT_DECAY = config['WEIGHT_DECAY']

    # Set seed
    pl.seed_everything(constants.SEED, workers=True)

    # Load dataset
    dataset_train = MagnitudeDataset(args.train_csv, expand_factor=EXPAND_FACTOR, 
                                     win_length=WIN_LENGTH, hop_length=HOP_LENGTH,
                                     patch_length=PATCH_LENGTH, sample_rate=SAMPLE_RATE)
    dataset_val = MagnitudeDataset(args.val_csv, expand_factor=EXPAND_FACTOR, 
                                   win_length=WIN_LENGTH, hop_length=HOP_LENGTH,
                                   patch_length=PATCH_LENGTH, sample_rate=SAMPLE_RATE)

    loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=LOADER_NUM_WORKERS, pin_memory=True)
    loader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=LOADER_NUM_WORKERS, pin_memory=True)

    model = UNetLightning(lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='min')
    model_checkpoint = ModelCheckpoint(monitor='val_loss', save_top_k=1, 
                                       mode='min', filename='best-{epoch}',
                                       save_last=True, dirpath=save_dir)
    logger = MyLogger(save_dir, args.resume)

    trainer = pl.Trainer(max_epochs=EPOCHS, callbacks=[early_stopping, model_checkpoint], logger=logger)

    if args.resume is not None:
        model_path = os.path.join(args.resume, 'last.ckpt')
        trainer.fit(model, loader_train, loader_val, ckpt_path=model_path)
    else:
        trainer.fit(model, loader_train, loader_val)
    
if __name__ == '__main__':
    main()