import os
from argparse import ArgumentParser
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pytorch_lightning as pl
from tqdm import tqdm
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import constants
from data import MagnitudeDataset
from logger import MyLogger
from models import UNetLightning

def train(model_list, epoch, loader, combined_loss, optimizer, device,
          model_dir):

    for m in model_list:
        m.train()

    os.makedirs(model_dir, exist_ok=True)

    loss_sum = 0
    samples_sum = 0

    for i, post_stft_list in tqdm(enumerate(loader)):
        for j in range(len(model_list)):
            model_list[j] = model_list[j].to(device)

        for j in range(len(post_stft_list)):
            post_stft_list[j] = post_stft_list[j].to(device)
            post_stft_list[j] = post_stft_list[j].transpose(2, 3)

        samples_batch_num = len(post_stft_list[0])
        samples_sum += samples_batch_num

        loss = combined_loss(post_stft_list[0], post_stft_list[1:])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item() * samples_batch_num

    epoch_loss = loss_sum / samples_sum
    print(f'Epoch: {epoch:3d} Loss: {epoch_loss:.4f}')

    for i in range(len(model_list)):
        torch.save({'epoch': epoch,
                    'model_state_dict': model_list[i].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': epoch_loss},
                    f'{model_dir}/net_{i}_{epoch:03d}.pth')
    
    return epoch_loss


def main():
    parser = ArgumentParser()
    parser.add_argument('--train_csv', type=str, default='./train.csv')
    parser.add_argument('--val_csv', type=str, default='./val.csv')
    parser.add_argument('-e', '--experiment', type=str, default='exp')
    args = parser.parse_args()

    load_dotenv(override=True)
    TRAIN_DIR = os.getenv('TRAIN_DIR')
    MODEL_DIR = os.getenv('MODEL_DIR')
    BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
    EPOCHS = int(os.getenv('EPOCHS'))
    LOADER_NUM_WORKERS = int(os.getenv('LOADER_NUM_WORKERS'))

    SAMPLE_RATE = int(os.getenv('SAMPLE_RATE'))
    WIN_LENGTH = int(os.getenv('WIN_LENGTH'))
    HOP_LENGTH = int(os.getenv('HOP_LENGTH'))
    PATCH_LENGTH = int(os.getenv('PATCH_LENGTH'))
    EXPAND_FACTOR = int(os.getenv('EXPAND_FACTOR'))

    LEARNING_RATE = float(os.getenv('LEARNING_RATE'))
    WEIGHT_DECAY = float(os.getenv('WEIGHT_DECAY'))

    pl.seed_everything(constants.SEED, workers=True)

    dataset_train = MagnitudeDataset(args.train_csv, expand_factor=EXPAND_FACTOR, 
                                     win_length=WIN_LENGTH, hop_length=HOP_LENGTH,
                                     patch_length=PATCH_LENGTH)
    dataset_val = MagnitudeDataset(args.val_csv, expand_factor=EXPAND_FACTOR, 
                                   win_length=WIN_LENGTH, hop_length=HOP_LENGTH,
                                   patch_length=PATCH_LENGTH)

    loader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=LOADER_NUM_WORKERS, pin_memory=True)
    loader_val = DataLoader(dataset_val, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=LOADER_NUM_WORKERS, pin_memory=True)

    now = datetime.now()
    date_time = now.strftime("%Y%m%d_%H%M%S")
    save_dir = f'{MODEL_DIR}/{args.experiment}/{date_time}'
    os.makedirs(save_dir, exist_ok=True)

    model = UNetLightning(lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    early_stopping = EarlyStopping(monitor='val_loss', patience=20, mode='min')
    model_checkpoint = ModelCheckpoint(monitor='val_loss', save_top_k=1, 
                                       mode='min', filename='best-{epoch}',
                                       save_last=True, dirpath=save_dir)
    logger = MyLogger(save_dir)

    trainer = pl.Trainer(max_epochs=EPOCHS, callbacks=[early_stopping, model_checkpoint], logger=logger)
    trainer.fit(model, loader_train, loader_val)
    
if __name__ == '__main__':
    main()