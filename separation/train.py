import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from torch.utils.data import DataLoader

import constants
from data import MusicDataset
from model import UNet, CombinedLoss

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
    load_dotenv()
    TRAIN_DIR = os.getenv('TRAIN_DIR')
    MODEL_DIR = os.getenv('MODEL_DIR')
    BATCH_SIZE = int(os.getenv('BATCH_SIZE'))
    EPOCHS = int(os.getenv('EPOCHS'))
    LOADER_NUM_WORKERS = int(os.getenv('LOADER_NUM_WORKERS'))

    LEARNING_RATE = constants.LEARNING_RATE
    WEIGHT_DECAY = constants.WEIGHT_DECAY

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(constants.SEED)
    np.random.seed(constants.SEED)

    dataset = MusicDataset(TRAIN_DIR)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=LOADER_NUM_WORKERS, pin_memory=True)

    model_list = nn.ModuleList()
    for i in range(constants.MODEL_NUM):
        model_list.append(UNet())

    criterion = nn.L1Loss()
    combined_loss = CombinedLoss(model_list, criterion)
    optimizer = torch.optim.Adam(combined_loss.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)

    df = pd.DataFrame(columns=['epoch', 'loss'])

    for epoch in range(EPOCHS):
        loss = train(model_list, epoch, loader, combined_loss, optimizer,
                     device, MODEL_DIR)
        df.loc[len(df)] = [epoch, loss]
        df.to_csv(f'{MODEL_DIR}/loss.csv', index=False)
    
if __name__ == '__main__':
    main()