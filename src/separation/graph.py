import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dotenv import load_dotenv

def plot_loss_graph(df, path):
    plt.figure(figsize=(20, 10))
    plt.plot(df['epoch'], df['loss'], label='loss')
    # x label epoch
    plt.xlabel('epoch')
    # y label loss
    plt.ylabel('loss')
    plt.savefig(path)

def main():
    load_dotenv()
    MODEL_DIR = os.getenv('MODEL_DIR')
    path = f'{MODEL_DIR}/loss.csv'
    df = pd.read_csv(path)
    df['epoch'] += 1
    plot_loss_graph(df, f'{MODEL_DIR}/loss.png')
    # loss to log loss
    df['loss'] = df['loss'].apply(lambda x: np.log(x))
    plot_loss_graph(df, f'{MODEL_DIR}/log_loss.png')

if __name__ == '__main__':
    main()