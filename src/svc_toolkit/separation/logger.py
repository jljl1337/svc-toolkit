import os

import pandas as pd
import matplotlib.pyplot as plt
from lightning_fabric.utilities.logger import _convert_params
from pytorch_lightning.loggers.logger import Logger
from pytorch_lightning.loggers.csv_logs import ExperimentWriter
from pytorch_lightning.utilities import rank_zero_only

class MyLogger(Logger):
    def __init__(self, save_dir, old_dir=''):
        super().__init__()
        if old_dir != '':
            self.df = pd.read_csv(os.path.join(old_dir, 'loss.csv'))
        else:
            self.df = pd.DataFrame(columns=['epoch', 'train_loss', 'val_loss'])
        self.dir = save_dir
        self.experiment = ExperimentWriter(save_dir)

    @property
    def name(self):
        return "MyLogger"

    @property
    def version(self):
        return "0.1"

    @rank_zero_only
    def log_hyperparams(self, params):
        self.experiment.log_hparams(_convert_params(params))
        self.experiment.save()

    @rank_zero_only
    def log_metrics(self, metrics, step):

        epoch = metrics['epoch']
        if epoch not in self.df['epoch'].values:
            self.df.loc[len(self.df)] = [epoch, None, None]

        loss_name = 'train_loss' if 'train_loss' in metrics else 'val_loss'
        loss = metrics[loss_name]
        
        self.df.loc[self.df['epoch'] == epoch, loss_name] = loss

    @rank_zero_only
    def save(self):
        csv_path = os.path.join(self.dir, 'loss.csv')
        self.df.to_csv(csv_path, index=False)

        graph_path = os.path.join(self.dir, 'loss.png')
        plt.figure(figsize=(10, 5))
        plt.plot(self.df['epoch'], self.df['train_loss'], label='train loss')
        plt.plot(self.df['epoch'], self.df['val_loss'], label='validation loss')
        plt.axhline(y=self.df['val_loss'].min(), color='grey', linestyle='--', label='lowest validation loss')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.title('Loss')
        plt.legend()
        plt.savefig(graph_path)
        plt.close()
