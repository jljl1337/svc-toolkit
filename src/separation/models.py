import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.aggregation import MeanMetric

def _down_layer(in_channels, out_channels, kernel_size=5, stride=2, padding=2):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
    )

def _up_layer(in_channels, out_channels, kernel_size=5, stride=2, padding=1, dropout=False, last=False):
    layers = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
    )

    if dropout:
        layers.append(nn.Dropout(0.5))

    layers.append(nn.Sigmoid() if last else nn.ReLU(inplace=True))

    return layers

class UNet(nn.Module):
    def __init__(self, channels = 1):
        super(UNet, self).__init__()

        self.down1 = _down_layer(channels, 16)

        self.down2 = _down_layer(16, 32)

        self.down3 = _down_layer(32, 64)

        self.down4 = _down_layer(64, 128)

        self.down5 = _down_layer(128, 256)

        self.down6 = _down_layer(256, 512)

        self.up1 = _up_layer(512, 256, dropout=True)

        self.up2 = _up_layer(512, 128, dropout=True)

        self.up3 = _up_layer(256, 64, dropout=True)

        self.up4 = _up_layer(128, 32)

        self.up5 = _up_layer(64, 16)

        self.up6 = _up_layer(32, channels, last=True)

    def forward(self, x):
        x_down1 = self.down1(x)

        x_down2 = self.down2(x_down1)

        x_down3 = self.down3(x_down2)

        x_down4 = self.down4(x_down3)

        x_down5 = self.down5(x_down4)

        x_down6 = self.down6(x_down5)

        x_up1 = self.up1(x_down6)
        x_up1 = x_up1[:, :, : -1, : -1]

        x_up2 = torch.cat((x_up1, x_down5), 1)
        x_up2 = self.up2(x_up2)
        x_up2 = x_up2[:, :, : -1, : -1]

        x_up3 = torch.cat((x_up2, x_down4), 1)
        x_up3 = self.up3(x_up3)
        x_up3 = x_up3[:, :, : -1, : -1]

        x_up4 = torch.cat((x_up3, x_down3), 1)
        x_up4 = self.up4(x_up4)
        x_up4 = x_up4[:, :, : -1, : -1]

        x_up5 = torch.cat((x_up4, x_down2), 1)
        x_up5 = self.up5(x_up5)
        x_up5 = x_up5[:, :, : -1, : -1]

        x_up6 = torch.cat((x_up5, x_down1), 1)
        x_up6 = self.up6(x_up6)
        x_up6 = x_up6[:, :, : -1, : -1]

        # print(x_down1.shape)
        # print(x_down2.shape)
        # print(x_down3.shape)
        # print(x_down4.shape)
        # print(x_down5.shape)
        # print(x_down6.shape)
        # print(x_up1.shape)
        # print(x_up2.shape)
        # print(x_up3.shape)
        # print(x_up4.shape)
        # print(x_up5.shape)

        return x_up6

class DeeperUNet(nn.Module):
    def __init__(self, channels = 1):
        super(DeeperUNet, self).__init__()

        self.down1 = _down_layer(channels, 16)

        self.down2 = _down_layer(16, 32)

        self.down3 = _down_layer(32, 64)

        self.down4 = _down_layer(64, 128)

        self.down5 = _down_layer(128, 256)

        self.down6 = _down_layer(256, 512)

        self.down7 = _down_layer(512, 1024)

        self.down8 = _down_layer(1024, 2048)

        self.up1 = _up_layer(2048, 1024, dropout=True)

        self.up2 = _up_layer(2048, 512, dropout=True)

        self.up3 = _up_layer(1024, 256, dropout=True)

        self.up4 = _up_layer(512, 128, dropout=True)

        self.up5 = _up_layer(256, 64)

        self.up6 = _up_layer(128, 32)

        self.up7 = _up_layer(64, 16)

        self.up8 = _up_layer(32, channels, last=True)

    def forward(self, x):
        x_down1 = self.down1(x)

        x_down2 = self.down2(x_down1)

        x_down3 = self.down3(x_down2)

        x_down4 = self.down4(x_down3)

        x_down5 = self.down5(x_down4)

        x_down6 = self.down6(x_down5)

        x_down7 = self.down7(x_down6)

        x_down8 = self.down8(x_down7)

        x_up1 = self.up1(x_down8)
        x_up1 = x_up1[:, :, : -1, : -1]

        x_up2 = torch.cat((x_up1, x_down7), 1)
        x_up2 = self.up2(x_up2)
        x_up2 = x_up2[:, :, : -1, : -1]

        x_up3 = torch.cat((x_up2, x_down6), 1)
        x_up3 = self.up3(x_up3)
        x_up3 = x_up3[:, :, : -1, : -1]

        x_up4 = torch.cat((x_up3, x_down5), 1)
        x_up4 = self.up4(x_up4)
        x_up4 = x_up4[:, :, : -1, : -1]

        x_up5 = torch.cat((x_up4, x_down4), 1)
        x_up5 = self.up5(x_up5)
        x_up5 = x_up5[:, :, : -1, : -1]

        x_up6 = torch.cat((x_up5, x_down3), 1)
        x_up6 = self.up6(x_up6)
        x_up6 = x_up6[:, :, : -1, : -1]

        x_up7 = torch.cat((x_up6, x_down2), 1)
        x_up7 = self.up7(x_up7)
        x_up7 = x_up7[:, :, : -1, : -1]

        x_up8 = torch.cat((x_up7, x_down1), 1)
        x_up8 = self.up8(x_up8)
        x_up8 = x_up8[:, :, : -1, : -1]

        # print(x_down1.shape)
        # print(x_down2.shape)
        # print(x_down3.shape)
        # print(x_down4.shape)
        # print(x_down5.shape)
        # print(x_down6.shape)
        # print(x_down7.shape)
        # print(x_down8.shape)
        # print(x_up1.shape)
        # print(x_up2.shape)
        # print(x_up3.shape)
        # print(x_up4.shape)
        # print(x_up5.shape)
        # print(x_up6.shape)
        # print(x_up7.shape)

        return x_up8

class UNetLightning(pl.LightningModule):
    def __init__(self, in_channels=1, lr=0.0001, weight_decay=0.00001, deeper=False, optimizer='adam'):
        super(UNetLightning, self).__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.weight_decay = weight_decay
        self.optimizer = optimizer

        self.model = DeeperUNet(in_channels) if deeper else UNet(in_channels)
        self.loss = nn.L1Loss()
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        if self.optimizer == 'adam':
            return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.optimizer == 'adamw':
            return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        else:
            raise ValueError(f'Invalid optimizer: {self.optimizer}')

    def get_loss(self, batch):
        x, y = batch
        y_hat = self.forward(x) * x
        return self.loss(y_hat, y)

    def training_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.train_loss.update(loss)
        return loss

    def on_train_epoch_end(self):
        train_loss = self.train_loss.compute()
        self.train_loss.reset()
        self.log('train_loss', train_loss)

    def validation_step(self, batch, batch_idx):
        loss = self.get_loss(batch)
        self.val_loss.update(loss)
        return loss
    
    def on_validation_epoch_end(self):
        val_loss = self.val_loss.compute()
        self.val_loss.reset()
        self.log('val_loss', val_loss)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = None):
        return self.forward(batch)