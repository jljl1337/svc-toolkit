import torch
import torch.nn as nn

def down_pad_conv(in_channels, out_channels, kernel_size=5, stride=2, padding=(1, 2, 1, 2)):
    return nn.Sequential(
        nn.ZeroPad2d(padding),
        nn.Conv2d(in_channels, out_channels, kernel_size, stride),
    )

def down_batch_act(out_channels, eps=1e-3, momentum=1e-2):
    return nn.Sequential(
        nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum),
        nn.LeakyReLU(0.2),
    )

def up_layer(in_channels, out_channels, dropout=False, kernel_size=5, stride=2, eps=1e-3, momentum=1e-2):
    layers = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum),
    )

    if dropout:
        layers.add_module(f'dropout{in_channels}', nn.Dropout(0.5))

    return layers

def up_pad(x):
    return x[:, :, 1: -2, 1: -2]

class UNet(nn.Module):
    def __init__(self, in_channels = 2):
        super(UNet, self).__init__()

        self.down_conv1 = down_pad_conv(in_channels, 16)
        self.down_batch1 = down_batch_act(16)

        self.down_conv2 = down_pad_conv(16, 32)
        self.down_batch2 = down_batch_act(32)

        self.down_conv3 = down_pad_conv(32, 64)
        self.down_batch3 = down_batch_act(64)

        self.down_conv4 = down_pad_conv(64, 128)
        self.down_batch4 = down_batch_act(128)

        self.down_conv5 = down_pad_conv(128, 256)
        self.down_batch5 = down_batch_act(256)

        self.down_conv6 = down_pad_conv(256, 512)
        self.down_batch6 = down_batch_act(512)

        self.up1 = up_layer(512, 256, dropout=True)

        self.up2 = up_layer(512, 128, dropout=True)

        self.up3 = up_layer(256, 64, dropout=True)

        self.up4 = up_layer(128, 32)

        self.up5 = up_layer(64, 16)

        self.up6 = up_layer(32, 1)

        self.mask = nn.Conv2d(1, 2, kernel_size=4, dilation=2, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.down_conv1(x)
        x2 = self.down_batch1(x1)

        x3 = self.down_conv2(x2)
        x4 = self.down_batch2(x3)

        x5 = self.down_conv3(x4)
        x6 = self.down_batch3(x5)

        x7 = self.down_conv4(x6)
        x8 = self.down_batch4(x7)

        x9 = self.down_conv5(x8)
        x10 = self.down_batch5(x9)

        x11 = self.down_conv6(x10)
        _x12 = self.down_batch6(x11)

        x13 = self.up1(x11)
        x13 = up_pad(x13)
        x14 = torch.cat((x9, x13), 1)

        x15 = self.up2(x14)
        x15 = up_pad(x15)
        x16 = torch.cat((x7, x15), 1)

        x17 = self.up3(x16)
        x17 = up_pad(x17)
        x18 = torch.cat((x5, x17), 1)

        x19 = self.up4(x18)
        x19 = up_pad(x19)
        x20 = torch.cat((x3, x19), 1)

        x21 = self.up5(x20)
        x21 = up_pad(x21)
        x22 = torch.cat((x1, x21), 1)

        x23 = self.up6(x22)
        x23 = up_pad(x23)

        mask = self.mask(x23)
        mask = self.sigmoid(mask)

        return x * mask


class CombinedLoss(nn.Module):
    def __init__(self, model_list, criterion) -> None:
        super(CombinedLoss, self).__init__()
        self.model_list = model_list
        self.criterion = criterion
        self.model_num = len(model_list)
    
    def forward(self, mixture, separate):
        loss = 0

        for i in range(self.model_num):
            pred = self.model_list[i](mixture)
            loss += self.criterion(pred, separate[i])

        return loss / self.model_num

