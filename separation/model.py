import torch
import torch.nn as nn
import torch.nn.functional as F

def down_layer(in_channels, out_channels, kernel_size=5, stride=2, padding=2):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
    )

def up_layer(in_channels, out_channels, kernel_size=5, stride=2, padding=1, dropout=False):
    layers = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )

    if dropout:
        layers.insert(2, nn.Dropout(0.5))

    return layers

class UNet(nn.Module):
    def __init__(self, in_channels = 1):
        super(UNet, self).__init__()

        self.down1 = down_layer(in_channels, 16)

        self.down2 = down_layer(16, 32)

        self.down3 = down_layer(32, 64)

        self.down4 = down_layer(64, 128)

        self.down5 = down_layer(128, 256)

        self.down6 = down_layer(256, 512)

        self.up1 = up_layer(512, 256, dropout=True)

        self.up2 = up_layer(512, 128, dropout=True)

        self.up3 = up_layer(256, 64, dropout=True)

        self.up4 = up_layer(128, 32)

        self.up5 = up_layer(64, 16)

        self.up6 = up_layer(32, 1)

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

        return F.sigmoid(x_up6)

# Test the model
if __name__ == '__main__':
    model = UNet()
    print(model)
    print(model(torch.randn(1, 1, 512, 128)).shape)