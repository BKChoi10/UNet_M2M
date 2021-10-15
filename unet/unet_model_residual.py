""" Full assembly of the parts to form the complete network """

from .unet_parts import *


class UNet_residual(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_residual, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Downsample(64, 128, 1)
        self.down2 = Downsample(128, 256, 1)
        self.down3 = Downsample(256, 512, 1)
        self.down4 = Downsample(512, 1024, 0)
        self.up1 = Upsamples(1024+512, 512)
        self.up2 = Upsamples(512+256, 256)
        self.up3 = Upsamples(256+128, 128)
        self.outc = OutConv(128, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        logits = self.outc(x)
        return logits
