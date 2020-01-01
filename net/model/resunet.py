from torch.nn import Module, Conv2d, ConvTranspose2d
from torch import cat
from .components import Residual_Unit, ResidualBlock, padding


class ResUnet(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        self.resunit1l = Residual_Unit(64, 64, padding=1)
        self.resblock2l = ResidualBlock(64, 128, f_stride=2, padding=1)
        self.resblock3l = ResidualBlock(128, 256, f_stride=2, padding=1)
        self.resbridge = ResidualBlock(256, 512, f_stride=2, padding=1)

        self.up3 = ConvTranspose2d(512, 256, kernel_size=3, stride=2)
        self.up2 = ConvTranspose2d(256, 128, kernel_size=3, stride=2)
        self.up1 = ConvTranspose2d(128, 64, kernel_size=3, stride=2)

        self.resblock3r = ResidualBlock(512, 256, padding=1)
        self.resblock2r = ResidualBlock(256, 128, padding=1)
        self.resblock1r = ResidualBlock(128, 64, padding=1)

        self.final = Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.resunit1l(x)
        l1 = x

        x = self.resblock2l(x)
        l2 = x

        x = self.resblock3l(x)
        l3 = x

        x = self.resbridge(x)

        x = self.up3(x)
        x = padding(x, l3)
        x = cat([l3, x], dim=1)
        x = self.resblock3r(x)

        x = self.up2(x)
        x = padding(x, l2)
        x = cat([l2, x], dim=1)
        x = self.resblock2r(x)

        x = self.up1(x)
        x = padding(x, l1)
        x = cat([l1, x], dim=1)
        x = self.resblock1r(x)

        x = self.final(x)
        return x
