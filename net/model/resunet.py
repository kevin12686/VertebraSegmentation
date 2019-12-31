from torch.nn import Module, Conv2d, UpsamplingBilinear2d
from torch import cat
from .components import Residual_Unit, ResidualBlock


class ResUnet(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, padding=1)
        self.resunit1l = Residual_Unit(64, 64, padding=1)
        self.resblock2l = ResidualBlock(64, 128, f_stride=2, padding=1)
        self.resblock3l = ResidualBlock(128, 256, f_stride=2, padding=1)
        self.resbridge = ResidualBlock(256, 512, f_stride=2, padding=1)

        self.upsample = UpsamplingBilinear2d(scale_factor=2)

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

        x = cat([l3, self.upsample(x)])
        x = self.resblock3r(x)

        x = cat([l2, self.upsample(x)])
        x = self.resblock2r(x)

        x = cat([l1, self.upsample(x)])
        x = self.resblock1r(x)

        x = self.final(x)
        return x
