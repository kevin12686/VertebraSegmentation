from torch.nn import Module, MaxPool2d, Conv2d
import torch
from .components import Double_Conv2d, DeConv2D, padding


class Unet(Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double1l = Double_Conv2d(in_channels, 64, padding=1)
        self.double2l = Double_Conv2d(64, 128, padding=1)
        self.double3l = Double_Conv2d(128, 256, padding=1)
        self.double4l = Double_Conv2d(256, 512, padding=1)
        self.doubleb = Double_Conv2d(512, 1024, padding=1)

        self.maxpooling = MaxPool2d(kernel_size=2, stride=2)

        self.up1 = DeConv2D(1024, 512)
        self.up2 = DeConv2D(512, 256)
        self.up3 = DeConv2D(256, 128)
        self.up4 = DeConv2D(128, 64)

        self.double1r = Double_Conv2d(1024, 512, padding=1)
        self.double2r = Double_Conv2d(512, 256, padding=1)
        self.double3r = Double_Conv2d(256, 128, padding=1)
        self.double4r = Double_Conv2d(128, 64, padding=1)

        self.final = Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        l1 = self.double1l(x)
        x = self.maxpooling(l1)

        l2 = self.double2l(x)
        x = self.maxpooling(l2)

        l3 = self.double3l(x)
        x = self.maxpooling(l3)

        l4 = self.double4l(x)
        x = self.maxpooling(l4)

        x = self.doubleb(x)

        x = self.up1(x)
        x = padding(x, l4)
        x = torch.cat([l4, x], dim=1)
        x = self.double1r(x)

        x = self.up2(x)
        x = padding(x, l3)
        x = torch.cat([l3, x], dim=1)
        x = self.double2r(x)

        x = self.up3(x)
        x = padding(x, l2)
        x = torch.cat([l2, x], dim=1)
        x = self.double3r(x)

        x = self.up4(x)
        x = padding(x, l1)
        x = torch.cat([l1, x], dim=1)
        x = self.double4r(x)

        x = self.final(x)
        return x
