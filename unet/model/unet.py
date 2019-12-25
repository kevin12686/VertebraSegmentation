from torch.nn import Module, MaxPool2d, Conv2d
from torch import cat
from .components import Double_Conv2d, DeConv2D


class Unet(Module):
    def __init__(self):
        super().__init__()
        self.double1l = Double_Conv2d(1, 64, padding=1)
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

        self.final = Conv2d(64, 1, kernel_size=1)

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
        l4 = self.crop(l4, x)
        x = cat([l4, x], dim=1)
        x = self.double1r(x)

        x = self.up2(x)
        l3 = self.crop(l3, x)
        x = cat([l3, x], dim=1)
        x = self.double2r(x)

        x = self.up3(x)
        l2 = self.crop(l2, x)
        x = cat([l2, x], dim=1)
        x = self.double3r(x)

        x = self.up4(x)
        l1 = self.crop(l1, x)
        x = cat([l1, x], dim=1)
        x = self.double4r(x)

        x = self.final(x)
        return x

    @classmethod
    def crop(cls, target, ref):
        if target.shape == ref.shape:
            return target
        lower1 = int((target.shape[2] - ref.shape[2]) / 2)
        upper1 = int(target.shape[2] - lower1)
        lower2 = int((target.shape[3] - ref.shape[3]) / 2)
        upper2 = int(target.shape[3] - lower2)
        diff1 = abs(upper1 - lower1 - ref.shape[2])
        diff2 = abs(upper2 - lower2 - ref.shape[3])
        return target[:, :, lower1 + diff1:upper1, lower2 + diff2:upper2]
