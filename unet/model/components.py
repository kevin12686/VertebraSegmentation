from torch.nn import Sequential, Conv2d, ConvTranspose2d, ReLU


def Double_Conv2d(in_channels, out_channels, padding=0):
    return Sequential(
        Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=padding),
        ReLU(inplace=True),
        Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=padding),
        ReLU(inplace=True),
    )


def DeConv2D(in_channels, out_channels):
    return Sequential(
        ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
        ReLU(inplace=True),
    )
