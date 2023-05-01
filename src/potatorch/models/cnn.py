import torch
from torch import nn
from torch.jit import Final
from torchvision.ops import SqueezeExcitation

class DepthwiseSeparable2d(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size=(2,2),
                 stride=1,
                 dilation=1,
                 activation=nn.ReLU(),
                 padding='same',
                 normalize=True,
                 pool=False,
                 se_layer=False):
        super(DepthwiseSeparable2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.activation = activation
        self.padding = padding
        self.normalize = normalize
        self.pool = pool

        # depthwise
        self.depthwise = nn.Conv2d(
                self.in_channels, self.in_channels,
                kernel_size=self.kernel_size,
                stride=stride,
                dilation=dilation,
                bias=(not self.normalize),
                padding=self.padding,
                groups=self.in_channels
                )
        # pointwise
        self.pointwise = nn.Conv2d(
                self.in_channels, self.out_channels,
                kernel_size=1,
                stride=stride,
                dilation=dilation,
                bias=(not self.normalize)
                )

        if self.normalize:
            self.norm_layer = nn.BatchNorm2d(self.out_channels)
            # self.norm_layer = nn.GroupNorm(1, self.out_channels)
        if self.pool:
            self.pool_layer = nn.MaxPool2d(kernel_size=(2,2))

    def forward(self, x):
        x = self.depthwise(x)

        if self.activation:
            x = self.activation(x)

        x = self.pointwise(x)

        if self.normalize:
            x = self.norm_layer(x)

        # TODO: try activation also after pointwise conv

        if self.pool:
            x = self.pool_layer(x)
        return x




class Conv2dBlock(nn.Module):
    normalize: Final[bool]
    pool: Final[bool]
    se_layer: Final[nn.Module]

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size=(2,2),
                 stride=1,
                 dilation=1,
                 activation=nn.ReLU(),
                 padding='same',
                 normalize=True,
                 pool=False,
                 se_layer=False):
        super(Conv2dBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation
        self.activation = activation
        self.padding = padding
        self.normalize = normalize
        self.pool = pool
        self.se_layer = se_layer

        self.conv = nn.Conv2d(
                self.in_channels, self.out_channels,
                kernel_size=self.kernel_size,
                stride=stride,
                dilation=dilation,
                bias=(not self.normalize),
                padding=self.padding
                )
        if self.normalize:
            self.norm_layer = nn.BatchNorm2d(self.out_channels)
            # self.norm_layer = nn.GroupNorm(1, self.out_channels)

        if self.se_layer:
            squeeze_channels = max(1, self.out_channels // 12)
            self.se = SqueezeExcitation(self.out_channels, squeeze_channels)

        if self.pool:
            self.pool_layer = nn.MaxPool2d(kernel_size=(2,2))


    def forward(self, x):
        x = self.conv(x)
        if self.activation is not None:
            x = self.activation(x)
        if self.normalize:
            x = self.norm_layer(x)
        if self.se_layer:
            x = self.se(x)
        if self.pool:
            x = self.pool_layer(x)
        return x


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        self.conv_stack = nn.Sequential(
                Conv2dBlock(12, 32, padding='same', activation=nn.ReLU()),
                Conv2dBlock(32, 32, padding='same', activation=nn.ReLU()),
                Conv2dBlock(32, 64, padding='same', activation=nn.ReLU()),
                Conv2dBlock(64, 128, padding='same', activation=nn.ReLU(), pool=True),
                Conv2dBlock(128, 128, padding='same', activation=nn.ReLU()),
                Conv2dBlock(128, 128, padding='same', activation=nn.ReLU()),
                Conv2dBlock(128, 128, padding='same', activation=nn.ReLU()),
                Conv2dBlock(128, 256, padding='same', activation=nn.ReLU())
                )

        self.linear_output_stack = nn.Sequential(
                nn.Linear(256*4*4 + 3, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
                )

    def forward(self, x, y):
        x = self.conv_stack(x)
        x = torch.flatten(x, start_dim=1)
        x = torch.cat([x, y], 1)
        y = self.linear_output_stack(x)
        return y
