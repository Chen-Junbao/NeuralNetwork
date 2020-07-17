import torch
import torch.nn as nn


class ChannelsShuffle(nn.Module):
    def __init__(self, g):
        super().__init__()

        self.g = g

    def forward(self, x):
        x = x.reshape(x.shape[0], self.g, x.shape[1] // self.g, x.shape[2], x.shape[3])
        x = x.transpose(1, 2).contiguous()
        x = x.reshape(x.shape[0], -1, x.shape[3], x.shape[4])

        return x


class ShuffleNetUnit(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups):
        super().__init__()

        self.GConv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 4, 1, groups=groups),
            nn.BatchNorm2d(out_channels // 4),
            nn.ReLU(inplace=True)
        )

        self.channels_shuffle = ChannelsShuffle(groups)

        self.DWConv = nn.Sequential(
            nn.Conv2d(out_channels // 4, out_channels // 4, 3, groups=out_channels // 4, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels // 4),
        )

        self.GConv2 = nn.Sequential(
            nn.Conv2d(out_channels // 4, out_channels, 1, groups=groups),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()

        self.fusion = self._add

        self.activation = nn.ReLU(inplace=True)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

            self.GConv2 = nn.Sequential(
                nn.Conv2d(out_channels // 4, out_channels - in_channels, 1, groups=groups),
                nn.BatchNorm2d(out_channels - in_channels)
            )

            self.fusion = self._cat

    @staticmethod
    def _add(x, y):
        return torch.add(x, y)

    @staticmethod
    def _cat(x, y):
        return torch.cat([x, y], dim=1)

    def forward(self, x):
        shortcut = self.shortcut(x)

        output = self.GConv1(x)
        output = self.channels_shuffle(output)
        output = self.DWConv(output)
        output = self.GConv2(output)

        output = self.fusion(shortcut, output)
        output = self.activation(output)

        return output


class ShuffleNet(nn.Module):
    def __init__(self, num_class, groups=3):
        super().__init__()

        out_channels = []
        if groups == 1:
            out_channels = [24, 144, 288, 567]
        elif groups == 2:
            out_channels = [24, 200, 400, 800]
        elif groups == 3:
            out_channels = [24, 240, 480, 960]
        elif groups == 4:
            out_channels = [24, 272, 544, 1088]
        elif groups == 8:
            out_channels = [24, 384, 768, 1536]

        self.block1 = nn.Sequential(
            nn.Conv2d(3, out_channels[0], 3, padding=1),
            nn.BatchNorm2d(out_channels[0]),
            nn.ReLU(inplace=True)
        )
        self.in_channels = out_channels[0]

        self.block2 = self.generate_layer(4, out_channels[1], stride=2, groups=groups)

        self.block3 = self.generate_layer(8, out_channels[2], stride=2, groups=groups)

        self.block4 = self.generate_layer(4, out_channels[3], stride=2, groups=groups)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(out_channels[3], num_class)

    def generate_layer(self, num_block, out_channels, stride, groups):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for stride in strides:
            layers.append(ShuffleNetUnit(self.in_channels, out_channels, stride=stride, groups=groups)),
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.block1(x)
        output = self.block2(output)
        output = self.block3(output)
        output = self.block4(output)
        output = self.avg_pool(output)
        output = output.reshape(output.size(0), -1)
        output = self.fc(output)

        return output
