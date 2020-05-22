import torch
import torch.nn as nn


class BottleNeck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()

        self.bottleneck = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm2d(4 * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return torch.cat([x, self.bottleneck(x)], 1)


class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.transition_layers = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.transition_layers(x)


class DenseNet(nn.Module):
    def __init__(self, block_nums, growth_rate, theta, num_class=10):
        super().__init__()

        self.growth_rate = growth_rate
        channels = 2 * growth_rate

        self.conv_input = nn.Conv2d(3, channels, kernel_size=3, padding=1, bias=False)

        self.features = nn.Sequential()

        for i in range(len(block_nums) - 1):
            self.features.add_module("dense_block_" + str(i), self.generate_layer(channels, block_nums[i]))
            channels += growth_rate * block_nums[i]

            out_channels = int(theta * channels)
            self.features.add_module("transition_layer_{}".format(i), Transition(channels, out_channels))
            channels = out_channels

        self.features.add_module("dense_block_" + str(len(block_nums) - 1),
                                 self.generate_layer(channels, block_nums[-1]))
        channels += growth_rate * block_nums[-1]
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.output = nn.Linear(channels, num_class)

    def generate_layer(self, in_channels, block_num):
        layers = []
        for i in range(block_num):
            layers.append(BottleNeck(in_channels, self.growth_rate))
            in_channels += self.growth_rate

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv_input(x)
        output = self.features(output)
        output = self.bn(output)
        output = self.relu(output)
        output = self.avg_pool(output)
        output = output.reshape(output.size(0), -1)
        output = self.output(output)

        return output


def densenet_121():
    return DenseNet([6, 12, 24, 16], growth_rate=12, theta=0.5)


def densenet_169():
    return DenseNet([6, 12, 32, 32], growth_rate=32, theta=0.5)


def densenet_201():
    return DenseNet([6, 12, 48, 32], growth_rate=32, theta=0.5)
