import torch.nn as nn


class Bottleneck(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, t=6):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, kernel_size=1),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels * t, in_channels * t, 3, groups=in_channels * t, stride=stride, padding=1),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),
            nn.Conv2d(in_channels * t, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU6(inplace=True)
            )

    def forward(self, x):
        return self.residual_function(x) + self.shortcut(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_class):
        super().__init__()

        self.in_channels = 32

        self.conv_input = nn.Conv2d(3, 32, kernel_size=1)

        self.block1 = self.generate_layer(16, 1, 1)

        self.block2 = self.generate_layer(24, 2, 2)

        self.block3 = self.generate_layer(32, 3, 2)

        self.block4 = self.generate_layer(64, 4, 2)

        self.block5 = self.generate_layer(96, 3, 1)

        self.block6 = self.generate_layer(160, 3, 1)

        self.block7 = self.generate_layer(320, 1, 1)

        self.conv_output = nn.Sequential(
            nn.Conv2d(320, 1280, kernel_size=1),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.fc = nn.Linear(1280, num_class)

    def generate_layer(self, out_channels, num_block, stride):
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for stride in strides:
            layers.append(Bottleneck(self.in_channels, out_channels, stride))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv_input(x)
        output = self.block1(output)
        output = self.block2(output)
        output = self.block3(output)
        output = self.block4(output)
        output = self.block5(output)
        output = self.block6(output)
        output = self.block7(output)
        output = self.conv_output(output)
        output = self.avg_pool(output)
        output = output.reshape(output.size(0), -1)
        output = self.fc(output)

        return output
