import torch.nn as nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        return self.residual_function(x) + self.shortcut(x)


class BottleBlock(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()

        self.residual_function = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleBlock.expansion, kernel_size=1, bias=False)
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleBlock.expansion:
            self.shortcut = nn.Sequential(
                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels, out_channels * BottleBlock.expansion, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        return self.residual_function(x) + self.shortcut(x)


class ResNet(nn.Module):
    def __init__(self, block, num_block, num_class):
        super().__init__()

        self.in_channels = 64

        self.conv_input = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv_block1 = self.generate_layer(block, 64, num_block[0], 1)

        self.conv_block2 = self.generate_layer(block, 128, num_block[1], 2)

        self.conv_block3 = self.generate_layer(block, 256, num_block[2], 2)

        self.conv_block4 = self.generate_layer(block, 512, num_block[3], 2)

        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)

        self.fc = nn.Linear(512 * block.expansion, num_class)

        # Kaiming He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def generate_layer(self, block, out_channels, num_block, stride):
        """
        Generate ResNet layers. The stride of first block could be 1 or 2, the others would always be 1.
        :param block: the type of block (BasicBlock or BottleBlock)
        :param out_channels: output channels of this layer
        :param num_block: the number of blocks per layer
        :param stride: the stride of the first block
        :return: a ResNet layer
        """
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv_input(x)
        output = self.conv_block1(output)
        output = self.conv_block2(output)
        output = self.conv_block3(output)
        output = self.conv_block4(output)
        output = self.avg_pool(output)
        output = output.reshape(output.size(0), -1)
        output = self.fc(output)

        return output


def resnet_18(num_class):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_class)


def resnet_34(num_class):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_class)


def resnet_50(num_class):
    return ResNet(BottleBlock, [3, 4, 6, 3], num_class)


def resnet_101(num_class):
    return ResNet(BottleBlock, [3, 4, 23, 3], num_class)


def resnet_152(num_class):
    return ResNet(BottleBlock, [3, 8, 36, 3], num_class)
