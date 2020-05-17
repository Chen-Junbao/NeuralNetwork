import torch
import torch.nn as nn


class Inception(nn.Module):
    def __init__(self, in_channels, channels_1x1, reduce_3x3, channels_3x3, reduce_5x5, channels_5x5, pool_proj):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=channels_1x1, kernel_size=1),
            nn.BatchNorm2d(channels_1x1),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=reduce_3x3, kernel_size=1),
            nn.BatchNorm2d(reduce_3x3),
            nn.ReLU(),
            nn.Conv2d(in_channels=reduce_3x3, out_channels=channels_3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels_3x3),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=reduce_5x5, kernel_size=1),
            nn.BatchNorm2d(reduce_5x5),
            nn.ReLU(),
            nn.Conv2d(in_channels=reduce_5x5, out_channels=channels_5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(channels_5x5),
            nn.ReLU()
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels=in_channels, out_channels=pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU()
        )

    def forward(self, x):
        return torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], dim=1)


class GoogLeNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU()
        )
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)
        self.avg_pool = nn.AvgPool2d(kernel_size=8, stride=1)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(in_features=1024, out_features=10)

    def forward(self, x):
        output = self.input_layer(x)
        output = self.a3(output)
        output = self.b3(output)
        output = self.pool1(output)
        output = self.a4(output)
        output = self.b4(output)
        output = self.c4(output)
        output = self.d4(output)
        output = self.e4(output)
        output = self.pool2(output)
        output = self.a5(output)
        output = self.b5(output)
        output = self.avg_pool(output)
        output = self.dropout(output)
        output = output.reshape(output.size(0), -1)
        output = self.fc(output)

        return output
