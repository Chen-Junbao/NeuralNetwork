import torch.nn as nn


class DepthwisePointwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.depthwise = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, groups=in_channels, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        output = self.depthwise(x)
        output = self.pointwise(output)

        return output


class MobileNet(nn.Module):
    def __init__(self, num_class=10):
        super().__init__()

        self.conv_input = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.dp1 = DepthwisePointwiseConv(32, 64)
        self.dp2 = nn.Sequential(
            DepthwisePointwiseConv(64, 128, 2),
            DepthwisePointwiseConv(128, 128)
        )
        self.dp3 = nn.Sequential(
            DepthwisePointwiseConv(128, 256, 2),
            DepthwisePointwiseConv(256, 256)
        )
        self.dp4 = nn.Sequential(
            DepthwisePointwiseConv(256, 512, 2),
            DepthwisePointwiseConv(512, 512),
            DepthwisePointwiseConv(512, 512),
            DepthwisePointwiseConv(512, 512),
            DepthwisePointwiseConv(512, 512),
            DepthwisePointwiseConv(512, 512)
        )
        self.dp5 = nn.Sequential(
            DepthwisePointwiseConv(512, 1024, 2),
            DepthwisePointwiseConv(1024, 1024)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, num_class)

    def forward(self, x):
        output = self.conv_input(x)
        output = self.dp1(output)
        output = self.dp2(output)
        output = self.dp3(output)
        output = self.dp4(output)
        output = self.dp5(output)
        output = self.avg_pool(output)
        output = output.reshape(output.size(0), -1)
        output = self.fc(output)

        return output
