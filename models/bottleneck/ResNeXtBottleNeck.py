from torch import nn


class ResNeXtBottleNeck(nn.Module):
    def __init__(self, in_places, places, stride=1, expansion=2, cardinality=32):
        super(ResNeXtBottleNeck, self).__init__()
        self.expansion = expansion

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False,
                      groups=cardinality),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places * self.expansion),
        )

        self.downsample = nn.Sequential(
            nn.Conv2d(in_channels=in_places, out_channels=places * self.expansion, kernel_size=1, stride=stride,
                      bias=False),
            nn.BatchNorm2d(places * self.expansion)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.bottleneck(x)

        residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out
