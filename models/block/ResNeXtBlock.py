from torch import nn

from models.bottleneck.ResNeXtBottleNeck import ResNeXtBottleNeck


class ResNeXtBlock(nn.Module):
    def __init__(self,in_places,places, n_layer=3,stride=1, expansion = 2, cardinality=32):
        super(ResNeXtBlock,self).__init__()
        self.expansion = expansion
        self.n_layer = n_layer


        self.bottleneck = nn.Sequential()
        for _ in range(n_layer):
            self.bottleneck.append(ResNeXtBottleNeck(in_places=in_places,places=places,stride=stride,expansion=expansion,cardinality=cardinality))
            self.bottleneck.append(nn.Conv2d(in_channels=expansion*places, out_channels=places, kernel_size=1, stride=1, bias=False))

    def forward(self, x):
        out = self.bottleneck(x)

        return out
