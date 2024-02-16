import torch
from torch import nn
import copy


class ExpandLayer(nn.Module):
    def __init__(self, model, in_places, places, dropout_prob=0.5, n_iter=3):
        super(ExpandLayer, self).__init__()
        self.n_iter = n_iter
        self.dropout_prob = dropout_prob
        self.conv1 = nn.Conv2d(in_channels=in_places, out_channels=places, kernel_size=1, stride=1, bias=False)
        self.block = nn.ModuleList()
        for i in range(n_iter):
            self.block.append(copy.deepcopy(model))

    def forward(self, x):
        feature = dict(
            no_padding=[],
            padding=[],
        )
        x = self.conv1(x)
        for i in range(self.n_iter):
            if self.training:
                if i == 0:
                    feature["padding"].append(self.block[i](x))
                    feature["no_padding"].append(self.block[i](x))
                elif torch.rand(1).item() < self.dropout_prob:
                    feature["padding"].append(torch.zeros_like(x))
                else:
                    feature["padding"].append(self.block[i](x))
                    feature["no_padding"].append(self.block[i](x))
            else:
                feature["padding"].append(self.block[i](x))
                feature["no_padding"].append(self.block[i](x))
        return feature

    def drop_forward(self, x, drop=0.3):
        feature = dict(
            no_padding=[],
            padding=[],
        )
        x = self.conv1(x)
        for i in range(self.n_iter):

            if i == 0:
                feature["padding"].append(self.block[i](x))
                feature["no_padding"].append(self.block[i](x))
            elif torch.rand(1).item() < drop:
                feature["padding"].append(torch.zeros_like(x))
            else:
                feature["padding"].append(self.block[i](x))
                feature["no_padding"].append(self.block[i](x))

        return feature

    def get_model(self):
        return self.block
