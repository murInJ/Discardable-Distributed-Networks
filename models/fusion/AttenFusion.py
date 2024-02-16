from torch import nn
import torch

from models.attention.ChannelAttention import ChannelAttention


class AttenFusion(nn.Module):
    def __init__(self, places, n_iter=3):
        super(AttenFusion, self).__init__()
        self.n_iter = n_iter
        self.conv1 = nn.Conv2d(in_channels=places * n_iter, out_channels=places, kernel_size=1, stride=1, bias=False)
        self.norm = nn.BatchNorm2d(places)
        self.silu = nn.SiLU()
        self.att = ChannelAttention(places * n_iter)
        self.flatten = nn.Flatten()

    def forward(self, feature):
        x = torch.cat(feature, 1)
        x = self.att(x)
        x = self.conv1(x)
        x = self.silu(x)
        x = self.norm(x)

        mu_vars = []

        if self.training:
            if len(feature) != 1:
                for feat in feature:
                    flat = self.flatten(feat)
                    mu = torch.mean(flat, 0)
                    log_var = torch.log(torch.var(flat, 0))
                    mu_vars.append((mu, log_var))

        return mu_vars, x
