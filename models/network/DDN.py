import torch
from torch import nn

from models.block.ResNeXtBlock import ResNeXtBlock
from models.fusion.AttenFusion import AttenFusion
from models.head.ClassifyHead import ClassifyHead
from models.layer.ExpandLayer import ExpandLayer
from utils.explain_utils import calculate_ifc


class DNN(nn.Module):
    def __init__(self, in_places, depth=[64, 128, 256],dropout_probs=[0.3, 0.4, 0.5], n_inter=3, num_cls=10):
        super(DNN, self).__init__()
        self.l = len(depth)

        self.n_iter = n_inter
        self.conv1 = nn.Conv2d(in_channels=in_places, out_channels=depth[0], kernel_size=1, stride=1,
                               bias=False)  # (W,H)
        self.blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.fusions = nn.ModuleList()
        self.flatten = nn.Flatten()
        for i in range(len(depth)):
            self.blocks.append(ExpandLayer(ResNeXtBlock(in_places=depth[i], places=depth[i]),
                                           in_places=depth[i], places=depth[i], dropout_prob=dropout_probs[i],
                                           n_iter=n_inter))
            if i != len(depth) - 1:
                self.downsamples.append(
                    nn.Conv2d(in_channels=depth[i], out_channels=depth[i + 1], kernel_size=3, stride=1,
                              bias=False))  # (W-2,H-2)
            else:
                self.downsamples.append(nn.Conv2d(in_channels=depth[i], out_channels=depth[i], kernel_size=3, stride=1,
                                                  bias=False))  # (W-2,H-2)
            self.fusions.append(AttenFusion(places=depth[i], n_iter=n_inter))
        self.head = ClassifyHead(in_places=depth[-1], num_class=num_cls)

    def forward(self, x):
        mu_vars_list = []
        fusion_list = []
        x = self.conv1(x)
        for i in range(self.l):

            x = self.blocks[i](x)

            mu_vars, fusion = self.fusions[i](x['padding'])

            if self.training:
                mu_vars_list += mu_vars
                f = dict()
                f['feature'] = []
                for feature in x['no_padding']:
                    f['feature'].append(self.flatten(feature))
                f['fusion'] = self.flatten(fusion)
                fusion_list.append(f)
            x = self.downsamples[i](fusion)

        x = self.head(x)
        return mu_vars_list, fusion_list, x

    def drop_foward(self, x, drop_List=[0.3, 0.6, 0.9]):
        x = self.conv1(x)
        for i in range(self.l):
            x = self.blocks[i].drop_forward(x, drop_List[i])['padding']

            mu_vars, x = self.fusions[i](x)

            x = self.downsamples[i](x)

        x = self.head(x)
        return x

    def explain_foward(self, x):
        ifc_matrix = torch.zeros((self.l,self.n_iter,self.n_iter))
        x = self.conv1(x)
        for i in range(self.l):
            x = self.blocks[i].drop_forward(x,0.0)['padding']
            ifc_matrix[i] += calculate_ifc(x)
            mu_vars, x = self.fusions[i](x)

            x = self.downsamples[i](x)

        x = self.head(x)

        ifc_li = []

        for row in range(1,self.n_iter):
            for col in range(row + 1, self.n_iter):
                ifc_li.append(ifc_matrix[:, row, col].item())

        return x,ifc_matrix,ifc_li
