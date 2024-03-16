import os.path

import torch
from torch import nn
from torch.nn import functional as F
from models.block.ResNeXtBlock import ResNeXtBlock
from models.layer.AttenFusion import AttenFusion
from models.layer.ExpandLayer import ExpandLayer
from models.loss.infoLoss import InfoMax_loss
from utils.io_utils import create_directory_if_not_exists, save_json_file


class DiscardableDistributedBlock(nn.Module):
    def __init__(self, in_places, out_places, dropout_prob, groups):
        super(DiscardableDistributedBlock, self).__init__()
        assert in_places % groups == 0
        self.groups = groups
        self.split_places = in_places // groups
        self.expandLayer = ExpandLayer(ResNeXtBlock(in_places=self.split_places, places=self.split_places),dropout_prob=dropout_prob,
                                       groups=groups)
        self.fusion = AttenFusion(split_places=self.split_places, out_places=out_places)
        self.infoConv = nn.Conv2d(out_places, self.split_places, kernel_size=1)
        self.flatten = nn.Flatten()

        self.infoMin_loss = 0.0
        self.infoMax_loss = 0.0

    def forward(self, x):
        self.infoMin_loss = 0.0
        self.infoMax_loss = 0.0

        features = self.expandLayer(x)
        x = self.fusion(features)
        if self.training:
            self.infoMin_loss += self.fusion.infoMin_loss
            for feature in features:
                B1, C1, W1, H1 = feature.size()
                feature2 = F.interpolate(x, size=(W1, H1), mode='bilinear', align_corners=False)
                feature2 = self.infoConv(feature2)
                self.infoMax_loss += InfoMax_loss(self.flatten(feature), self.flatten(feature2))

        return x

    def drop_forward(self, x, drop_prob):
        x = self.expandLayer.drop_forward(x, drop_prob)
        x = self.fusion(x)

        return x

    def save_onnx_model(self, input_sample, dir_path):
        create_directory_if_not_exists(dir_path)
        infer_map = {}

        # save expandLayer
        expandLayer_path = os.path.join(dir_path, "expandLayer")
        infer_map['input'] = [expandLayer_path]
        self.expandLayer.eval()
        self.expandLayer.save_onnx_model(input_sample,expandLayer_path)

        # save fusion
        fusion_path = os.path.join(dir_path,"fusion")
        input_sample = self.expandLayer(input_sample)
        self.fusion.save_onnx_model(input_sample,fusion_path)
        infer_map[expandLayer_path] = [fusion_path]
        infer_map[fusion_path] = ["output"]
        infer_map_path = os.path.join(dir_path, "infer_map.json")
        save_json_file(infer_map_path, infer_map)
