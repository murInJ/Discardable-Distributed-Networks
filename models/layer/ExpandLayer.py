import os
import random

import torch
from torch import nn
import copy
from utils.io_utils import save_json_file, create_directory_if_not_exists


class ExpandLayer(nn.Module):
    def __init__(self, model, dropout_prob=0.5, groups=3):
        super(ExpandLayer, self).__init__()
        self.groups = groups
        self.dropout_prob = dropout_prob
        self.blocks = nn.ModuleList()
        for i in range(groups):
            self.blocks.append(copy.deepcopy(model))

    def forward(self, x):
        features = torch.chunk(x, self.groups, dim=1)
        feature = [self.blocks[0](features[0])]
        cnt = 0
        for i in range(1, len(self.blocks)):
            if torch.rand(1).item() >= self.dropout_prob:
                feature.append(self.blocks[i](features[i]))
            else:
                cnt += 1

        random.shuffle(feature)
        for i in range(cnt):
            feature.append(torch.zeros_like(feature[0]))

        return feature

    def drop_forward(self, x, drop=0.3):
        features = torch.chunk(x, self.groups, dim=1)
        feature = [self.blocks[0](features[0])]
        cnt = 0
        for i in range(1, len(self.blocks)):
            if torch.rand(1).item() >= drop:
                feature.append(self.blocks[i](features[i]))
            else:
                cnt += 1

        random.shuffle(feature)
        for i in range(cnt):
            feature.append(torch.zeros_like(feature[0]))

        return feature

    def save_onnx_model(self, input_sample, dir_path):
        create_directory_if_not_exists(dir_path)
        infer_map = {}
        features = torch.chunk(input_sample, self.groups, dim=1)
        # save block
        feature = []
        self.blocks.eval()
        for i in range(len(self.blocks)):
            block_path = os.path.join(dir_path, f"block{i}.onnx")
            infer_map["input"] = [block_path]
            infer_map[block_path] = ["output"]
            torch.onnx.export(self.blocks[i], features[i], block_path, export_params=True, opset_version=11,
                              do_constant_folding=True, input_names=['input'], output_names=['output'],
                              dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
            feature.append(self.blocks[i](features[i]))


        infer_map_path = os.path.join(dir_path, "infer_map.json")
        save_json_file(infer_map_path, infer_map)
