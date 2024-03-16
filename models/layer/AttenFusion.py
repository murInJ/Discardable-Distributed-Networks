import math
import os
import random

from torch import nn
import torch
from models.attention.ChannelAttention import ChannelAttention
from models.loss.infoLoss import InfoMin_loss
from models.ops.DepthwiseSeparableConv import DepthwiseSeparableGroupConv
from utils.io_utils import create_directory_if_not_exists, save_json_file


class AttenFusion(nn.Module):
    def __init__(self, split_places, out_places, n_iter=3):
        super(AttenFusion, self).__init__()
        self.infoMin_loss = 0.0
        self.conv1 = DepthwiseSeparableGroupConv(in_channels=split_places, out_channels=out_places, n=n_iter, g1=3,
                                                 kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.n_iter = n_iter

        self.attenBlock = nn.Sequential(ChannelAttention(out_places), nn.SiLU(), nn.BatchNorm2d(out_places))

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.places = split_places

    def forward(self, feature):
        if self.training:
            self.infoMin_loss = 0.0
            if len(feature) != 1:
                for feat in feature:
                    flat = self.flatten(feat)
                    mu = torch.mean(flat, 0)
                    log_var = torch.log(torch.var(flat, 0))
                    l = InfoMin_loss(mu, log_var)
                    if l != math.inf:
                        self.infoMin_loss += l
        x = feature
        # attention
        x = self.conv1(x)
        x = self.attenBlock(x)
        assert self.infoMin_loss != math.inf
        return x

    def save_onnx_model(self, input_sample, dir_path):
        self.eval()
        create_directory_if_not_exists(dir_path)
        infer_map = {}

        # save conv1
        streamMerge_path = os.path.join(dir_path, f"streamMerge")
        infer_map["input"] = [streamMerge_path]
        self.conv1.save_onnx_model(input_sample, streamMerge_path)
        input_sample = self.conv1(input_sample)

        # save att
        att_path = os.path.join(dir_path, f"attenBlock")
        torch.onnx.export(self.attenBlock, input_sample, att_path, export_params=True, opset_version=11,
                          do_constant_folding=True, input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
        infer_map[streamMerge_path] = [att_path]
        infer_map[att_path] = ['output']

        infer_map_path = os.path.join(dir_path, "infer_map.json")
        save_json_file(infer_map_path, infer_map)
