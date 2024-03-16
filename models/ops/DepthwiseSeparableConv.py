import os.path

import torch
from torch import nn
import math

from utils.io_utils import create_directory_if_not_exists, save_json_file


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                                   groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class DepthwiseSeparableGroupConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n, g1, stride=1, padding=0):
        super(DepthwiseSeparableGroupConv, self).__init__()
        self.n = n
        self.g1 = g1 + self.n % g1
        self.g2 = math.gcd(self.g1, out_channels)
        self.middle_channel = out_channels // self.g2

        c1 = in_channels * self.n // self.g1
        self.depth_convs = [nn.Conv2d(c1, c1, kernel_size=kernel_size, stride=stride, padding=padding,
                                      groups=c1).cuda() for i in range(self.g1)]

        c2 = in_channels * self.n // self.g2
        self.point_convs = [nn.Conv2d(c2, self.middle_channel, kernel_size=1).cuda() for i in range(self.g2)]

    def forward(self, x):
        x = torch.concat(x, dim=1)
        x = torch.chunk(x, self.g1, dim=1)
        x = [self.depth_convs[i](x[i]) for i in range(len(x))]
        x = torch.concat(x, dim=1)
        x = torch.chunk(x, self.g2, dim=1)
        x = [self.point_convs[i](x[i]) for i in range(len(x))]
        x = torch.concat(x, dim=1)
        return x

    def save_onnx_model(self, input_sample, dir_path):
        self.eval()
        create_directory_if_not_exists(dir_path)
        infer_map = {}

        input_sample = torch.concat(input_sample, dim=1)
        input_sample = torch.chunk(input_sample, self.g1, dim=1)

        depthWise = []
        for i in range(len(input_sample)):
            depth_path = os.path.join(dir_path, f'DW{i}.onnx')
            torch.onnx.export(self.depth_convs[i], input_sample[i], depth_path, export_params=True, opset_version=11,
                              do_constant_folding=True, input_names=['input'], output_names=['output'],
                              dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
            depthWise.append(self.depth_convs[i](input_sample[i]))
        input_sample = torch.concat(depthWise, dim=1)
        input_sample= torch.chunk(input_sample, self.g2, dim=1)
        pointWise = []
        for i in range(len(input_sample)):
            point_path = os.path.join(dir_path, f'PW{i}.onnx')
            torch.onnx.export(self.point_convs[i], input_sample[i], point_path, export_params=True, opset_version=11,
                              do_constant_folding=True, input_names=['input'], output_names=['output'],
                              dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
            pointWise.append(self.point_convs[i](input_sample[i]))
        infer_map_path = os.path.join(dir_path, "infer_map.json")
        save_json_file(infer_map_path, infer_map)

