import math
import os.path

import torch.onnx
from torch import nn
from models.block.DiscardableDistributedBlock import DiscardableDistributedBlock
from models.block.ResNeXtBlock import ResNeXtBlock
from models.layer.AttenFusion import AttenFusion
from models.head.ClassifyHead import ClassifyHead
from models.layer.ExpandLayer import ExpandLayer
from models.loss.infoLoss import InfoMax_loss
from models.ops.DepthwiseSeparableConv import DepthwiseSeparableConv
import torch.nn.functional as F

from utils.io_utils import create_directory_if_not_exists, save_json_file


class DDN(nn.Module):
    def __init__(self, in_places, depth=[64*3, 128*3, 256*3], dropout_probs=[0.3, 0.4, 0.5], groups=3,
                 num_cls=10):
        super(DDN, self).__init__()
        self.l = len(depth)
        self.n_iter = groups
        self.conv1 = DepthwiseSeparableConv(in_channels=in_places, out_channels=depth[0], kernel_size=1,
                                            stride=1)  # (W,H)
        self.DDBlock = nn.ModuleList()

        for i in range(len(depth)):
            self.DDBlock.append(
                DiscardableDistributedBlock(depth[i], out_places=depth[i + 1] if i != len(depth) - 1 else depth[i],
                                            dropout_prob=dropout_probs[i], groups=groups))

        self.head = ClassifyHead(in_places=depth[-1], num_class=num_cls)

        self.infoMin_loss = 0.0
        self.infoMax_loss = 0.0

    def forward(self, x):
        self.infoMin_loss = 0.0
        self.infoMax_loss = 0.0
        x = self.conv1(x)
        for i in range(self.l):
            x = self.DDBlock[i](x)
            if self.training:
                self.infoMin_loss += self.DDBlock[i].infoMin_loss
                self.infoMax_loss += self.DDBlock[i].infoMax_loss
        x = self.head(x)
        assert self.infoMax_loss != math.inf
        assert self.infoMin_loss != math.inf
        return x

    def drop_forward(self, x, drop_List=[0.3, 0.6, 0.9]):
        x = self.conv1(x)
        for i in range(self.l):
            x = self.DDBlock[i].drop_forward(x, drop_prob=drop_List[i])
        x = self.head(x)
        return x

    def save_onnx_model(self, input_sample, dir_path):
        self.eval()
        infer_map = {}
        create_directory_if_not_exists(dir_path)
        # save conv1
        conv1_path = os.path.join(dir_path, "conv1.onnx")
        infer_map["input"] = [conv1_path]
        torch.onnx.export(self.conv1, input_sample, conv1_path, export_params=True, opset_version=11,
                          do_constant_folding=True, input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

        # save DDBlock
        DDBlock_dir1 = conv1_path
        input_sample = self.conv1(input_sample)
        for i in range(self.l):
            DDBlock_dir2 = os.path.join(dir_path, f"DDB{i}")
            create_directory_if_not_exists(DDBlock_dir2)
            self.DDBlock[i].save_onnx_model(input_sample, DDBlock_dir2)
            infer_map[DDBlock_dir1] = [DDBlock_dir2]
            DDBlock_dir1 = DDBlock_dir2
            input_sample = self.DDBlock[i](input_sample)

        # save head
        head_path = os.path.join(dir_path, "head.onnx")
        self.head.eval()
        torch.onnx.export(self.head, input_sample, head_path, export_params=True, opset_version=11,
                          do_constant_folding=True, input_names=['input'], output_names=['output'],
                          dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})
        infer_map[DDBlock_dir1] = [head_path]
        infer_map[head_path] = "output"
        infer_map_path = os.path.join(dir_path, "infer_map.json")
        save_json_file(infer_map_path, infer_map)
