#!/usr/bin/env python
# !-*-coding:utf-8 -*-
# !@Time    :2023/11/8 12:33
# !@Author  : murInj
# !@Filer    : .py
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10


class CIFAR10Loader:
    def __init__(self, root='./data', transform=None, batch_size=16):

        self.train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
        self.test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

        # 一次性加载BATCH_SIZE个打乱顺序的数据
        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=True)
