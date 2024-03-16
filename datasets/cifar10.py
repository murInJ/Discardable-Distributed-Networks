#!/usr/bin/env python
# !-*-coding:utf-8 -*-
# !@Time    :2023/11/8 12:33
# !@Author  : murInj
# !@Filer    : .py
import cv2
import imageio as imageio
import torchvision
from torch.utils.data import DataLoader
import numpy as np
import pickle
import os

from utils.io_utils import create_directory_if_not_exists


def unpickle(file):
    fo = open(file, 'rb')
    dict = pickle.load(fo, encoding='latin1')
    fo.close()
    return dict



class CIFAR10Loader:
    def __init__(self, root='./data', transform=None, batch_size=16):
        self.transform = transform
        self.root = root
        self.batch_size = batch_size
        self.train_set = torchvision.datasets.CIFAR10(root=root, train=True, download=True, transform=transform)
        self.test_set = torchvision.datasets.CIFAR10(root=root, train=False, download=True, transform=transform)

        # 一次性加载BATCH_SIZE个打乱顺序的数据
        self.train_loader = DataLoader(self.train_set, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_set, batch_size=batch_size, shuffle=True)

    def generate_img(self,resize=(32,32)):
        testXtr = unpickle(os.path.join(self.root,"cifar-10-batches-py/test_batch"))
        for i in range(0, 10000):
            img = np.reshape(testXtr['data'][i], (3, 32, 32))
            img = img.transpose(1, 2, 0)
            img = cv2.resize(img,resize)
            picPath = os.path.join(self.root,"cifar-10-batches-py/",f'val/{testXtr["labels"][i]}')
            picName = str(i) + '.png'
            create_directory_if_not_exists(picPath)
            imageio.imwrite(os.path.join(picPath,picName), img)
            print("write",picName)
