#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
tricks:
1.torch-optimizer:实现了最新的一些优化器.
2.numba:import numba as nb,纯python或numpy加速,加@nb.njit或@nb.jit(nopython=True)
3.swifter:df.apply()→·df.swifter.apply()，加速pandas
4.cupy:1000万以上数据更快
5.modin:import modin.pandas as mdpd,用mdpd代替pd即可，加速pandas,加载数据和查询数据更快,统计方法pandas更快
"""
import torch
from torch import nn
import torch.nn.functional as F


class HyCnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(1, 20, kernel_size=(3, 3, 3), padding=(1, 0, 0))
        self.relu1 = nn.ReLU()
        self.pool1 = nn.Conv3d(20, 20, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        # Then, a duplicate of the first and second layers is created with
        # 35 hidden neurons per layer.
        self.conv2 = nn.Conv3d(20, 35, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 0, 0))
        self.relu2 = nn.ReLU()
        self.pool2 = nn.Conv3d(35, 35, kernel_size=(3, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        # Finally, the 1D spatial dimension is progressively reduced
        # thanks to the use of two Conv layers, 35 neurons each,
        # with respective kernel sizes of (1,1,3) and (1,1,2) and strides
        # respectively equal to (1,1,1) and (1,1,2)
        self.conv3 = nn.Conv3d(35, 35, kernel_size=(3, 1, 1), stride=(1, 1, 1), padding=(1, 0, 0))
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv3d(35, 35, kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(1, 0, 0))
        self.relu4 = nn.ReLU()
        self.flat1 = nn.Flatten()
        self.lin1 = nn.Linear(12544, 512)
        self.relu5 = nn.ReLU()
        self.drop1 = nn.Dropout(0.4)
        self.lin2 = nn.Linear(910, 16)
        self.relu6 = nn.ReLU()
        self.drop2 = nn.Dropout(0.4)
        self.out1 = nn.Linear(128, 16)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.flat1(x)
        # x = self.drop1(x)
        x = self.lin2(x)
        # x = self.relu6(x)
        # x = self.drop2(x)
        # x = self.out1(x)
        return x

    def loadWeights(self, sPath, device):
        self.load_state_dict(torch.load(sPath, map_location=device))
        return True

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight.data, 0, 0.01)
                m.bias.data.zero_()
