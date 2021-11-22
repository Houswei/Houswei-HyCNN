#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 30 19:59:45 2020

@author: krishna

"""

import torch.nn as nn
import torch


class HyCnn(nn.Module):
    def __init__(self, input_dim=9, num_classes=16):
        super(HyCnn, self).__init__()
        self.tdnn1 = nn.Conv1d(input_dim, 128, kernel_size=5, padding=2)
        self.tdnn2 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.tdnn3 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        self.tdnn4 = nn.Conv1d(128, 128, kernel_size=1, padding=0)
        self.tdnn5 = nn.Conv1d(128, 128, kernel_size=1, padding=0)
        self.bn5 = nn.BatchNorm1d(128)
        # self.tdnn1 = TDNN(input_dim=input_dim, output_dim=512, context_size=5, dilation=1, dropout_p=0.5)
        # self.tdnn2 = TDNN(input_dim=512, output_dim=512, context_size=3, dilation=1, dropout_p=0.5)
        # self.tdnn3 = TDNN(input_dim=512, output_dim=512, context_size=2, dilation=2, dropout_p=0.5)
        # self.tdnn4 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=1, dropout_p=0.5)
        # self.tdnn5 = TDNN(input_dim=512, output_dim=512, context_size=1, dilation=3, dropout_p=0.5)
        #### Frame levelPooling
        self.segment6 = nn.Linear(128*200, 512)
        self.segment7 = nn.Linear(512, 512)
        self.output = nn.Linear(512, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        shape = inputs.shape
        inputs = inputs.view(-1, shape[1]*shape[2], shape[3]*shape[4])
        inputs = inputs.permute(0, 2, 1)
        tdnn1_out = self.tdnn1(inputs)
        tdnn2_out = self.tdnn2(tdnn1_out)
        tdnn3_out = self.tdnn3(tdnn2_out)
        tdnn4_out = self.tdnn4(tdnn3_out)
        tdnn5_out = self.tdnn5(tdnn4_out)
        tdnn5_out = self.bn5(tdnn5_out)
        tdnn5_out = tdnn5_out.view(-1,tdnn5_out.shape[1]*tdnn5_out.shape[2])
        ### Stat Pool

        # mean = torch.mean(tdnn5_out, 1)
        # std = torch.var(tdnn5_out, 1)
        # stat_pooling = torch.cat((mean, std), 1)
        segment6_out = self.segment6(tdnn5_out)
        x_vec = self.segment7(segment6_out)
        predictions = self.output(x_vec)
        return predictions

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()