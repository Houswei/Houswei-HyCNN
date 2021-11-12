#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
tricks:
1.torch-optimizer:实现了最新的一些优化器.
2.numba:import numba as nb,纯python或numpy加速,加@nb.njit或@nb.jit(nopython=True)
3.swifter:df.apply()→·df.swifter.apply()，加速pandas
4.cupy:1000万以上数据更快
5.modin:import modin.pandas as mdpd,用mdpd代替pd即可，加速pandas,加载数据和查询数据更快,统计方法pandas更快
'''

import os
import random
import numpy as np
# import numba as nb
# import pandas as pd
import torch
import scipy.io as scio
from torch.utils.data import Dataset, TensorDataset, DataLoader
from sklearn.decomposition import PCA


class HyDataSet(Dataset):

    def __init__(self, dataPath, labelPath, name, windowSize):
        self.windowSize = windowSize
        self.padSize = int((windowSize-1)/2)
        self.outputUnits = 9 if (name == 'PU' or name == 'PC') else 16
        self.inputData, self.inputLabel = self.load_data(dataPath, labelPath, name)
        # self.inputData,self.pca = self.applyPCA(self.inputData, numComponents=30)
        self.patchData,self.patchLabel = self.createImageCubes(self.inputData, self.inputLabel)
        self.patchData,self.patchLabel = self.transposeData(self.patchData, self.patchLabel)
        # self.patchLabel = self.toCategorical(self.patchLabel)
        self.dataLen = self.patchData.shape[0]

    def applyPCA(self, X, numComponents=75):
        newX = np.reshape(X, (-1, X.shape[2]))
        pca = PCA(n_components=numComponents, whiten=True)
        newX = pca.fit_transform(newX)
        newX = np.reshape(newX, (X.shape[0], X.shape[1], numComponents))
        return newX, pca

    def transposeData(self, inputData, inputLabel):
        dataShape = inputData.shape
        inputData = inputData.reshape(dataShape[0], 1, dataShape[1], dataShape[2], dataShape[3])
        inputData = np.transpose(inputData, (0,1,4,2,3))
        # inputData = np.transpose(inputData, (0, 3, 1, 2))
        inputData = inputData.astype('float32')
        inputLabel = inputLabel.astype('int64')
        return inputData, inputLabel

    def toCategorical(self, inputLabel):
        inputLabel = np.eye(self.outputUnits)[inputLabel]
        return inputLabel

    def __getitem__(self, idx):
        assert idx < self.dataLen
        rdata = self.patchData[idx]
        rlabel = self.patchLabel[idx]
        return rdata, rlabel

    def __len__(self):
        return self.dataLen

    def padWithZeros(X, padSize):
        newX = np.zeros((X.shape[0] + 2 * padSize, X.shape[1] + 2 * padSize, X.shape[2]))
        x_offset = padSize
        y_offset = padSize
        newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
        return newX

    def padData(self, inputD):
        inputD = np.pad(inputD, ((self.padSize,self.padSize), (self.padSize,self.padSize), (0,0)), 'constant')
        # self.inputLabel = np.pad(inputLabel, ((self.padSize, self.padSize),(self.padSize,self.padSize)), 'constant')
        return inputD

    def createImageCubes(self, data, label, removeZeroLabels=True):
        assert data.shape[:2]==label.shape

        margin = self.padSize
        zeroPaddedX = self.padData(data)
        # split patches
        patchesData = np.zeros((data.shape[0] * data.shape[1], self.windowSize, self.windowSize, data.shape[2]))
        patchesLabels = np.zeros((label.shape[0] * label.shape[1]))
        patchIndex = 0
        for r in range(margin, zeroPaddedX.shape[0] - margin):
            for c in range(margin, zeroPaddedX.shape[1] - margin):
                patch = zeroPaddedX[r - margin:r + margin + 1, c - margin:c + margin + 1]
                patchesData[patchIndex, :, :, :] = patch
                patchesLabels[patchIndex] = label[r - margin, c - margin]
                patchIndex = patchIndex + 1
        if removeZeroLabels:
            pindex = patchesLabels > 0
            patchesData = patchesData[pindex, :, :, :]
            patchesLabels = patchesLabels[pindex]
            patchesLabels -= 1
        return patchesData, patchesLabels

    def load_data(self, dataPath, labelPath, name):
        print(f'load data {dataPath} {labelPath}...')
        if(name == 'indian_pines'):
            inputData = scio.loadmat(dataPath)['indian_pines_corrected']
            inputLabel = scio.loadmat(labelPath)['indian_pines_gt']
        elif(name == 'salinas'):
            inputData = scio.loadmat(dataPath)['salinas_corrected']
            inputLabel = scio.loadmat(labelPath)['salinas_gt']
        elif(name == 'paviaU'):
            inputData = scio.loadmat(dataPath)['paviaU']
            inputLabel = scio.loadmat(labelPath)['paviaU_gt']
        else:
            raise ValueError('wrong data type name!!!')

        return inputData,inputLabel

"""
示例一：
data_df = pd.read_csv("xxx.csv")
feature_col = [
    "No",
    "year",
    "month",
    "day",
    "hour",
    "DWEP",
    "TEMP",
    "PRES",
    "Iws",
    "Is",
    "Ir",
]
data_df_x = data_df.loc[:127, feature_col]
label_col = ["pm2.5"]
data_df_y = data_df.loc[:127, label_col]

data_numpy_x=data_df_x.values
data_numpy_y=data_df_y.values

X=torch.from_numpy(data_numpy_x)
Y=torch.from_numpy(data_numpy_y)

dataset=TensorDataset(X,Y)
dataloader=DataLoader(dataset=dataset,batch_size=64,shuffle=True,)

利用TensorDataset的时候传入的应该是tensor类型，如果是df需要先转换成numpy.array在转换成tensor，输出的也是tensor，事情其实可以分为以下三步：

1.加载数据，提取出feature和label，并转换成tensor
2. 传入TensorDataset中，实例化TensorDataset为datsset
3. 再将dataset传入到Dataloader中，最后通过enumerate输出我们想要的经过shuffle的bachsize大小的feature和label数据
"""

if __name__ == '__main__':
    dataPath = '../data/Indian_pines_corrected.mat'
    labelPath = '../data/Indian_pines_gt.mat'
    h0 = HyDataSet(dataPath, labelPath, 'indian_pines', windowSize=5)
