#coding:utf8

"""
tricks:
1.torch-optimizer:实现了最新的一些优化器.
2.numba:import numba as nb,纯python或numpy加速,加@nb.njit或@nb.jit(nopython=True)
3.swifter:df.apply()→·df.swifter.apply()，加速pandas
4.cupy:1000万以上数据更快
5.modin:import modin.pandas as mdpd,用mdpd代替pd即可，加速pandas,加载数据和查询数据更快,统计方法pandas更快
"""
import os
import sys
import argparse
import time
import random, wandb
from tqdm import tqdm
import numpy as np
# import numba as nb
# import pandas as pd
import torch
from torch.utils.data import DataLoader,SubsetRandomSampler
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
# from torchsummary import summary
from models.hycnn1 import HyCnn
from utils.util import EarlyStopping,Logger
from utils.hydataset import HyDataSet
import utils.training as training


def train(start_epoch, num_epochs):

    for epoch in tqdm(range(start_epoch, num_epochs)):
        torch.cuda.empty_cache()
        print("Epoch: %d" % epoch)
        train_losses = []
        model.train()
        past = time.time()
        for batch_idx, (inputs, targets) in enumerate(train_data_loader):

            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

            # if (batch_idx + 1) % num == 0:
            # 	print(batch_idx + 1, len(dataloader), 'Loss: %.3f' % (train_loss / num))
            # 	train_loss = 0
        now = time.time()
        train_loss = np.mean(np.array(train_losses))
        print(epoch, "loss:%.3f,time:%.2fs" % (train_loss, now - past))
        writer.add_scalar("train_loss", train_loss, epoch)
        wandb.log({"train_loss": train_loss}, step=epoch)
        train_loss = 0
        # checkpoint = {
        #     "model_state_dict": models.module.state_dict(),
        #     "opt_state_dict": optimizer.state_dict(),
        #     "epoch": epoch,
        # }

        scheduler.step()
        # the end of one epoch
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(validation_data_loader):
                inputs, targets = inputs.to(device), targets.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_losses.append(loss.item())
        val_loss = np.mean(np.array(val_losses))
        wandb.log({"val_loss": val_loss}, step=epoch)

        #####some testing#####

        #####some logging#####

        path_to_checkpoint = "snapshot/model_weights.pt"
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            path_to_checkpoint,
        )
        wandb.save("mymodel.h5")


if __name__ == "__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    # torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # 准备数据
    testRation = 0.5
    batchSize = 32
    snapshotPath = "./snapshot/snapshot-loss-1-acc-1.pth"
    dataPath = "data/Indian_pines_corrected.mat"
    labelPath = "data/Indian_pines_gt.mat"
    allDataset = HyDataSet(dataPath, labelPath, name="indian_pines", windowSize=5)  # 定义的数据集
    img_inds = np.arange(len(allDataset))
    np.random.shuffle(img_inds)
    train_inds = img_inds[:int(testRation * len(img_inds))]
    val_inds = img_inds[int(testRation * len(img_inds)):]

    train_data_loader = DataLoader(
        allDataset,
        batch_size=batchSize,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        drop_last=True,
        sampler=SubsetRandomSampler(train_inds)
    )
    validation_data_loader = DataLoader(
        allDataset,
        batch_size=batchSize,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        drop_last=True,
        sampler=SubsetRandomSampler(val_inds)
    )

    # 自定义的model
    model = HyCnn()
    model.to(device)
    isLoadWeights = False
    if(len(snapshotPath)>0 and isLoadWeights):
        model.loadWeights(snapshotPath, device)
    else:
        model.initialize_weights()
    

    # 并行运算，如果需要的话
    # model = nn.DataParallel(model, device_ids=[0]).to(device)
    # summary(model, input_size=(channels, H, W))
    # hl.build_graph(model, torch.zeros([1, 2, 3]))

    # loss function， 比如交叉熵
    criterion = nn.CrossEntropyLoss()
    criterion.to(device)
    metrics = {
        'acc': training.accuracy
    }

    # optimizer，比如Adam
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-06)
    # 调整学习率
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)

    # 训练
    num_epochs = 100

    start_epoch = 0
    writer = SummaryWriter("runs/models")
    writer.iteration, writer.interval = 0, 10

    isTrainFirst = True
    try:
        for epoch in range(num_epochs):
            print('\nEpoch {}/{}'.format(epoch + 1, num_epochs))
            print('-' * 10)

            model.train()
            training.pass_epoch(model, criterion, optimizer, train_data_loader, Logger, writer=writer, device=device, batch_metrics=metrics)

            model.eval()
            valLoss,valMetrics = training.pass_epoch(model, criterion, optimizer, validation_data_loader, Logger, writer=writer, device=device, batch_metrics=metrics)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    finally:
        print("finally")

    savePath = './snapshot/snapshot-loss-{}-acc-{}.pth'.format('4', '4')
    torch.save(model.state_dict(), savePath)
    print("model saved!!!")

    writer.close()

    # train(start_epoch, num_epochs)

