import os
import random
from tqdm import tqdm
import time
import numpy as np
# import numba as nb
# import pandas as pd
import torch

def accuracy(logits, y):
    _, preds = torch.max(logits, 1)
    return (preds == y).float().mean()

def pass_epoch(model, criterion, optimizer, dataLoader, Logger, batch_metrics={'acc':accuracy} , scheduler=None, show_running=True, device='cpu', writer=None):
    mode = 'Train' if model.training else 'Valid'
    logger = Logger(mode, length=len(dataLoader), calculate_mean=show_running)
    loss = 0

    metrics = {}

    for i_batch, (x, y) in enumerate(dataLoader):
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        loss_batch = criterion(y_pred, y)

        if model.training:
            optimizer.zero_grad()
            loss_batch.backward()
            optimizer.step()

        metrics_batch = {}
        for metric_name, metric_fn in batch_metrics.items():
            metrics_batch[metric_name] = metric_fn(y_pred, y).cpu()
            metrics[metric_name] = metrics.get(metric_name, 0) + metrics_batch[metric_name]

        if writer is not None and model.training:
            if writer.iteration % writer.interval == 0:
                writer.add_scalars('loss', {mode: loss_batch.cpu()}, writer.iteration)
                for metric_name, metric_batch in metrics_batch.items():
                    writer.add_scalars(metric_name, {mode: metric_batch}, writer.iteration)
            writer.iteration += 1

        loss_batch = loss_batch.detach().cpu()
        loss += loss_batch
        if show_running:
            logger(loss, metrics, i_batch)
        else:
            logger(loss_batch, metrics_batch, i_batch)

    if model.training and scheduler is not None:
        scheduler.step()

    loss = loss / (i_batch + 1)
    metrics = {k: v / (i_batch + 1) for k, v in metrics.items()}

    if writer is not None and not model.training:
        writer.add_scalars('loss', {mode: loss.detach()}, writer.iteration)
        for metric_name, metric in metrics.items():
            writer.add_scalars(metric_name, {mode: metric})

    return loss, metrics