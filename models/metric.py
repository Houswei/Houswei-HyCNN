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
# #自定义损失函数
# class CustomLoss(nn.Module):
#
#     def __init__(self):
#         super(CustomLoss, self).__init__()
#
#     def forward(self, x, y):
#         loss = torch.mean((x - y) ** 2)
#         return loss
