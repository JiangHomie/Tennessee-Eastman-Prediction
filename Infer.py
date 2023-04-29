# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 12:34:56 2022
Author: JHM
Date: 2022-04-30
Description: 预测模型的测试
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from Data_Process import DataProcess
from main import ConvLSTMModel, MyDataset


if __name__ == '__main__':

    # 参数初始化
    batch_size = 100
    time_step = 10
    chanel = 1
    width = 7
    height = 7
    n_epoch = 1

    # 数据处理
    processor = DataProcess(time_step, chanel, width, height)
    processor.dat_to_csv()
    processor.creat_data()

    # 创建自定义数据集
    Test_X = torch.from_numpy(np.load(r'TE_Data/Test_X.npy'))
    Test_y = torch.from_numpy(np.load(r'TE_Data/Test_y.npy'))
    test_set = MyDataset(Test_X, Test_y)

    # 创建数据加载器
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # 加载模型
    model = ConvLSTMModel()
    model.load_state_dict(torch.load('TE_Model.pt'))
    model.eval()  # 设置模型为评估模式

    # 对测试数据进行预测
    with torch.no_grad():
        for inputs, targets in test_loader:
            test_outputs = model(inputs)
            # 对模型预测结果进行评估
            criterion = nn.MSELoss()
            test_loss = criterion(test_outputs, targets)
            print('Test loss:', test_loss.item())
