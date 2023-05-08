# -*- coding: utf-8 -*-
"""
Author: JHM
Date: 2022-05-08
Description: 预测模型的测试
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from CreatData import creatData
from main import GCNLSTM, MyDataset


if __name__ == '__main__':

    # 参数初始化
    batch_size = 100
    num_feature = 10

    # 数据处理
    cd = creatData(time_step=num_feature)
    cd.process()

    # 创建自定义数据集
    adj = torch.tensor(np.load('adj_matrix.npy')).to(torch.float32)
    X = torch.from_numpy(np.load(r'TE_Data_1/Test_X.npy'))
    y = torch.from_numpy(np.load(r'TE_Data_1/Test_y.npy'))

    # 创建数据加载器
    test_set = MyDataset(X, y)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # 加载模型
    model = GCNLSTM(input_dim=num_feature, hidden_dim=64, num_layers=2)
    model.load_state_dict(torch.load('TE_Model.pt'))
    model.eval()  # 设置模型为评估模式

    # 对测试数据进行预测
    with torch.no_grad():
        batch = 1
        for inputs, targets in test_loader:
            test_outputs = model(inputs, adj)
            criterion = nn.MSELoss()
            test_loss = criterion(test_outputs, targets)
            print('Test loss:', test_loss.item())
            batch += 1
