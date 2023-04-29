# -*- coding: utf-8 -*-
"""
Created on Sat Apr 30 12:34:56 2022
Author: JHM
Date: 2022-04-30
Description: 预测TE的21种故障
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from Model_ConvLSTM import ConvLSTM
from Data_Process import DataProcess


class MyDataset(Dataset):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input = torch.tensor(self.inputs[idx])
        target = torch.tensor(self.targets[idx])
        return input, target


class ConvLSTMModel(nn.Module):
    def __init__(self):
        super(ConvLSTMModel, self).__init__()
        self.conv_lstm = ConvLSTM(input_dim=1,
                                  hidden_dim=[64, 32],
                                  kernel_size=(3, 3),
                                  num_layers=2,
                                  batch_first=True,
                                  bias=True,
                                  return_all_layers=False)
        self.linear = nn.Linear(32, 1)

    def forward(self, x):
        output, _ = self.conv_lstm(x)
        output = output[-1][0]  # 只保留最后一个时间步的输出
        output = self.linear(output.view(-1, 32))
        return output


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
    Train_X = torch.from_numpy(np.load(r'TE_Data/Train_X.npy'))
    Train_y = torch.from_numpy(np.load(r'TE_Data/Train_y.npy'))
    train_set = MyDataset(Train_X, Train_y)

    # 创建数据加载器
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # 实例化模型
    model = ConvLSTMModel()

    # 编译和训练模型
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    for epoch in range(n_epoch):
        batch = 1
        for input, target in train_loader:
            output = model(input)
            loss = criterion(output, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print('Epoch [{}/{}], Batch [{}], Loss: {:.4f}'.format(epoch+1, n_epoch, batch, loss.item()))
            batch += 1

    # 保存模型
    torch.save(model.state_dict(), 'TE_Model.pt')
