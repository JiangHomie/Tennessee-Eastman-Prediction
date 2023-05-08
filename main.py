# -*- coding: utf-8 -*-
"""
Author: JHM
Date: 2022-05-08
Description: 预测TE的21种故障
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from CreatData import creatData


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


class GCNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(GCNLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gcn_layers = nn.ModuleList([GCN(input_dim, hidden_dim) if i == 0 else GCN(hidden_dim, hidden_dim) for i in range(num_layers)])
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x, adj):
        adj = adj.unsqueeze(0)  # 在第0个位置插入一个新的维度
        for i in range(self.num_layers):
            x = self.gcn_layers[i](x, adj)
            x = F.relu(x)
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out


class GCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCN, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x, adj):
        x = torch.matmul(adj, x)
        x = self.fc(x)
        return x


if __name__ == '__main__':

    # 参数初始化
    batch_size = 100
    num_feature = 10
    num_epoch = 2

    # 数据处理
    cd = creatData(time_step=num_feature)
    cd.process()

    # 创建自定义数据集
    adj = torch.tensor(np.load('adj_matrix.npy')).to(torch.float32)
    X = torch.from_numpy(np.load(r'TE_Data_1/Train_X.npy'))
    y = torch.from_numpy(np.load(r'TE_Data_1/Train_y.npy'))

    # 创建数据加载器
    train_set = MyDataset(X, y)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # 实例化模型
    model = GCNLSTM(input_dim=num_feature, hidden_dim=64, num_layers=2)

    # 编译和训练模型
    criterion = nn.MSELoss()  # 损失函数
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 定义优化器
    for epoch in range(num_epoch):
        batch = 1
        for input, target in train_loader:
            optimizer.zero_grad()
            output = model(input, adj)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            print('Epoch [{}/{}], Batch [{}], Loss: {:.4f}'.format(epoch+1, num_epoch, batch, loss.item()))
            batch += 1

    # 保存模型
    torch.save(model.state_dict(), 'TE_Model.pt')
