# -*- coding: utf-8 -*-
"""
Author: JHM
Date: 2023-06-02
Description: 预测模型的测试
"""
import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from Proj_2.CreatData import creatData
from Proj_2.main import GCNLSTM, MyDataset


if __name__ == '__main__':

    # 参数初始化
    batch_size = 1
    num_feature = 10    # 节点的输入特征维度
    num_hidden = 64     # LSTM隐藏层维度
    num_layers = 2      # GCN的层数
    out_dim = 1         # 输出维数

    # 数据处理
    cd = creatData(time_step=num_feature)
    cd.process()

    # 创建Test数据集
    adj = torch.tensor(np.load('adj_matrix.npy')).to(torch.float32)
    for item in os.listdir(r'Test'):
        X = torch.from_numpy(np.load(r'Test/%s/X.npy' % item))
        y = torch.from_numpy(np.load(r'Test/%s/y.npy' % item))

        # 创建数据加载器
        test_set = MyDataset(X, y)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

        # 加载模型
        model = GCNLSTM(input_dim=num_feature, hidden_dim=num_hidden, output_dim=out_dim, num_layers=num_layers)
        model.load_state_dict(torch.load('TE_Model.pt'))
        model.eval()  # 设置模型为评估模式

        # 对测试数据进行预测
        pred_arr, true_arr = np.zeros((0,)), np.zeros((0,))
        with torch.no_grad():
            for inputs, targets in test_loader:
                test_outputs = model(inputs, adj)
                # criterion = nn.MSELoss()
                # test_loss = criterion(test_outputs, targets)
                pred_arr = np.append(pred_arr, test_outputs.item())
                true_arr = np.append(true_arr, targets.item())

        print(item)
        MSE = float(np.mean((pred_arr - true_arr) ** 2))
        MAE = float(np.mean(np.abs(pred_arr - true_arr)))
        RMSE = float(np.sqrt(np.mean((pred_arr - true_arr) ** 2)))
        print('MSE: {:.4f}, MAE: {:.4f}, RMSE: {:.4f}'.format(MSE, MAE, RMSE))
