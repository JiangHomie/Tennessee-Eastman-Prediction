# -*- coding: utf-8 -*-
"""
Author: JHM
Date: 2023-06-02
Description: 预测模型的测试
"""
import torch
from torch.utils.data import DataLoader
import numpy as np
from Proj_1.CreatData import creatData
from Proj_1.main import GCNLSTM, MyDataset


if __name__ == '__main__':

    # 参数初始化
    batch_size = 1
    num_feature = 10    # 节点的输入特征维度
    num_hidden = 64     # LSTM隐藏层维度
    num_layers = 2      # GCN的层数
    num_fault = 22      # 输出类别数

    # 数据处理
    cd = creatData(time_step=num_feature)
    cd.process()

    # 创建Test数据集
    adj = torch.tensor(np.load('adj_matrix.npy')).to(torch.float32)
    X = torch.from_numpy(np.load(r'Test_X.npy'))
    y = torch.from_numpy(np.load(r'Test_y.npy'))

    # 创建数据加载器
    test_set = MyDataset(X, y)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # 加载模型
    model = GCNLSTM(input_dim=num_feature, hidden_dim=num_hidden, output_dim=num_fault, num_layers=num_layers)
    model.load_state_dict(torch.load('TE_Model.pt'))
    model.eval()  # 设置模型为评估模式

    # 对测试数据进行预测
    num_1 = 0  # 正确预测类别的数量
    num_2 = 0  # 正确预测有无故障的数量
    with torch.no_grad():
        for inputs, targets in test_loader:
            test_outputs = model(inputs, adj)
            # 将输出转换为one-hot编码
            predicted_labels = torch.argmax(test_outputs, dim=1)
            one_hot_labels = torch.zeros_like(test_outputs)
            one_hot_labels.scatter_(1, predicted_labels.unsqueeze(1), 1)
            fault_pred = torch.argmax(one_hot_labels).item()
            fault_true = torch.argmax(targets).item()
            # 记录预测结果
            if fault_pred != fault_true:
                num_1 += 1
            if (fault_pred * fault_true != 0) or (fault_pred + fault_true == 0):
                num_2 += 1

    print('ACC: {:.4f}'.format(num_1 / len(test_loader)))
    print('FDR: {:.4f}'.format(num_2 / len(test_loader)))

