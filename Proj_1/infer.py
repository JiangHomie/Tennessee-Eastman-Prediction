# -*- coding: utf-8 -*-
"""
Author: JHM
Date: 2023-06-02
Description: 预测模型的测试
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from Proj_1.CreatData import creatData
from Proj_1.main import GCNLSTM, MyDataset
from torch.utils.data import Dataset, DataLoader
import os

class MyTestDataset(Dataset):
    def __init__(self, root_path='./Test'):
        self.root_path = root_path
        self.file_list = os.listdir(self.root_path)
        self.file_list.sort()
        self.input_list = []
        self.target_list = []
        for file_name in self.file_list:
            if 'data' in file_name:
                data = np.load(os.path.join(self.root_path, file_name))
                self.input_list.append(data)
            elif 'label' in file_name:
                label = np.load(os.path.join(self.root_path, file_name))
                self.target_list.append(label)

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        input = torch.tensor(self.input_list[idx])
        target = torch.tensor(self.target_list[idx])
        return input, target


if __name__ == '__main__':

    # 参数初始化1372475723@qq.com
    batch_size = 1
    num_feature = 22    # 节点的输入特征维度
    num_hidden = 64     # LSTM隐藏层维度
    num_layers = 2      # GCN的层数
    num_fault = 22      # 输出类别数

    # 数据处理
    # cd = creatData()
    # cd.process()

    # 创建Test数据集
    adj = torch.tensor(np.load('adj_matrix.npy')).to(torch.float32)
    test_set = MyTestDataset()
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # 加载模型
    multi_model = []
    checkpoints_list = ['TE_Model_01.pt', 'TE_Model_02.pt', 'TE_Model_03.pt', 'TE_Model_04.pt', 'TE_Model_05.pt',
                        'TE_Model_06.pt', 'TE_Model_07.pt', 'TE_Model_08.pt', 'TE_Model_09.pt', 'TE_Model_10.pt',
                        'TE_Model_11.pt', 'TE_Model_12.pt', 'TE_Model_13.pt', 'TE_Model_14.pt', 'TE_Model_15.pt',
                        'TE_Model_16.pt', 'TE_Model_17.pt', 'TE_Model_18.pt', 'TE_Model_19.pt', 'TE_Model_20.pt',
                        'TE_Model_21.pt']
    for ckpt in checkpoints_list:
        model = GCNLSTM(input_dim=num_feature, hidden_dim=num_hidden, output_dim=num_fault, num_layers=num_layers)
        model.load_state_dict(torch.load(os.path.join('./checkpoints', ckpt)))
        model.eval()
        multi_model.append(model)

    # 对测试数据进行预测
    acc_list = []
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            multi_model_vote = multi_model[i]
            output_cls, output_abnorm = multi_model_vote(input.squeeze(), adj)
            output_cls_pred = nn.functional.softmax(output_cls, dim=2)

            # 计算预测准确率
            labels_cls = target.squeeze().long()
            predicted = torch.argmax(output_cls_pred, dim=2)
            print('故障{}: cls prediction'.format(i+1))
            accuracy_cls = torch.eq(predicted, labels_cls).float().mean()
            print('accuracy: ', accuracy_cls)
            acc_list.append(accuracy_cls.item())

            # 计数错误判断故障的个数
            predicted_numpy = predicted.numpy()
            unique_elements, counts = np.unique(predicted_numpy, return_counts=True)
            element_counts = list(zip(unique_elements, counts))
            # 输出每个元素和出现的次数
            for element, count in element_counts:
                print("元素:", element, "，出现次数:", count)

    print('ACC: {:.4f}'.format(np.mean(acc_list)))
    # print('FDR: {:.4f}'.format(num_2 / len(test_loader)))

