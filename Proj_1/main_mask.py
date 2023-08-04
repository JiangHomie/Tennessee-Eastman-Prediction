# -*- coding: utf-8 -*-
"""
Author: JHM
Date: 2023-06-02
Description: 预测TE的21种故障
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tensorboardX import SummaryWriter
import torch.nn.functional as F
import numpy as np
import os
from Proj_1.CreatData import creatData

MODEL_NAMES = ('TE_Model_01', 'TE_Model_02', 'TE_Model_03', 'TE_Model_04', 'TE_Model_05',
               'TE_Model_06', 'TE_Model_07', 'TE_Model_08', 'TE_Model_09', 'TE_Model_10',
               'TE_Model_11', 'TE_Model_12', 'TE_Model_13', 'TE_Model_14', 'TE_Model_15',
               'TE_Model_16', 'TE_Model_17', 'TE_Model_18', 'TE_Model_19', 'TE_Model_20',
               'TE_Model_21')

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


class MyTrainDataset(Dataset):
    def __init__(self, root_path='./Train', model_name=None):
        self.root_path = root_path
        self.file_list = os.listdir(self.root_path)
        self.file_list.sort()
        input_list = []
        target_list = []
        index = MODEL_NAMES.index(model_name)
        for file_name in self.file_list[index*2:(index+1)*2]:
            if 'data' in file_name:
                data = np.load(os.path.join(self.root_path, file_name))
                input_list.append(data)
            elif 'label' in file_name:
                label = np.load(os.path.join(self.root_path, file_name))
                target_list.append(label)
        self.inputs = np.concatenate(input_list, axis=0)
        self.targets = np.concatenate(target_list, axis=0)

    def __len__(self):
        return self.inputs.shape[0]

    def __getitem__(self, idx):
        input = torch.tensor(self.inputs[idx])
        target = torch.tensor(self.targets[idx])
        return input, target


class GCNLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GCNLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gcn = GCN(input_dim=input_dim, output_dim=hidden_dim)
        self.fc = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.classifier_cls = nn.Linear(hidden_dim, output_dim)
        self.classifier_abnorm = nn.Linear(hidden_dim, 2)

    def pre_process(self, x):
        mean = torch.mean(x, dim=1, keepdim=True)
        std = torch.std(x, dim=1, keepdim=True)
        normalized_x = (x - mean) / std
        return normalized_x

    def forward(self, x, adj, mask):
        adj = adj.unsqueeze(0)  # 在第0个位置插入一个新的维度
        x = self.pre_process(x)
        x = x * mask
        x_gcn = self.gcn(x, adj)
        x_fc = self.fc(x)
        x_fc = F.relu(x_fc)
        lstm_out, _ = self.lstm(x_gcn)
        out_abnorm = self.classifier_abnorm(x_fc)
        out_cls = self.classifier_cls((x_fc+lstm_out)*out_abnorm[:, :, 1:])
        return out_cls, out_abnorm


class GCN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GCN, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, x, adj):
        """
        x, adj 从输入到输出shape不发生变化
        :param x: torch.Size([batch_size, seq_length, num_feature])
        :param adj: torch.Size([1, num_fault, num_fault])
        :return:
        """
        if self.training:
            x = torch.bmm(x, adj.expand(10, -1, -1))
        else:
            x = torch.bmm(x, adj.expand(1000, -1, -1))
        x = self.fc(x)
        return x


if __name__ == '__main__':

    # 参数初始化
    batch_size = 10
    num_feature = 22    # 节点的输入特征维度
    num_hidden = 64     # LSTM隐藏层维度
    num_layers = 2      # GCN的层数
    num_fault = 22      # 输出类别数
    num_epoch = 5
    lr = 0.0001         # 学习率
    model_name = 'TE_Model_01'  # 模型名称

    # 数据处理
    # cd = creatData()
    # cd.process()

    # 创建Train数据集
    adj = torch.tensor(np.load('adj_matrix.npy')).to(torch.float32)
    train_set = MyTrainDataset(model_name=model_name)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)

    # 实例化模型，定义损失函数和优化器
    model = GCNLSTM(input_dim=num_feature, hidden_dim=num_hidden, output_dim=num_fault, num_layers=num_layers)
    # 定义类别权重向量
    class_weights = torch.ones(num_fault) * 2
    class_weights[0] = 1
    criterion_cls = nn.CrossEntropyLoss(weight=class_weights)
    criterion_abnorm = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 创建TensorboardX的summary writer
    writer = SummaryWriter(comment=model_name, filename_suffix=model_name)
    for epoch in range(num_epoch):
        loss = None
        for i, (input, target) in enumerate(train_loader):
            # 前向传播
            output_cls, output_abnorm = model(input, adj)

            # 故障类别的损失函数
            output_cls_pred = nn.functional.softmax(output_cls, dim=2)
            labels_cls = target.squeeze().long()
            loss_cls = criterion_cls(output_cls_pred.transpose(1, 2), labels_cls)

            # 故障预测的损失函数
            output_abnorm_pred = nn.functional.softmax(output_abnorm, dim=2)
            labels_abnorm = torch.where(labels_cls != 0, torch.tensor(1), labels_cls)
            loss_abnorm = criterion_abnorm(output_cls_pred.transpose(1, 2), labels_abnorm)

            # 计算预测准确率
            predicted = torch.argmax(output_cls_pred, dim=2)
            print('cls prediction')
            print(predicted[0, :])
            accuracy_cls = torch.eq(predicted, labels_cls).float().mean()

            predicted = torch.argmax(output_abnorm_pred, dim=2)
            print('abnorm prediction')
            print(predicted[0, :])
            accuracy_abnorm = torch.eq(predicted, labels_abnorm).float().mean()

            # 反向传播和优化
            optimizer.zero_grad()
            loss_total = loss_cls + loss_abnorm
            # loss_total = loss_cls
            loss_total.backward()
            optimizer.step()
            # 每个batch打印一次损失
            print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}, Accuracy: cls: {:.4f} abnorm: {:.4f}'.format(epoch+1, num_epoch, i+1, len(train_loader),
                                                                                        loss_total.item(), accuracy_cls.item(), accuracy_abnorm.item()))
            # 记录训练损失到Tensorboard
            writer.add_scalar('Train Loss (batch)', loss_total, epoch * len(train_loader)+i)
            writer.add_scalar('Class Accuracy (batch)', accuracy_cls, epoch * len(train_loader) + i)
            writer.add_scalar('Abnorm Accuracy (batch)', accuracy_abnorm, epoch * len(train_loader) + i)

        # 记录训练损失到Tensorboard
        writer.add_scalar('Train Loss (epoch)', loss_total, epoch + 1)
        for name, param in model.named_parameters():
            if name == 'gcn.fc.weight':
                gcn_weight = param.data.clone().numpy()
                agg_gcn_weight = np.sum(gcn_weight, axis=0, keepdims=True)
                agg_gcn_weight = agg_gcn_weight / np.sum(agg_gcn_weight, axis=1, keepdims=True)
                image = np.expand_dims(agg_gcn_weight, axis=0)  # 添加通道维度
                writer.add_image('%s' % name, image, epoch + 1)
            elif 'weight' in name:
                image = np.expand_dims(param.data.clone().numpy(), axis=0)  # 添加通道维度
                writer.add_image('%s' % name, image, epoch + 1)

        # writer.flush()  # 实时显示
    writer.close()  # 关闭TensorboardX的summary writer
    torch.save(model.state_dict(), './checkpoints/{}.pt'.format(model_name))  # 保存模型


"""
训练损失写入到Tensorboard。默认情况下，这些数据会保存在当前目录下的runs文件夹中。
打开终端或命令提示符，并导航到包含代码的目录。
运行以下命令启动 tensorboard --logdir=runs
在浏览器中打开生成的URL，通常是http://localhost:6006/。
在Tensorboard的网页界面上，你将看到训练损失随时间的变化曲线。

如果报错的话大概率是版本问题，我的版本是：
torch==2.0.0
tensorboard==2.11.2
tensorboardX==2.4
"""
