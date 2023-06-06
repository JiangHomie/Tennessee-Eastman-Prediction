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
from Proj_1.CreatData import creatData


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
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GCNLSTM, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.gcn_layers = nn.ModuleList([GCN(input_dim, hidden_dim) if i == 0 else GCN(hidden_dim, hidden_dim) for i in range(num_layers)])
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

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
        """
        x, adj 从输入到输出shape不发生变化
        :param x: torch.Size([batch_size, num_fault, num_feature])
        :param adj: torch.Size([1, num_fault, num_fault])
        :return:
        """
        x = torch.matmul(adj, x)
        x = self.fc(x)
        return x


if __name__ == '__main__':

    # 参数初始化
    batch_size = 100
    num_feature = 20    # 节点的输入特征维度
    num_hidden = 64     # LSTM隐藏层维度
    num_layers = 2      # GCN的层数
    num_fault = 22      # 输出类别数
    num_epoch = 50
    lr = 0.001         # 学习率

    # 数据处理
    cd = creatData(time_step=num_feature)
    cd.process()

    # 创建Train数据集
    adj = torch.tensor(np.load('adj_matrix.npy')).to(torch.float32)
    Xs = torch.from_numpy(np.load(r'Train_X.npy'))
    ys = torch.from_numpy(np.load(r'Train_y.npy'))

    # 创建数据加载器
    train_set = MyDataset(Xs, ys)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

    # 实例化模型，定义损失函数和优化器
    model = GCNLSTM(input_dim=num_feature, hidden_dim=num_hidden, output_dim=num_fault, num_layers=num_layers)
    # 定义类别权重向量
    class_weights = torch.ones(num_fault) * 10
    class_weights[0] = 1
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 创建TensorboardX的summary writer
    writer = SummaryWriter()
    for epoch in range(num_epoch):
        loss = None
        for i, (input, target) in enumerate(train_loader):
            # 前向传播
            output = model(input, adj)
            output_pred = nn.functional.softmax(output, dim=1)
            _, labels = torch.max(target, 1)
            loss = criterion(output_pred, labels)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 每个batch打印一次损失
            print('Epoch [{}/{}], Batch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epoch, i+1, len(train_loader), loss.item()))
            # 记录训练损失到Tensorboard
            writer.add_scalar('Train Loss (batch)', loss, epoch*len(train_loader)+i)

        # 记录训练损失到Tensorboard
        writer.add_scalar('Train Loss (epoch)', loss, epoch + 1)
        if not (epoch + 1) % 10:
            # 保存参数权重的变化
            for name, param in model.named_parameters():
                if 'weight' in name:
                    image = np.expand_dims(param.data.clone().numpy(), axis=0)  # 添加通道维度
                    writer.add_image('%s' % name, image, epoch + 1)

        # writer.flush()  # 实时显示
    writer.close()  # 关闭TensorboardX的summary writer
    torch.save(model.state_dict(), 'TE_Model.pt')  # 保存模型


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
