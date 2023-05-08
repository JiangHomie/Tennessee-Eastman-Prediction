# -*- coding: utf-8 -*-
"""
Author: JHM
Date: 2022-05-08
Description: 创建数据集
"""
import numpy as np
import pandas as pd
import os


class creatData():
    """
    整个TE数据集由训练集和测试集构成，TE集中的数据由22次不同的仿真运行数据构成。

    1、【d00.dat至d21.dat为训练集样本，d00_te.dat至d21_te.dat为测试集样本。】
    其中d00.dat和d00_te.dat为正常工况下的样本；
    d00.dat训练样本是在25h运行仿真下获得的，观测数据总数为500；
    d00_te.dat测试样本是在48h运行仿真下获得的，观测数据总数为960。

    2、【d01.dat至d21.dat为带有故障的训练集样本，d01_te.dat至d21_te.dat为带有故障的测试集样本。】
    每个训练集、测试样本代表一种故障。仿真开始时没有故障情况,故障是在仿真时间为1h的时候引入的。3min一个点。
    带有故障的训练集样本是在25h运行仿真下获得的，故障在1h的时候引入，共采集480个观测值，其中前160个观测值为正常数据。
    带有故障的测试集样本是在48h运行仿真下获得的，故障在8h的时候引入，共采集960个观测值，其中前160个观测值为正常数据。
    :return:
    """

    def __init__(self, time_step):
        self.time_step = time_step

    def sliding_window(self, arr, target):
        """
        滑动窗口取样函数
        :param arr: 输入数组
        :param target: 当前样本对应的故障类型
        :return:
        """
        Xs, ys = [], []
        for i in range(arr.shape[0] - self.time_step + 1):
            Xs.append(arr[i:i + self.time_step, :].T)
            ys.append(int(target))
        return Xs, ys

    def process(self):
        """
        遍历数据文件夹，制作dataset
        :return:
        """
        if {'Test_X.npy', 'Test_y.npy', 'Train_X.npy', 'Train_y.npy'}.issubset(set(os.listdir(r'TE_Data_1'))):
            # 判断 set1 是否为 set2 的子集
            return

        for csv_dir in [r'TE_Data_1/Train', r'TE_Data_1/Test']:
            X_all, y_all = [], []
            for csv_name in os.listdir(csv_dir):
                print(csv_name)
                data = pd.read_csv(os.path.join(csv_dir, csv_name)).values
                Xs_1, ys_1 = self.sliding_window(data[:160, :], 0)                      # 引入故障前
                Xs_2, ys_2 = self.sliding_window(data[160:, :], int(csv_name[1:3]))     # 引入故障后
                X_all += (Xs_1 + Xs_2)
                y_all += (ys_1 + ys_2)

            save_path = r'TE_Data_1/%s_X.npy' % csv_dir.split('/')[1]
            if not os.path.exists(save_path):
                np.save(save_path, np.array(X_all).astype(np.float32))

            save_path = r'TE_Data_1/%s_y.npy' % csv_dir.split('/')[1]
            if not os.path.exists(save_path):
                np.save(save_path, np.array(y_all).astype(np.float32))


if __name__ == '__main__':

    CD = creatData(time_step=10)
    CD.process()

    X = np.load(r'TE_Data_1/Train_X.npy')
    y = np.load(r'TE_Data_1/Train_y.npy')
    X_te = np.load(r'TE_Data_1/Test_X.npy')
    y_te = np.load(r'TE_Data_1/Test_y.npy')
    print('Train_X', X.shape, 'Train_y', y.shape, 'Test_X', X_te.shape, 'Test_y', y_te.shape)
