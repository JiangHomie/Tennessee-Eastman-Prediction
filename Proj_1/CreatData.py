# -*- coding: utf-8 -*-
"""
Author: JHM
Date: 2023-06-02
Description: 创建数据集
整个TE数据集由训练集和测试集构成，TE集中的数据由22次不同的仿真运行数据构成。
1、【d00.dat至d21.dat为训练集样本，d00_te.dat至d21_te.dat为测试集样本。】
其中d00.dat和d00_te.dat为正常工况下的样本；
d00.dat训练样本是在25h运行仿真下获得的，观测数据总数为500；
d00_te.dat测试样本是在48h运行仿真下获得的，观测数据总数为960。
2、【d01.dat至d21.dat为带有故障的训练集样本，d01_te.dat至d21_te.dat为带有故障的测试集样本。】
每个训练集、测试样本代表一种故障。仿真开始时没有故障情况,故障是在仿真时间为1h的时候引入的。3min一个点。
带有故障的训练集样本是在25h运行仿真下获得的，故障在1h的时候引入，共采集480个观测值，其中前160个观测值为正常数据。
update: 带有故障的训练集样本是在25h运行仿真下获得的，故障在1h的时候引入，共采集480个观测值，所采集的观测值均为故障数据。
带有故障的测试集样本是在48h运行仿真下获得的，故障在8h的时候引入，共采集960个观测值，其中前160个观测值为正常数据。
"""
import numpy as np
import pandas as pd
import os
import random


class creatData():

    def __init__(self, data_len=600, random_range=300):
        self.data_len = data_len
        self.data_range = (int(data_len/2-random_range/2), int(data_len/2+random_range/2))
        self.sample_num = 1000   # 对于同一种故障的拼接采样次数
        self.num_fault = 22  # 正常情况 + 21种故障情况

    def sliding_window(self, arr, target):
        """
        滑动窗口取样函数
        :param arr: 输入数组
        :param target: 当前样本对应的故障类型
        :return: X 的 shape=(22, time_step)，y 的 shape=(1, num_fault)
        """
        Xs, ys = [], []
        for i in range(arr.shape[0] - self.time_step + 1):
            Xs.append(arr[i:i + self.time_step, :].T)
            y = [0] * self.num_fault
            y[int(target)] = 1
            ys.append(y)
        return Xs, ys

    def norm_abnorm_concatenate(self, data_norm, data_abnorm, label, data_len, data_range):
        """
        根据正常数据和异常数据拼接成训练数据集，将正常样本的后半部分和异常样本的前半部分拼接起来，构成新的数据集
        :param data_norm: 正常样本 (500, 22)
        :param data_abnorm: 带故障的异常样本 (480, 22)
        :param label: 故障样本的故障类型
        :param data_len: 整个采样数据的时序长度
        :param data_range: 采样随机数的范围
        :return:
            data_train_l: (data_len, 22)
            label_l: (data_len, 1)
        """
        split = random.randint(data_range[0], data_range[1])
        data_norm_part = data_norm[-split:, :]
        data_abnorm_part = data_abnorm[:data_len-split, :]
        data_train_l = np.concatenate((data_norm_part, data_abnorm_part), axis=0)
        label_l = np.zeros((data_len, 1))
        label_l[split:, :] = label
        return data_train_l[None, :, :], label_l[None, :, :]

    def process(self):
        """
        遍历数据文件夹，制作dataset
        :return:
        """
        # if {'Test_X.npy', 'Test_y.npy', 'Train_X.npy', 'Train_y.npy'}.issubset(set(os.listdir(r'../Proj_1'))):
        #     判断 set1 是否为 set2 的子集
            # print('数据集构建完毕')
            # return

        # 处理训练集数据
        for csv_dir in [r'../TE_Data_1/Train']:
            csv_file_list = os.listdir(csv_dir)
            csv_file_list.sort()
            data_norm = pd.read_csv(os.path.join(csv_dir, csv_file_list[0])).values     # 获取正常工况数据 (500, 22)
            for l, csv_name in enumerate(csv_file_list):
                if l == 0:  # 跳过正常工况的数据
                    continue
                print(csv_name)
                data_abnorm = pd.read_csv(os.path.join(csv_dir, csv_name)).values       # shape=(n, 22)
                data_train_list = []
                label_list = []
                for i in range(self.sample_num):
                    # 根据正常数据和异常数据拼接成训练数据集
                    data_train_l, label_l = self.norm_abnorm_concatenate(data_norm, data_abnorm, l,
                                                                    self.data_len, self.data_range)
                    data_train_list.append(data_train_l)
                    label_list.append(label_l)
                data_train = np.concatenate(data_train_list)
                label = np.concatenate(label_list)

                save_path = r'./Train/%s_data.npy' % csv_name.split('.')[0]
                if not os.path.exists(save_path):
                    np.save(save_path, np.array(data_train).astype(np.float32))
                save_path = r'./Train/%s_label.npy' % csv_name.split('.')[0]
                if not os.path.exists(save_path):
                    np.save(save_path, np.array(label).astype(np.int8))
            print('Train data finish')

        # 处理测试集数据
        for csv_dir in [r'../TE_Data_1/Test']:
            csv_file_list = os.listdir(csv_dir)
            csv_file_list.sort()
            for l, csv_name in enumerate(csv_file_list):
                print(csv_name)
                if csv_name == 'd00_te.csv':
                    data_test = pd.read_csv(os.path.join(csv_dir, csv_name)).values
                    label = np.zeros((data_test.shape[0], 1))
                else:
                    data_test = pd.read_csv(os.path.join(csv_dir, csv_name)).values
                    label = np.zeros((data_test.shape[0], 1))
                    label[160:, :] = l

                save_path = r'./Test/%s_data.npy' % csv_name.split('.')[0]
                if not os.path.exists(save_path):
                    np.save(save_path, np.array(data_test).astype(np.float32))
                save_path = r'./Test/%s_label.npy' % csv_name.split('.')[0]
                if not os.path.exists(save_path):
                    np.save(save_path, np.array(label).astype(np.int8))
            print('Test data finish')

if __name__ == '__main__':

    CD = creatData()
    CD.process()

    print(np.load(r'./Train/d01_data.npy').shape)  # (100, 600, 22)
    print(np.load(r'./Train/d01_label.npy').shape)  # (100, 600, 1)
    print(np.load(r'./Test/d01_te_data.npy').shape)  # (960, 22)
    print(np.load(r'./Test/d01_te_label.npy').shape)  # (960, 1)
