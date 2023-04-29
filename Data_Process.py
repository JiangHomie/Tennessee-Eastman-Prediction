import numpy as np
import pandas as pd
import os


class DataProcess():

    def __init__(self, time_step, chanel, width, height):
        self.time_step = time_step
        self.chanel = chanel
        self.width = width
        self.height = height

    def dat_to_csv(self):
        """
        将.dat格式转换为.csv格式
        其中：训练集的d00有问题，需要转置一下
        注：d02.dat的数据量有问题，原因在源文件的第二个周期前少了一个回车，我改了一下
        :return:
        """
        for data_dir, csv_dir in zip([r'TE_Data/Train', r'TE_Data/Test'], [r'TE_Data/TrainCSV', r'TE_Data/TestCSV']):
            if not os.path.exists(csv_dir):
                os.makedirs(csv_dir)
            for data_name in os.listdir(data_dir):
                csv_path = os.path.join(csv_dir, '%s.csv' % data_name.split('.dat')[0])
                if os.path.exists(csv_path):
                    # print('文件已存在：', csv_path)
                    continue
                data = pd.read_table(os.path.join(data_dir, data_name), header=None, sep='\s+')
                if 'd00.dat' in data_name:
                    data = data.T
                data.to_csv(csv_path, index=False, header=False)

    def creat_data(self):
        """
        整个TE数据集由训练集和测试集构成，TE集中的数据由22次不同的仿真运行数据构成，TE集中每个样本都有52个观测变量。

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
        for item in ['Test_X.npy', 'Test_y.npy', 'Train_X.npy', 'Train_y.npy']:
            if item in os.listdir(r'TE_Data'):
                # print('文件已存在：', item)
                return

        def sliding_window(arr, window_size, target):
            """
            滑动窗口取样函数
            :param arr: 输入数组，shape=n * width * height
            :param window_size: 窗口大小
            :param target: 当前样本对应的故障类型
            :return:
            """
            Xs, ys = [], []
            for i in range(arr.shape[0] - window_size + 1):
                Xs.append(arr[i:i + window_size, :].reshape(window_size, 1, 7, 7))
                ys.append(int(target))
            return Xs, ys

        for csv_dir in [r'TE_Data/TrainCSV', r'TE_Data/TestCSV']:
            X_all, y_all = [], []
            for csv_name in os.listdir(csv_dir):
                data = pd.read_csv(os.path.join(csv_dir, csv_name), header=None).values
                # 引入故障前
                Xs_1, ys_1 = sliding_window(data[:160, :self.width * self.height], self.time_step, 0)
                # 引入故障后
                Xs_2, ys_2 = sliding_window(data[160:, :self.width * self.height], self.time_step, int(csv_name[1:3]))
                X_all += (Xs_1 + Xs_2)
                y_all += (ys_1 + ys_2)
            np.save(r'TE_Data/%s_X.npy' % csv_dir.split('CSV')[0].split('/')[1], np.array(X_all).astype(np.float32))
            np.save(r'TE_Data/%s_y.npy' % csv_dir.split('CSV')[0].split('/')[1], np.array(y_all).astype(np.float32))


"""
d19.dat (480, 52)
d18.dat (480, 52)
d20.dat (480, 52)
d08.dat (480, 52)
d09.dat (480, 52)
d21.dat (480, 52)
d04.dat (480, 52)
d10.dat (480, 52)
d11.dat (480, 52)
d05.dat (480, 52)
d13.dat (480, 52)
d07.dat (480, 52)
d06.dat (480, 52)
d12.dat (480, 52)
d16.dat (480, 52)
d02.dat (480, 52)
d03.dat (480, 52)
d17.dat (480, 52)
d01.dat (480, 52)
d15.dat (480, 52)
d14.dat (480, 52)
d00.dat (500, 52)
d00_te.dat (960, 52)
d12_te.dat (960, 52)
d10_te.dat (960, 52)
d02_te.dat (960, 52)
d06_te.dat (960, 52)
d18_te.dat (960, 52)
d14_te.dat (960, 52)
d20_te.dat (960, 52)
d08_te.dat (960, 52)
d16_te.dat (960, 52)
d04_te.dat (960, 52)
d01_te.dat (960, 52)
d13_te.dat (960, 52)
d11_te.dat (960, 52)
d03_te.dat (960, 52)
d19_te.dat (960, 52)
d07_te.dat (960, 52)
d15_te.dat (960, 52)
d17_te.dat (960, 52)
d09_te.dat (960, 52)
d05_te.dat (960, 52)
d21_te.dat (960, 52)
"""
