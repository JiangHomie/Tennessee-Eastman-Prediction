# -*- coding: utf-8 -*-
"""
Author: JHM
Date: 2023-05-31
Description: 创建邻接矩阵
"""
import matplotlib.pyplot as plt
import numpy as np


adj = np.zeros((22, 22)) + np.eye(22)
adj[0, 5] = 1
adj[0, 6] = 1
adj[0, 7] = 1
adj[1, 5] = 1
adj[2, 5] = 1
adj[2, 7] = 1
adj[3, 0] = 1
adj[3, 5] = 1
adj[3, 7] = 1
adj[3, 15] = 1
adj[4, 6] = 1
adj[4, 15] = 1
adj[6, 11] = 1
adj[6, 12] = 1
adj[7, 1] = 1
adj[7, 8] = 1
adj[7, 10] = 1
adj[8, 20] = 1
adj[9, 12] = 1
adj[10, 12] = 1
adj[10, 17] = 1
adj[10, 21] = 1
adj[11, 13] = 1
adj[12, 9] = 1
adj[12, 15] = 1
adj[13, 11] = 1
adj[13, 14] = 1
adj[14, 16] = 1
adj[15, 17] = 1
adj[16, 14] = 1
adj[17, 18] = 1
adj[18, 17] = 1
adj[20, 8] = 1
adj[21, 10] = 1

np.save('adj_matrix.npy', adj.astype(int))

plt.figure()
plt.imshow(adj, cmap='binary', interpolation='nearest')
plt.show()