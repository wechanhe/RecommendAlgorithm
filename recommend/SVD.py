#!/usr/bin/python 
# -*- coding: UTF-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from loadData import *

class SVD:
    def __init__(self, mat, k):
        self.mat = mat  # 评分矩阵
        self.K = k  # 隐含向量的维度
        self.bu = {}
        self.bi = {}
        self.pu = {}
        self.qi = {}
        print mat[0:10, 2]
        self.avg = 1
        for i in range(self.mat.shape[0]):
            u_id = self.mat[i, 0]
            i_id = self.mat[i, 1]
            # 初始化两个偏置量为0
            self.bu.setdefault(u_id, 0)
            self.bi.setdefault(i_id, 0)
            # 随机初始化K维向量
            self.pu.setdefault(u_id, np.random.random((self.K, 1)) / 10 * np.sqrt(self.K))
            self.qi.setdefault(i_id, np.random.random((self.K, 1)) / 10 * np.sqrt(self.K))

    def predict(self, uid, iid):
        '''
        :return:
        '''
        # setdefault的作用是当该用户或者物品未出现过时，新建它的bi,bu,qi,pu，并设置初始值为0
        self.bi.setdefault(iid, 0)
        self.bu.setdefault(uid, 0)
        self.qi.setdefault(iid, np.zeros((self.K, 1)))
        self.pu.setdefault(uid, np.zeros((self.K, 1)))
        # 预测评分公式
        # 由于评分范围在1到5，所以当分数大于5或小于1时，返回5,1.
        rating = self.avg + self.bi[iid] + self.bu[uid] + \
                 np.sum(self.qi[iid]*self.pu[uid])
        if rating > 5:
            rating = 5
        if rating < 1:
            rating = 1
        return rating

    # gamma=0.5 lam=7 k=?
    def train(self, epoches=10, lr=0.04, lam=0.15):
        '''
        模型训练，目标函数 = min 1/2*((Rui - R~ui)2 + lam(bu2 + bi2 + |pu|2 + |qi|2))
        R~ui = pu*qi + bu + bi + U
        eui = Rui - R~ui
        bu = bu + lr*(eui - lam*bu)
        bi = bi + lr*(eui - lam*bi)
        pu = pu + lr*(eui*qi - lam*pu)
        qi = pi + lr*(eui*pu - lam*qi)
        steps: 迭代次数
        gamma: 学习率
        lam: 规范化参数，防止过拟合
        :return:
        '''
        min = 10
        print('train data size', self.mat.shape)
        for epoch in range(epoches):
            print 'epoch', epoch + 1, 'is running'
            # permutatioin返回一个新的打乱顺序的数组，并不改变原来的数组
            # 在这里的目的是迭代多次，每次以不同的次序使用数据集
            KK = np.random.permutation(self.mat.shape[0])
            # 随机梯度下降算法，kk为对矩阵进行随机洗牌
            rmse = 0.0
            for i in range(self.mat.shape[0]):
                j = KK[i]
                uid = self.mat[j, 0]
                iid = self.mat[j, 1]
                rating = self.mat[j, 2]
                eui = rating - self.predict(uid, iid)
                rmse += eui**2
                self.bu[uid] += lr*(eui-lam*self.bu[uid])
                self.bi[iid] += lr*(eui-lam*self.bi[iid])
                tmp = self.qi[iid]
                self.qi[iid] += lr*(eui*self.pu[uid]-lam*self.qi[iid])
                self.pu[uid] += lr*(eui*tmp-lam*self.pu[uid])
                lr = 0.98*lr
            rmse = np.sqrt(rmse / self.mat.shape[0])
            print 'rmse is', rmse
            if rmse < min:
                min = rmse
        return min

    def test(self, test_data):
        '''
        预测
        gamma以0.93的学习率递减
        :return:
        '''
        test_data = np.array(test_data)
        print 'test data size', test_data.shape
        rmse = 0.0
        for i in range(test_data.shape[0]):
            uid = test_data[i, 0]
            iid = test_data[i, 1]
            rating = test_data[i, 2]
            eui = rating - self.predict(uid, iid)
            rmse += eui ** 2
        print 'rmse of test data is', np.sqrt(rmse / test_data.shape[0])

if __name__ == '__main__':
    # train_data, test_data, all_data = loadData()
    # svd = SVD(all_data, k=20)
    # svd.train()
    # svd.test(test_data)
    # rmse = {}
    # for i in range(10):
    #     k = 10*i + 10
    #     svd = SVD(all_data, k=k)
    #     rmse[str(k)] = svd.train()
    # for key, value in sorted(rmse.items(), key=lambda x:x[1]):
    #     print key, value
    # gamma = {}
    # for i in range(10):
    #     gam = 0.1*i + 0.1
    #     gamma[str(gam)] = svd.train(gamma=gam)
    # for key, value in sorted(gamma.items(), key=lambda x:x[1]):
    #     print key, value
    # svd.test(test_data)
    from surprise import SVD
    from surprise import Dataset
    from surprise.model_selection import cross_validate

    # Load the movielens-100k dataset (download it if needed).
    data = Dataset.load_builtin('ml-100k')
    pass
