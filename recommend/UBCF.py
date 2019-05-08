#!/usr/bin/python 
# -*- coding: UTF-8 -*-

import numpy as np
from loadData import *

class UBCF:
    '''
    基于用户的协同过滤
    '''
    def __init__(self, mat):
        '''
        初始化
        :param mat: 用户评分记录
        '''
        self.mat = mat
        self.rev = {}
        self.sim = {}
        self.reverse_index()
        self.similarity()

    def reverse_index(self):
        '''
        构造物品用户倒排索引
        :return:
        '''
        rev = {}
        for i in range(self.mat.shape[0]):
            uid = self.mat[i, 0]
            iid = self.mat[i, 1]
            rating = self.mat[i, 2]
            if rev.get(iid) is not None:
                rev[iid].add((uid, rating))
            else:
                rev[iid] = set()
                rev[iid].add((uid, rating))
        self.rev = rev

    def similarity(self):
        '''
        构造用户相似度矩阵
        :return:
        '''
        N = {}      # 分母
        M = {}      # 分子
        sim = {}    # 相似度
        for i, ur in self.rev.items():
            for u, r in ur:
                if N.get(u) is not None:
                    N[u] += 1
                else:
                    N[u] = 1
                for _u, _r in ur:
                    if u != _u:
                        if M.get((u, _u)) is not None:
                            M[(u, _u)] += 1
                        else:
                            M.setdefault((u, _u), 1)

        for key, value in M.items():
            sim[key] = float(value) / float(N[key[0]] + N[key[1]])
        self.sim = sim

    def rating(self, uid, iid):
        '''
        评分预测
        :param uid:
        :param iid:
        :return:
        '''
        u_list = []     # 与iid有交互的用户列表
        r_list = []     # 对iid的评分
        r_pred = 0.0      # 评分预测
        u_r = self.rev.get(iid)
        if u_r is not None:
            u_list = [t[0] for t in u_r]
            r_list = [t[1] for t in u_r]
        for u in range(len(u_list)):
            sim = self.sim.get((uid, u_list[u]))   # 用户uid与u的相似度
            if sim is None:
                sim = 0.0
            r = r_list[u]
            r_pred += float(sim)*float(r)
        return r_pred

    def recommend(self, uid, topk):
        '''
        生成推荐列表
        :param uid: user id
        :param topk: 推荐列表长度
        :return: 推荐列表
        '''
        rec_list = []
        for iid, u_r in self.rev.items():
            u_set = set([t[0] for t in u_r])
            # if uid not in u_set:      # 只为用户推荐没有见过的物品
            rating = self.rating(uid=uid, iid=iid)
            rec_list.append((iid, rating))
        rec_list.sort(key=lambda x: x[1], reverse=True)
        return rec_list[:topk]

if __name__ == '__main__':
    train, test, all = loadData()
    ubcf = UBCF(all)
    k = 10
    precision = 0.0
    recall = 0.0
    users = set([r[0] for r in ubcf.sim.keys()])
    print 'users:', len(users)
    recommend = {}
    c = 1
    for uid in users:
        print 'processing user ', c, uid
        c += 1
        count = 0.0
        i_list = set()  # 与用户uid有交互的物品
        for iid, u_r in ubcf.rev.items():
            u_set = set([t[0] for t in u_r])
            if uid in u_set:
                i_list.add(iid)
        rec_list = [t[0] for t in ubcf.recommend(uid=uid, topk=k)]
        recommend.setdefault(uid, rec_list)
        for i in rec_list:
            if i in i_list:
                count += 1
        precision += count/k
        recall += count/len(i_list)
    with open('/local/LogAnalysis/result/recommend.txt', 'w') as file:
        for user, rec_list in recommend.items():
            apps = ''
            for app in rec_list:
                apps += app+','
            file.write(user + '::' + apps + '\n')
    p = precision/len(users)
    r = recall / len(users)
    print 'precision:', p
    print 'recall:', r
    print 'f1:', 2*p*r/(p+r)
