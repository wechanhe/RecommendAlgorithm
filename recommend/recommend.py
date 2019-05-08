#!/usr/bin/python 
# -*- coding: UTF-8 -*-

from SVD import SVD
from loadData import getData

def recommend(uid, k):
    '''
    生成推荐列表
    uid : 用户id
    k : 推荐列表长度
    :return: 用户uid的推荐列表
    '''
    train_data, test_data, all_data = getData()
    model = SVD(all_data, k=20)
    model.train()
    # svd.test(test_data)
    # 存储所有物品
    all_items = set()
    # 存储用户uid有交互的所有物品
    u_items = set()
    # 推荐列表
    rec_list = {}
    for record in all_data:
        all_items.add(record[1])
        if record[0] == uid:
            u_items.add(record[1])
    unseen = all_items - u_items
    for item in unseen:
        rating = model.predict(uid, item)
        rec_list[item] = rating
    rec_list = sorted(rec_list.items(), key=lambda x: x[1], reverse=True)
    for item, rating in rec_list[:10]:
        print item, rating

if __name__ == '__main__':
    recommend(1, k=100)
