#!/usr/bin/python 
# -*- coding: UTF-8 -*-

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

DATA_PATH = '../../data/ml-100k/'
DATA_DIR_RATING = '../../data/ml-1m/ml-1m/ratings.dat'
DATA_DIR_USER = '../../data/ml-1m/ml-1m/users.dat'
DATA_DIR_ITEM = '../../data/ml-1m/ml-1m/movies.dat'
COLUMN_RATING = ['user', 'item', 'rating']
COLUMN_USERS = ['user', 'gender', 'age', 'occupation', 'zip-code']
COLUMN_ITEMS = ['item', 'title', 'genres']

def re_index(s):
    i = 1
    s_map = {}
    for key in s:
        s_map[key] = i
        i += 1
    return s_map

def load_rating_data():
    rating_data = pd.read_csv(filepath_or_buffer=DATA_DIR_RATING, header=None,
                              delimiter='::', usecols=[0, 1, 2], names=COLUMN_RATING,
                              dtype={0: str, 1: str, 2: float}, engine='python')
    print(rating_data.head())

    user_set = set(rating_data['user'].unique())
    item_set = set(rating_data['item'].unique())

    return rating_data, len(user_set), len(item_set)

def load_user_data():
    user_data = pd.read_csv(filepath_or_buffer=DATA_DIR_USER, header=None,
                            delimiter='::', names=COLUMN_USERS, engine='python',
                            dtype=str)
    print(user_data.head())
    return user_data

def load_item_data():
    item_data = pd.read_csv(filepath_or_buffer=DATA_DIR_ITEM, header=None,
                            delimiter='::', names=COLUMN_ITEMS, engine='python',
                            dtype=str)
    print(item_data.head())
    return item_data

def get_rating_feature(rating_data, user_map, item_map, user):
    rating_feature = np.zeros(len(item_map))
    user = user_map.get(user)
    user_items = rating_data[rating_data['user'] == user]
    for i in range(user_items.__len__()):
        item = item_map.get(user_items.at[i, 'item'])
        rating = user_items.at[i, 'rating']
        if item is not None:
            rating_feature[item] = rating
    print(rating_feature)
    return rating_feature

def get_user_feature(user_data, user_map, user):
    pass

def get_item_feature(item_data, item_map, item):
    pass

def dump_data(feature):
    pass

if __name__ == '__main__':
    user_data = load_user_data()
    # item_data = load_item_data()
    # rating_data = load_rating_data()
    user_map = re_index(user_data['user'].drop_duplicates())
    # item_map = re_index(item_data['item'].drop_duplicates())
    for user, u_idx in user_map.items():
        # rating_feature = get_rating_feature(rating_data, user_map, item_map, '100')
        user_feature = get_user_feature(user_data, user_map, user)

        break