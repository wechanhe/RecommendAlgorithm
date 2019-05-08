#!/usr/bin/python 
# -*- coding: UTF-8 -*-
import random
from collections import defaultdict
import numpy as np
import tensorflow as tf

def load_data():
    '''
    读取用户交互记录
    :return:
    '''
    path = '/local/LogAnalysis/data/data.txt'
    # path = '/local/LogAnalysis/data/ml-1m/ratings.dat'
    # path = '/local/LogAnalysis/data/ml-100k/ml-100k/u.data'
    users = defaultdict(int)
    items = defaultdict(int)
    user_items = defaultdict(set)
    with open(path, 'r') as file:
        u_idx = 0
        i_idx = 0
        lines = file.readlines()[0:100000]
        for idx in xrange(len(lines)):
            cols = lines[idx].split(',')
            # cols = lines[idx].split('\t')
            u = cols[0].strip('\t\n\r')
            i = cols[1].strip('\t\n\r')
            if users.get(u) is None:
                users.setdefault(u, u_idx)
                u_idx += 1
            if items.get(i) is None:
                items.setdefault(i, i_idx)
                i_idx += 1
            if user_items.get(u) is not None:
                user_items[u].add(i)
            else:
                user_items.setdefault(u, set())
    print 'users', len(users), 'items', len(items), 'user_items', len(user_items)
    return users, items, user_items

def select_test_sample(user_items):
    '''
    给每个用户随机选择一个物品i，用于构造测试集
    :param user_items:
    :return:
    '''
    user_test = {}
    for u, i_list in user_items.items():
        user_test.setdefault(u, random.sample(i_list, 1)[0])
    return user_test

def generate_train_batch(users, items, user_items, user_test, batch_size=100):
    '''
    构造训练数据
    :return: dtype:{u, i, j} rtype:np.array
    '''
    train_set = []
    for idx in xrange(batch_size):
        u = random.sample(users.keys(), 1)[0]
        i = items.get(random.sample(user_items[u], 1)[0])
        while i == user_test[u]:
            i = items.get(random.sample(user_items[u], 1)[0])
        j = items.get(random.sample(items.keys(), 1)[0])
        while j in user_items[u]:
            j = items.get(random.sample(items.keys(), 1)[0])
        if u != None and i != None and j != None:
            train_set.append([users.get(u), i, j])
    return np.asarray(train_set)

def generate_test_batch(users, items, user_items, user_test):
    '''
    构造测试数据
    :return: dtype:{u, i ,j}  rtype:np.array
    '''
    test_set = []
    for u, i_list in user_items.items():
        i = user_test[u]
        for j in items.keys():
            if j not in user_items[u]:
                test_set.append([users.get(u), items.get(i), items.get(j)])
    yield np.asarray(test_set)

class BPR:
    def __init__(self, users=None, items=None, user_items=None, user_test=None, emb_dim=None, reg=None, lr=None):
        self.emb_dim = emb_dim
        self.reguration_rate = reg
        self.lr = lr
        self.u = None
        self.i = None
        self.j = None
        self.training_op = None
        self.bpr_loss = None
        self.mf_auc = None
        self.users = users
        self.items = items
        self.users_items = user_items
        self.user_test = user_test

    def build_model(self):
        with tf.name_scope('input'):
            u = tf.placeholder(tf.int32, [None], name='u')
            i = tf.placeholder(tf.int32, [None], name='i')
            j = tf.placeholder(tf.int32, [None], name='j')

        with tf.name_scope('user_emb'):
            user_emb_w = tf.get_variable(name='user_emb', shape=[len(self.users), self.emb_dim],
                                         initializer=tf.initializers.random_normal(0, 0.1))
        with tf.name_scope('item_emb'):
            item_emb_w = tf.get_variable(name='item_emb', shape=[len(self.items), self.emb_dim],
                                         initializer=tf.initializers.random_normal(0, 0.1))
        u_emb = tf.nn.embedding_lookup(user_emb_w, u)
        i_emb = tf.nn.embedding_lookup(item_emb_w, i)
        j_emb = tf.nn.embedding_lookup(item_emb_w, j)

        x = tf.reduce_sum(tf.multiply(u_emb, (i_emb - j_emb)), 1, keep_dims=True)
        loss1 = -tf.reduce_mean(tf.log(tf.sigmoid(x)))
        l2_norm = tf.add_n([
            tf.reduce_sum(tf.multiply(u_emb, u_emb)),
            tf.reduce_sum(tf.multiply(i_emb, i_emb)),
            tf.reduce_sum(tf.multiply(j_emb, j_emb))
        ])

        mf_auc = tf.reduce_mean(tf.to_float(x > 0), name='mf_auc')

        with tf.name_scope('loss'):
            bpr_loss = loss1 + self.reguration_rate * l2_norm
        training_op = tf.train.GradientDescentOptimizer(self.lr).minimize(bpr_loss)
        return u, i, j, training_op, bpr_loss, mf_auc

    def train(self, epochs, batch_size):
        with tf.Session() as sess:
            u, i, j, training_op, bpr_loss, mf_auc = self.build_model()
            sess.run(tf.global_variables_initializer())
            for epoch in xrange(epochs):
                #### train
                batch_bprloss = 0.0
                print 'running epoch', epoch + 1,
                batch = 100
                for k in xrange(batch_size):
                    uij = generate_train_batch(users=self.users, items=self.items,
                                               user_items=self.users_items, user_test=self.user_test)
                    training, loss = sess.run([training_op, bpr_loss], {u: uij[:, 0], i: uij[:, 1], j: uij[:, 2]})
                    batch_bprloss += loss
                loss = batch_bprloss / batch_size*1.0
                print 'loss', loss

                # user_count = 0.0
                # _auc_sum = 0.0
                # for t_uij in generate_test_batch(users, items, user_items, user_test):
                #     _auc, _test_bprloss = sess.run([mf_auc, bpr_loss], {u: t_uij[:, 0], i: t_uij[:, 1], j: t_uij[:, 2]})
                #     user_count += 1
                #     _auc_sum += _auc
                # print "test_loss: ", _test_bprloss, "test_auc: ", _auc_sum / user_count

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            saver.save(sess, '/local/LogAnalysis/model/BPR/bpr.ckpt')

    def recommend(self, u_name='', topn=10):
        '''
        向用户推荐物品
        :param uid:
        :return:
        '''
        with tf.Session() as sess:
            tf.train.Saver().restore(sess, '/local/LogAnalysis/model/BPR/bpr.ckpt')
            uid = self.users.get(u_name)
            variable_names = [v.name for v in tf.trainable_variables()]
            values = sess.run(variable_names)
            # value[0]:(1882, 20), value[1]:(5878, 20)

        session1 = tf.Session()
        # 将一维数组变成 1*dim
        u1_dim = tf.expand_dims(values[0][uid], 0)
        u1_all = tf.matmul(u1_dim, values[1], transpose_b=True)
        result_1 = session1.run(u1_all)

        print "以下是给用户%s的推荐：" % u_name
        print 'App_package_name', '\t\t\t\t', 'rating'
        # 变成一维数组：从数组的形状中删除单维度条目，即把shape中为1的维度去掉
        p = np.squeeze(result_1)
        # 将p中的元素从小到大排列，提取其对应的index(索引)，然后输出到y
        idx = np.argsort(p)
        res = []
        for i in xrange(topn):
            index = idx[-(i + 1)]
            res.append((index, [item for item, k in items.items() if k == index][0], p[index]))
            print [item for item, k in items.items() if k == index][0], '\t', p[index]
        return res

def evaluate():
    precision = 0.0
    recall = 0.0
    topk = 10
    epoch = 1
    for user, u_idx in users.items():
        # print 'processing user', epoch
        epoch += 1
        i_list = user_items.get(user)  # 用户有交互物品
        r_list = bpr.recommend(u_name=user, topn=topk)
        count = 0.0
        for item in r_list:
            if item[1] in i_list:
                count += 1
        precision += count / topk * 1.0
        recall += count / len(i_list) * 1.0
    p = precision / len(users) * 1.0
    r = recall / len(users) * 1.0
    f1 = 2.0 * p * r / (p + r)
    print 'precision', p, \
        'recall', r, \
        'f1', f1

if __name__ == '__main__':
    users, items, user_items = load_data()
    user_test = select_test_sample(user_items)
    bpr = BPR(users=users, items=items, user_items=user_items, user_test=user_test,
              emb_dim=20, reg=0.0001, lr=0.01)
    bpr.train(epochs=10, batch_size=1000)
    # evaluate()
    bpr.recommend('10002bd24c15bb4d')