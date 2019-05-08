#!/usr/bin/python 
# -*- coding: UTF-8 -*-

import tensorflow as tf
import numpy as np

def add_layer(inputs, in_size, out_size, activation_func=None, n_layer=0):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='w')
            tf.summary.histogram(layer_name+'/weights', Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.53, name='b')
            tf.summary.histogram(layer_name+'/biases', biases)
        with tf.name_scope('wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
            tf.summary.histogram(layer_name+'/wx_plus_b', Wx_plus_b)
        if activation_func is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_func(Wx_plus_b)
        tf.summary.histogram(layer_name+'/outputs', outputs)
    return outputs

def auto_encoder():
    from user_profile.preprocessing import load_feature
    x_data = load_feature()
    n_inputs = x_data.shape[1]
    n_hidden = 1000
    n_outputs = n_inputs
    lr = 0.1
    # x_data = np.random.random([100, 100])
    print x_data[0]
    # 输入
    keep_prob = tf.placeholder(tf.float32)
    with tf.name_scope('input'):
        xs = tf.placeholder(tf.float32, [None, n_inputs], name='input')
        # ys = tf.placeholder(tf.float32, [None, 100])
    # l1 = add_layer(xs, n_inputs, n_hidden, activation_func=tf.nn.relu, n_layer=1)
    # prediction = add_layer(l1, n_hidden, n_outputs, activation_func=None, n_layer=2)
    l1 = tf.layers.dense(xs, n_hidden, activation=tf.nn.relu, name='hidden1',
                         kernel_regularizer=tf.keras.regularizers.l2())
    prediction = tf.layers.dense(l1, n_outputs, activation=None, name='output')
    # 损失函数
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.square(xs - prediction))
        tf.summary.scalar('loss', loss)
    # 训练
    with tf.name_scope('train'):
        train_step = tf.train.GradientDescentOptimizer(learning_rate=lr).minimize(loss)
    init = tf.global_variables_initializer()
    merge = tf.summary.merge_all()
    with tf.Session() as sess:
        sess.run(init)
        writer = tf.summary.FileWriter('../logs/AE', sess.graph)
        for i in range(100):
            # training
            sess.run(train_step, feed_dict={xs: x_data, keep_prob: 0.5})
            if i % 10 == 0:
                # to see the step improvement
                print(sess.run(loss, feed_dict={xs: x_data, keep_prob: 1}))
                result = sess.run(merge, feed_dict={xs: x_data, keep_prob: 1})
                writer.add_summary(result, global_step=i)
        saver = tf.train.Saver()
        saver.save(sess, '../model/recommend/ae_net.ckpt')
        writer.close()

def load_model():
    # 先建立 W, b 的容器
    W = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name="weights")
    b = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name="biases")

    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.53)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # 提取变量
        saver.restore(sess, "model/recommend/ae_net.ckpt")
        print("weights:", sess.run(W))
        print("biases:", sess.run(b))

if __name__ == '__main__':
    auto_encoder()