#!/usr/bin/python 
# -*- coding: UTF-8 -*-

import time
import datetime
from sklearn.model_selection import train_test_split  # 数据集划分训练集和测试集
import numpy as np
import tensorflow as tf
from preprocessing import *

def save_params(params):
    """
    Save parameters to file
    """
    pickle.dump(params, open('params/params.p', 'wb'))


def load_params():
    """
    Load parameters from file
    """
    return pickle.load(open('params/params.p', mode='rb'))


# 从本地读取数据
movies_name, title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = pickle.load(
    open('feature/feature.p', mode='rb'))
print features[0]
# 嵌入矩阵的维度：一个单词或其他变量的特征表示
embed_dim = 32
# features为
# [ [1, 1193, 0, ..., 10,list([ title]),list([ genres])],
#   [2, 1193, 1, ..., 16,list([ ]),list([ ])],
#   [12, 1193, 1, ..., 12,list([ ]),list([ ])],
#   ...,
#   [5938, 2909, 1, ..., 1,list([ ]),list([ ])]
# ]

# 用户ID个数
uid_max = max(features.take(0, 1)) + 1  # 6040
# features.take(0,1)得到userid的全部列，由于从0开始编号，则max取最大值再加1可以得到用户id个数
# ndarray.take(indices, axis=None, out=None, mode='raise')从轴axis上获取数组中的元素，并以一维数组或者矩阵返回
# 按axis选择处于indices位置上的值
# axis用于选择值的轴，0为横轴，1为纵向选
# 如features.take(0,0)就会选择横向第一条数据，(1,0)会选择横向第二条数据
# 性别个数
gender_max = max(features.take(2, 1)) + 1  # 1 + 1 = 2
# 年龄类别个数
age_max = max(features.take(3, 1)) + 1  # 6 + 1 = 7
# 职业个数
job_max = max(features.take(4, 1)) + 1  # 20 + 1 = 21
# 电影ID个数
movie_id_max = max(features.take(1, 1)) + 1  # 3952
# 电影类型个数
movie_categories_max = max(genres2int.values()) + 1  # 18 + 1 = 19
# 电影名单词个数
movie_title_max = len(title_set)  # 5216
# title_set是由空格分开的电影单词字符串构成的列表(set表)

# 对电影类型嵌入向量做加和操作的标志,后面调用combiner来使用作为参数
combiner = "sum"

# 电影名长度
sentences_size = title_count  # title_count=15重命名，一个电影title字段的长度，不够会补
# 文本卷积滑动窗口，分别滑动2, 3, 4, 5个单词
window_sizes = {2, 3, 4, 5}
# 文本卷积核数量
filter_num = 8

# 电影ID转下标的字典，注意数据集中电影ID跟下标不一致，比如第5行的数据电影ID不一定是5
movieid2idx = {val[0]: i for i, val in enumerate(movies.values)}  # 格式为{movieid ：i}
# 1:0,2:1，...

# 超参-----------------------------------------------
# Number of Epochs
num_epochs = 5
# Batch Size
batch_size = 256

dropout_keep = 0.5
# Learning Rate
learning_rate = 0.0001
# Show stats for every n number of batches
show_every_n_batches = 20
# 后面要保存模型的地址参数赋给变量save_dir

save_dir = 'save/'

# 输入-----------------------------------------------
# 定义输入的占位符
def get_inputs():
    # tf.placeholder(dtype, shape=None, name=None)此函数可以理解为形参，用于定义过程，在执行的时候再赋具体的值
    # dtype：数据类型
    # shape：数据形状。默认是None,[行数，列数]
    # name：名称
    uid = tf.placeholder(tf.int32, [None, 1], name="uid")  # 这里一行代表一个用户的id，是batch×1，每一行是一个列表
    user_gender = tf.placeholder(tf.int32, [None, 1], name="user_gender")
    user_age = tf.placeholder(tf.int32, [None, 1], name="user_age")
    user_job = tf.placeholder(tf.int32, [None, 1], name="user_job")

    movie_id = tf.placeholder(tf.int32, [None, 1], name="movie_id")
    movie_categories = tf.placeholder(tf.int32, [None, 18], name="movie_categories")
    # movie_titles = tf.placeholder(tf.int32, [None, 15], name="movie_titles")
    movie_titles = tf.placeholder(tf.float32, [None, embed_dim], name="movie_titles")

    targets = tf.placeholder(tf.int32, [None, 1], name="targets")
    LearningRate = tf.placeholder(tf.float32, name="LearningRate")
    dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")
    return uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, LearningRate, dropout_keep_prob


# 构建神经网络---------------------------------------
# 定义User的嵌入矩阵，完成原始矩阵经过嵌入层后得到的输出
def get_user_embedding(uid, user_gender, user_age, user_job):  # 见作者结构图，用户嵌入矩阵输入4个
    with tf.name_scope("user_embedding"):  # 用于后面tensorboard可视化图层关系
        # 用户id
        uid_embed_matrix = tf.Variable(tf.random_uniform([uid_max, embed_dim], -1, 1), name="uid_embed_matrix")
        # tf.Variable(initializer,name)：initializer是初始化参数，可以有tf.random_normal,tf.constant等,name是变量的名字
        # tf.random_uniform(shape, minval=0,maxval=None,dtype=tf.float32) 从均匀分布中输出随机值。
        # 返回shape形状矩阵：用户数×特征数，产生于low(-1)和high(1)之间，产生的值是均匀分布的。
        uid_embed_layer = tf.nn.embedding_lookup(uid_embed_matrix, uid, name="uid_embed_layer")
        # tf.nn.embedding_lookup(tensor, id) 选取一个张量tensor里面索引id对应的元素
        # 选取uid_embed_matrix的用户id对应的某个用户id的向量

        # 性别
        gender_embed_matrix = tf.Variable(tf.random_uniform([gender_max, embed_dim // 2], -1, 1),
                                          name="gender_embed_matrix")
        # 这里特征数降一半
        gender_embed_layer = tf.nn.embedding_lookup(gender_embed_matrix, user_gender, name="gender_embed_layer")
        # 选取gender_embed_matrix的用户性别对应的某个用户性别的向量

        # 年龄
        age_embed_matrix = tf.Variable(tf.random_uniform([age_max, embed_dim // 2], -1, 1), name="age_embed_matrix")
        age_embed_layer = tf.nn.embedding_lookup(age_embed_matrix, user_age, name="age_embed_layer")
        # 选取age_embed_matrix的用户年龄对应的某个用户年龄的向量

        # job
        job_embed_matrix = tf.Variable(tf.random_uniform([job_max, embed_dim // 2], -1, 1), name="job_embed_matrix")
        job_embed_layer = tf.nn.embedding_lookup(job_embed_matrix, user_job, name="job_embed_layer")
        # 选取job_embed_matrix的用户job对应的某个用户job的向量
    return uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer


# 将User的嵌入矩阵一起全连接生成User的特征
def get_user_feature_layer(uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer):
    with tf.name_scope("user_fc"):
        # 第一层全连接
        uid_fc_layer = tf.layers.dense(uid_embed_layer, embed_dim, name="uid_fc_layer", activation=tf.nn.relu)
        gender_fc_layer = tf.layers.dense(gender_embed_layer, embed_dim, name="gender_fc_layer", activation=tf.nn.relu)
        age_fc_layer = tf.layers.dense(age_embed_layer, embed_dim, name="age_fc_layer", activation=tf.nn.relu)
        job_fc_layer = tf.layers.dense(job_embed_layer, embed_dim, name="job_fc_layer", activation=tf.nn.relu)
        # tf.layers.dense(inputs,units,activation=None,use_bias=True,kernel_initializer=None,bias_initializer=tf.zeros_initializer(),
        #                                kernel_regularizer=None,bias_regularizer=None,activity_regularizer=None,kernel_constraint=None,
        #                                bias_constraint=None,trainable=True,name=None,reuse=None)
        # inputs:该层的输入; units:输出的大小(维数),整数或long; activation: 使用什么激活函数（神经网络的非线性层），默认为None，不使用激活函数
        # name该层的名字

        # 第二层全连接
        user_combine_layer = tf.concat([uid_fc_layer, gender_fc_layer, age_fc_layer, job_fc_layer], 2)  # (?, 1, 128)
        # tf.concat(values,axis,name)是连接两个矩阵的操作，本身不会增加维度，返回的是连接后的tensor.
        # values应该是一个tensor的list或者tuple。
        # axis则是我们想要连接的维度。对于二维来说0表示第一个括号维度，1表示第二个括号维度
        # t1 = [[1, 2, 3], [4, 5, 6]]
        # t2 = [[7, 8, 9], [10, 11, 12]]
        # tf.concat([t1, t2], 0)表示为 [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        # tf.concat([t1, t2], 1)表示为 [[1, 2, 3, 7, 8, 9], [4, 5, 6, 10, 11, 12]]
        #     对于三维来说0表示第一个括号维度，1表示第二个括号维度，2表示第三个括号维度
        # t1 = [ [[1],[2]] , [[3],[4]] ]
        # t2 = [ [[5],[6]] , [[7],[8]] ]
        # tf.concat([t1, t2], 0)表示为[[[1],[2]] , [[3],[4]] , [[5],[6]] , [[7],[8]]]
        # tf.concat([t1, t2], 1)表示为[[[1],[2],[5],[6]],[[3],[4],[7],[8]]]
        # tf.concat([t1, t2], 2)表示为[[[1 5],[2 6]],[[3 7],[4 8]]]

        user_combine_layer = tf.contrib.layers.fully_connected(user_combine_layer, 200, tf.tanh)  # (?, 1, 200)
        # tf.contrib.layers.fully_connected（F输入, num_outputs,activation_fn）用于构建全连接层

        user_combine_layer_flat = tf.reshape(user_combine_layer, [-1, 200])  # -1表示缺省值，满足其他维度要求，这里该是几就是几，(?, 200)
    return user_combine_layer, user_combine_layer_flat


# 对电影类型的多个嵌入向量做加和
def get_movie_categories_layers(movie_categories):
    with tf.name_scope("movie_categories_layers"):
        movie_categories_embed_matrix = tf.Variable(tf.random_uniform([movie_categories_max, embed_dim], -1, 1),
                                                    name="movie_categories_embed_matrix")
        movie_categories_embed_layer = tf.nn.embedding_lookup(movie_categories_embed_matrix, movie_categories,
                                                              name="movie_categories_embed_layer")  # (?,18,32)
        if combiner == "sum":
            movie_categories_embed_layer = tf.reduce_sum(movie_categories_embed_layer, axis=1,
                                                         keep_dims=True)  # (?,1,32)
            # tf.reduce_sum(input_tensor,axis=None,keep_dims=False,name=None,reduction_indices=None)要输出看！
            # 压缩求和，用于降维
            # 函数中的input_tensor是按照axis中已经给定的维度来减少的，axis表示按第几个维度求和
            # 但是keep_dims为true，则维度不会减小
            # 如果axis没有条目，则缩小所有维度，并返回具有单个元素的张量
            # 'x' is [[1, 1, 1]
            #        [1, 1, 1]]
            # 求和tf.reduce_sum(x) ==> 6
            # 按列求和tf.reduce_sum(x, 0) ==> [2, 2, 2]（即得到行结果）
            # 按行求和tf.reduce_sum(x, 1) ==> [3, 3] （即得到列结果）
            # 按照行的维度求和tf.reduce_sum(x, 1, keep_dims=True) ==> [[3], [3]]（维度不会减少）
            # x现在一行为一个类型向量，应该按axis=1进行sum
    #     elif combiner == "mean":
    return movie_categories_embed_layer


# 定义Movie ID的嵌入矩阵
def get_movie_id_embed_layer(movie_id):
    with tf.name_scope("movie_embedding"):
        movie_id_embed_matrix = tf.Variable(tf.random_uniform([movie_id_max, embed_dim], -1, 1),
                                            name="movie_id_embed_matrix")
        movie_id_embed_layer = tf.nn.embedding_lookup(movie_id_embed_matrix, movie_id, name="movie_id_embed_layer")
    return movie_id_embed_layer


# Movie Title的文本卷积网络实现
def get_movie_cnn_layer(movie_titles):
    # 从嵌入矩阵中得到电影名对应的各个单词的嵌入向量
    with tf.name_scope("movie_embedding"):
        movie_title_embed_matrix = tf.Variable(tf.random_uniform([movie_title_max, embed_dim], -1, 1),
                                               name="movie_title_embed_matrix")
        movie_title_embed_layer = tf.nn.embedding_lookup(movie_title_embed_matrix, movie_titles,
                                                         name="movie_title_embed_layer")
        movie_title_embed_layer_expand = tf.expand_dims(movie_title_embed_layer, -1)  # (?,15,32,1)
        # tf.expand_dims(input, axis, name=None)在axis轴，维度增加一维，数值为1，axis=-1表示最后一位
        # 当然,我们常用tf.reshape(input, shape=[])可达到相同效果,但有时在构建图的过程中,placeholder没有被feed具体的值,就会报错
        # 't' is a tensor of shape [2]
        # shape(expand_dims(t, 0)) ==> [1, 2]
        # shape(expand_dims(t, 1)) ==> [2, 1]
        # shape(expand_dims(t, -1)) ==> [2, 1]

    # 对文本嵌入层使用不同尺寸的卷积核做卷积和最大池化
    pool_layer_lst = []
    for window_size in window_sizes:  # [2,3,4,5]
        with tf.name_scope("movie_txt_conv_maxpool_{}".format(window_size)):  # 格式化字符串的函数 str.format()
            # 通过 {} 和 : 来代替以前的 %
            filter_weights = tf.Variable(tf.truncated_normal([window_size, embed_dim, 1, filter_num], stddev=0.1),
                                         name="filter_weights")
            # 初始化卷积核参数
            # tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
            # 得到正态分布输出为shape，mean均值，stddev标准差
            filter_bias = tf.Variable(tf.constant(0.1, shape=[filter_num]), name="filter_bias")
            # 初始化bias
            # tf.constant(value,dtype=None,shape=None,name=’Const’) 创建一个常量tensor,按照给出value来赋值,可以用shape来指定其形状
            # 这里的shape表示(8,)
            conv_layer = tf.nn.conv2d(movie_title_embed_layer_expand, filter_weights, [1, 1, 1, 1], padding="VALID",
                                      name="conv_layer")
            # tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
            # input：指需要做卷积的输入图像 filter：相当于CNN中的卷积核 strides：卷积时在图像每一维的步长通常为[1,X,X,1]
            # padding：string类型的量，只能是"SAME","VALID"其中之一,这里不填充
            # use_cudnn_on_gpu:bool类型，是否使用cudnn加速，默认为true
            relu_layer = tf.nn.relu(tf.nn.bias_add(conv_layer, filter_bias),
                                    name="relu_layer")  # 最后迭代到windowsize=5时(?, 11, 1, 8)

            maxpool_layer = tf.nn.max_pool(relu_layer, [1, sentences_size - window_size + 1, 1, 1], [1, 1, 1, 1],
                                           padding="VALID", name="maxpool_layer")
            # (?, 1, 1, 8)
            # tf.nn.max_pool(value, ksize, strides, padding, name=None)
            # value：需要池化的输入
            # ksize：池化窗口的大小，取一个四维向量，一般是[1, height, width, 1]，因为我们不想在batch和channels上做池化，所以这两个维度设为了1
            # height=15-windowsize
            # strides：和卷积类似，窗口在每一个维度上滑动的步长，一般也是[1, stride,stride, 1]
            pool_layer_lst.append(maxpool_layer)
            # 这里最后得到4个tensor堆叠的列表
            # [<tf.Tensor 'movie_txt_conv_maxpool_2/maxpool_layer:0' shape=(?, 1, 1, 8) dtype=float32>,
            # <tf.Tensor 'movie_txt_conv_maxpool_3/maxpool_layer:0' shape=(?, 1, 1, 8) dtype=float32>,
            # <tf.Tensor 'movie_txt_conv_maxpool_4/maxpool_layer:0' shape=(?, 1, 1, 8) dtype=float32>,
            # <tf.Tensor 'movie_txt_conv_maxpool_5/maxpool_layer:0' shape=(?, 1, 1, 8) dtype=float32>]

    # Dropout层
    with tf.name_scope("pool_dropout"):
        pool_layer = tf.concat(pool_layer_lst, 3, name="pool_layer")  # (?, 1, 1, 32)
        max_num = len(window_sizes) * filter_num  # 32
        pool_layer_flat = tf.reshape(pool_layer, [-1, 1, max_num], name="pool_layer_flat")  # (?, 1, 32)

        dropout_layer = tf.nn.dropout(pool_layer_flat, dropout_keep_prob, name="dropout_layer")  # (?, 1, 32)
        # tf.nn.dropout(x, keep_prob, noise_shape=None, seed=None, name=None)
        # x：输入;keep_prob：保留比例,取值 (0,1],每一个参数都将按这个比例随机变更
        # dropout是CNN中防止过拟合的一个trick
        # dropout是指在深度学习网络的训练过程中，对于神经网络单元，按照一定的概率将其暂时从网络中丢弃。
        # 注意是暂时，对于随机梯度下降来说，由于是随机丢弃，故而每一个mini-batch都在训练不同的网络。
    return pool_layer_flat, dropout_layer


# Movie Title的word2vec实现
def get_movie_w2v_layer(movie_titles):
    with tf.name_scope("movie_embedding"):
        movie_title_embed_layer = movie_titles
        return tf.reshape(movie_title_embed_layer, [-1, 1, embed_dim])

# 将Movie的各个层一起做全连接
def get_movie_feature_layer(movie_id_embed_layer, movie_categories_embed_layer, dropout_layer):
    with tf.name_scope("movie_fc"):
        # 第一层全连接
        movie_id_fc_layer = tf.layers.dense(movie_id_embed_layer, embed_dim,
                                            name="movie_id_fc_layer",
                                            activation=tf.nn.relu)
        movie_categories_fc_layer = tf.layers.dense(movie_categories_embed_layer, embed_dim,
                                                    name="movie_categories_fc_layer",
                                                    activation=tf.nn.relu)

        # 第二层全连接
        movie_combine_layer = tf.concat([movie_id_fc_layer, movie_categories_fc_layer, dropout_layer], 2)  # (?, 1, 96)
        movie_combine_layer = tf.contrib.layers.fully_connected(movie_combine_layer, 200, tf.tanh)  # (?, 1, 200)

        movie_combine_layer_flat = tf.reshape(movie_combine_layer, [-1, 200])  # (?,200)
    return movie_combine_layer, movie_combine_layer_flat


# 构建计算图-----------------------------------------

# 变量常量等等基本量的操作设置完成，意味着最基本的东西都有了，然后接下来很重要的就是那些量和操作怎么组成更大的集合，怎么运行这个集合
# 这些就是计算图谱Graph和Session的作用：（参见https://blog.csdn.net/xierhacker/article/details/53860379）
# 一、graph
# 一个TensorFlow的运算，被表示为一个数据流的图。
# 一幅图中包含一些操作（Operation）对象，这些对象是计算节点。前面说过的Tensor对象，则是表示在不同的操作（operation）间的数据节点
# 一旦开始任务，就已经有一个默认的图创建好了。可以通过调用tf.get_default_graph()来访问。添加一个操作到默认的图里面，只需调用一个定义了新操作的函数
# 另外一种典型用法是要使用到Graph.as_default()的上下文管理器(context manager),它能够在这个上下文里面覆盖默认的图.要在某个graph里面定义量,要在with语句的范围里面定义
# 二.Session(tf.Session)
# 运行TensorFLow操作（operations）的类,一个Seesion包含了操作对象执行的环境
#

# tensorflow中的计算以图数据流的方式表示，一个图包含一系列表示计算单元的操作对象，以及在图中流动的数据单元以tensor对象
tf.reset_default_graph()  # 用于清除默认图形堆栈并重置全局默认图形
train_graph = tf.Graph()  # first creat a simple graph

with train_graph.as_default():  # define a simple graph
    # 返回一个上下文管理器,使得这个Graph对象成为当前默认的graph.当你想在一个进程里面创建多个图的时候,就应该使用这个函数.
    # 为了方便,一个全局的图对象被默认提供,要是没有显式创建一个新的图的话,所有的操作(ops)都会被添加到这个默认的图里面来.
    # 通过with关键字和这个方法,来让这个代码块内创建的从操作(ops)添加到这个新的图里面.

    # with Object() as obj:的时候,自动调用obj对象的__enter__()方法,而当出去with block的时候,又会调用obj对象的__exit__方法。
    # 正是利用 __enter__()和__exit__()，才实现类似上下文管理器的作用。
    # 参见https://blog.csdn.net/u012436149/article/details/73555017
    # 获取输入占位符
    uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, lr, dropout_keep_prob = get_inputs()
    # 获取User的4个嵌入向量
    uid_embed_layer, gender_embed_layer, age_embed_layer, job_embed_layer = get_user_embedding(uid, user_gender,
                                                                                               user_age, user_job)
    # 得到用户特征
    user_combine_layer, user_combine_layer_flat = get_user_feature_layer(uid_embed_layer, gender_embed_layer,
                                                                         age_embed_layer, job_embed_layer)

    # 获取电影ID的嵌入向量
    movie_id_embed_layer = get_movie_id_embed_layer(movie_id)

    # 获取电影类型的嵌入向量
    movie_categories_embed_layer = get_movie_categories_layers(movie_categories)
    # 获取电影名的特征向量
    # pool_layer_flat, movie_title_embed_layer = get_movie_cnn_layer(movie_titles)
    movie_title_embed_layer = get_movie_w2v_layer(movie_titles)
    # 得到电影特征
    movie_combine_layer, movie_combine_layer_flat = get_movie_feature_layer(movie_id_embed_layer,
                                                                            movie_categories_embed_layer,
                                                                            movie_title_embed_layer)
    # 计算出评分，要注意两个不同的方案，inference的名字（name值）是不一样的，后面做推荐时要根据name取得tensor
    with tf.name_scope("inference"):
        # 将用户特征和电影特征作为输入，经过全连接，输出一个值的方案
        #         inference_layer = tf.concat([user_combine_layer_flat, movie_combine_layer_flat], 1)  #(?, 200)
        #         inference = tf.layers.dense(inference_layer, 1,
        #                                     kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
        #                                     kernel_regularizer=tf.nn.l2_loss, name="inference")

        # 简单的将用户特征和电影特征做矩阵乘法得到一个预测评分
        #        inference = tf.matmul(user_combine_layer_flat, tf.transpose(movie_combine_layer_flat)) 错误应该是下面一个用户与一个电影的评分
        inference = tf.reduce_sum(user_combine_layer_flat * movie_combine_layer_flat,
                                  axis=1)  # 按axis=1求和降维，得(?,),*表示对应元素相乘
        inference = tf.expand_dims(inference, axis=1)  # (batch,1)为了下面和target统一格式计算loss

    with tf.name_scope("loss"):
        # 将梯度在target network和learner间传递的功能在distributed tensorflow中默认已经实现好了
        # Between-graph的方式中，每个thread会拷贝一份Graph，计算后回传回主Graph。需要解决的主要是梯度累积的问题。
        # MSE损失，将计算值回归到评分
        cost = tf.losses.mean_squared_error(targets, inference)
        loss = tf.reduce_mean(cost)  # tf.reduce_mean 函数用于计算张量tensor沿着指定的数轴（tensor的某一维度）上的的平均值，主要用作降维或者计算tensor（图像）的平均值。

        # 优化损失
    #     train_op = tf.train.AdamOptimizer(lr).minimize(loss)  #cost
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(lr)  # 进行反向传播的训练方法
    gradients = optimizer.compute_gradients(loss)  # cost
    # minimize() = compute_gradients() + apply_gradients()拆分成计算梯度和应用梯度两个步骤
    # tf.train.Optimizer.compute_gradients(loss, var_list=None, gate_gradients=1) 计算“var_list”中变量的“loss”梯度
    # 返回一个元素为(loss对参数梯度，这个参数参数)对的列表[ (), () ],其中每个tuple对中梯度是该参数的梯度值
    # loss：包含要最小化的值的Tensor
    train_op = optimizer.apply_gradients(gradients, global_step=global_step)
    # tf.train.Optimizer.apply_gradients(grads_and_vars, global_step=None, name=None)进行更新网络，即应用梯度
    # minimize() 的第二部分，将上一步得到的字典，进行更新参数
    # grads_and_vars: 上一步得到的[(gradient, variable)]


# 取得batch------------------------------------------
def get_batches(Xs, ys, batch_size):
    for start in range(0, len(Xs), batch_size):
        end = min(start + batch_size, len(Xs))
        yield Xs[start:end], ys[start:end]  # yield 是一个类似 return 的关键字，迭代一次遇到yield时就返回yield后面(右边)的值。
        # 重点是：下一次迭代时，从上一次迭代遇到的yield后面的代码(下一行)开始执行。
        # 参见https://www.jianshu.com/p/d09778f4e055

# 训练网络-------------------------------------------

losses = {'train': [], 'test': []}

with tf.Session(graph=train_graph) as sess:
    # 搜集数据给tensorBoard用
    # 跟踪梯度值和稀疏度
    sess.run(tf.global_variables_initializer())
    grad_summaries = []
    for g, v in gradients:  # v对应的gradients
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name.replace(':', '_')), g)
            # tf.summary.histogram('summary_name', tensor)用来显示直方图信息
            # 将【计算图】中的【数据的分布/数据直方图】写入TensorFlow中的【日志文件】，以便为将来tensorboard的可视化做准备
            # 一般用来显示训练过程中变量的分布情况
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name.replace(':', '_')),
                                                 tf.nn.zero_fraction(g))
            # tf.summary.scalar(name, tensor, collections=None) 用来显示标量信息，一般在画loss,accuary时会用到这个函数
            # 将【计算图】中的【标量数据】写入TensorFlow中的【日志文件】，以便为将来tensorboard的可视化做准备
            # tf.nn.zero_fraction统计某个值的0的比例，这个tf.nn.zero_fraction计算出来的值越大，0的比例越高，稀疏度
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)
    # tf.summary.merge(inputs, collections=None, name=None)将上面几种类型的汇总再进行一次合并，具体合并哪些由inputs指定
    # merge_all将之前定义的所有summary整合在一起
    # [2]和TensorFlow中的其他操作类似，tf.summary.scalar、tf.summary.histogram、tf.summary.image函数也是一个op
    # 在定义的时候，也不会立即执行，需要通过sess.run来明确调用这些函数。因为，在一个程序中定义的写日志操作比较多
    # 如果一一调用，将会十分麻烦，所以Tensorflow提供了tf.summary.merge_all()函数将所有的summary整理在一起
    # 在TensorFlow程序执行的时候，只需要运行这一个操作就可以将代码中定义的所有【写日志操作】执行一次，从而将所有的日志写入【日志文件】。

    # 模型和summaries的输出目录
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    # os.path.curdir:当前目录
    # os.path.join(): 常用来链接路径
    # os.path.abspath 绝对路径
    # Python中有join()和os.path.join()两个函数
    # join()： 连接字符串数组。将字符串、元组、列表中的元素以指定的字符(分隔符)连接生成一个新的字符串
    # os.path.join()： 将多个路径组合后返回:os.path.join(path1[,path2[,……]])
    # 第一个以”/”开头的参数开始拼接，之前的参数全部丢弃
    # 以上一种情况为先。在上一种情况确保情况下，若出现”./”开头的参数，会从”./”开头的参数的上一个参数开始拼接
    #
    # os.path.join('aaaa','/bbbb','ccccc.txt')--> /bbbb\ccccc.txt只有一个以”/”开头的，参数从它开始往后拼接，之前的参数全部丢弃
    # os.path.join('/aaaa','/bbbb','/ccccc.txt')-->/ccccc.txt有多个以”/”开头的参数，从最后”/”开头的的开始往后拼接，之前的参数全部丢弃
    # os.path.join('aaaa','./bbb','ccccc.txt')-->aaaa\./bbb\ccccc.txt若出现”./”开头的参数，会从”./”开头的参数的上一个参数开始拼接,即只包含上一个参数加往后的
    print("Writing to {}\n".format(out_dir))

    # 关于损失和准确性的summaries，显示loss标量信息
    loss_summary = tf.summary.scalar("loss", loss)

    # 训练 Summaries
    train_summary_op = tf.summary.merge([loss_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
    # FileWriter类提供了一种用于在给定目录下创建事件文件的机制，并且将summary数据写入硬盘
    # sess.graph这个参数表示将前面定义的框架信息收集起来，放在train_summary_dir目录下

    # Inference summaries
    inference_summary_op = tf.summary.merge([loss_summary])
    inference_summary_dir = os.path.join(out_dir, "summaries", "inference")
    inference_summary_writer = tf.summary.FileWriter(inference_summary_dir, sess.graph)
    # tf.summary.FileWritter(path,sess.graph)指定一个文件用来保存图
    # 可以调用其add_summary（）方法将训练过程数据保存在filewriter指定的文件中

    sess.run(tf.global_variables_initializer())  # 参数的初始化

    saver = tf.train.Saver()  # 用于后面保存数据，创建一个saver对象
    for epoch_i in range(num_epochs):

        # 将数据集分成训练集和测试集，随机种子不固定
        train_X, test_X, train_y, test_y = train_test_split(features,
                                                            targets_values,
                                                            test_size=0.2,
                                                            random_state=0)

        train_batches = get_batches(train_X, train_y, batch_size)  # 在分好的训练集中再选batch个
        test_batches = get_batches(test_X, test_y, batch_size)

        # 训练的迭代，保存训练损失
        for batch_i in range(len(train_X) // batch_size):  # //取结果的最小整数
            x, y = next(train_batches)  # next() 返回迭代器的下一个项目，next(get_batches(train_X, train_y, batch_size))
            # 在这个for循环每次都进行上一个batch的后面一个batch

            categories = np.zeros([batch_size, 18])
            for i in range(batch_size):
                categories[i] = x.take(6, 1)[i]  # x取纵着的下标为6的全部数据

            # titles = np.zeros([batch_size, sentences_size])
            # for i in range(batch_size):
            #     titles[i] = x.take(5, 1)[i]

            feed = {
                uid: np.reshape(x.take(0, 1), [batch_size, 1]),
                user_gender: np.reshape(x.take(2, 1), [batch_size, 1]),
                user_age: np.reshape(x.take(3, 1), [batch_size, 1]),
                user_job: np.reshape(x.take(4, 1), [batch_size, 1]),
                movie_id: np.reshape(x.take(1, 1), [batch_size, 1]),
                movie_categories: categories,  # x.take(6,1)
                # movie_titles: titles,
                movie_titles: np.reshape(np.array(list(x.take(5, 1))), [batch_size, embed_dim]),  # x.take(5,1)
                targets: np.reshape(y, [batch_size, 1]),
                dropout_keep_prob: dropout_keep,  # dropout_keep
                lr: learning_rate
            }

            step, train_loss, summaries, _ = sess.run([global_step, loss, train_summary_op, train_op], feed)  # cost
            losses['train'].append(train_loss)
            train_summary_writer.add_summary(summaries, step)  #

            # Show every <show_every_n_batches> batches
            if (epoch_i * (len(train_X) // batch_size) + batch_i) % show_every_n_batches == 0:
                time_str = datetime.datetime.now().isoformat()
                print('{}: Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    time_str,
                    epoch_i,
                    batch_i,
                    (len(train_X) // batch_size),
                    train_loss))

        # 使用测试数据的迭代
        for batch_i in range(len(test_X) // batch_size):
            x, y = next(test_batches)

            categories = np.zeros([batch_size, 18])
            for i in range(batch_size):
                categories[i] = x.take(6, 1)[i]

            # titles = np.zeros([batch_size, sentences_size])
            # for i in range(batch_size):
            #     titles[i] = x.take(5, 1)[i]

            feed = {
                uid: np.reshape(x.take(0, 1), [batch_size, 1]),
                user_gender: np.reshape(x.take(2, 1), [batch_size, 1]),
                user_age: np.reshape(x.take(3, 1), [batch_size, 1]),
                user_job: np.reshape(x.take(4, 1), [batch_size, 1]),
                movie_id: np.reshape(x.take(1, 1), [batch_size, 1]),
                movie_categories: categories,  # x.take(6,1)
                # movie_titles: titles,  # x.take(5,1)
                movie_titles: np.reshape(np.array(list(x.take(5, 1))), [batch_size, embed_dim]),  # x.take(5,1)
                targets: np.reshape(y, [batch_size, 1]),
                dropout_keep_prob: 1,
                lr: learning_rate
            }

            step, test_loss, summaries = sess.run([global_step, loss, inference_summary_op], feed)  # cost

            # 保存测试损失
            losses['test'].append(test_loss)
            inference_summary_writer.add_summary(summaries, step)  # 调用add_summary（）方法将训练过程数据保存在filewriter指定的文件中

            time_str = datetime.datetime.now().isoformat()  # datetime.datetime：表示日期(year, month, day)时间(hour, minute, second, microsecond)
            # date.isoformat()：返回格式如'YYYY-MM-DD’的字符串
            if (epoch_i * (len(test_X) // batch_size) + batch_i) % show_every_n_batches == 0:
                print('{}: Epoch {:>3} Batch {:>4}/{}   test_loss = {:.3f}'.format(
                    time_str,
                    epoch_i,
                    batch_i,
                    (len(test_X) // batch_size),
                    test_loss))

    # Save Model保存模型
    saver.save(sess, save_dir)  # sess是一个session对象，这里save_dir是给模型起的名字
    # 可以加参数 global_step=epoch_i，表示进行epoch_i次保存一次模型
    print('Model Trained and Saved')

save_params((save_dir))  # 保存参数 保存save_dir 在生成预测时使用

# # 显示训练Loss
# plt.plot(losses['train'], label='Training loss')  # plt.plot(x,y)
# plt.legend()  # 用于显示图例
# _ = plt.ylim()  # 设置y轴刻度的取值范围,不指定则自适应横纵坐标轴
#
# # 显示测试loss
# plt.plot(losses['test'], label='Test loss')
# plt.legend()
# _ = plt.ylim()
#
# plt.show()

print 'average test loss', sum(losses['test']) / len(losses['test'])*1.0