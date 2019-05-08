#!/usr/bin/python 
# -*- coding: UTF-8 -*-

import tensorflow as tf
import pickle
import numpy as np
import random


load_dir = 'save/'

# -----------这里要从前一个文件copy过来参数，因为这部分不在graph和session中的模型参数中-------
# 从本地读取数据
movies_name, title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = pickle.load(
    open('feature/feature.p', mode='rb'))

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

# 超参---
# Number of Epochs
num_epochs = 5
# Batch Size
batch_size = 256

dropout_keep = 0.5
# Learning Rate
learning_rate = 0.0001
# Show stats for every n number of batches
show_every_n_batches = 20


# --------------------------------------------------------------------------------------------

# 获取 Tensors
# 使用函数 get_tensor_by_name()从 loaded_graph 中获取tensors，后面的推荐功能要用到
def get_tensors(loaded_graph):
    uid = loaded_graph.get_tensor_by_name("uid:0")  # 想要恢复这个网络，我们不仅需要恢复图（graph）和权重，而且也需要准备一个新的feed_dict
    # 将新的训练数据喂给网络。我们可以通过使用graph.get_tensor_by_name()方法来获得
    # 已经保存的操作（operations）和placeholder variables。
    user_gender = loaded_graph.get_tensor_by_name("user_gender:0")
    user_age = loaded_graph.get_tensor_by_name("user_age:0")
    user_job = loaded_graph.get_tensor_by_name("user_job:0")
    movie_id = loaded_graph.get_tensor_by_name("movie_id:0")
    movie_categories = loaded_graph.get_tensor_by_name("movie_categories:0")
    movie_titles = loaded_graph.get_tensor_by_name("movie_titles:0")
    targets = loaded_graph.get_tensor_by_name("targets:0")
    dropout_keep_prob = loaded_graph.get_tensor_by_name("dropout_keep_prob:0")
    lr = loaded_graph.get_tensor_by_name("LearningRate:0")
    # 两种不同计算预测评分的方案使用不同的name获取tensor inference
    #     inference = loaded_graph.get_tensor_by_name("inference/inference/BiasAdd:0")
    inference = loaded_graph.get_tensor_by_name(
        "inference/ExpandDims:0")  # 之前是MatMul:0 因为inference代码修改了 这里也要修改 感谢网友 @清歌 指出问题
    movie_combine_layer_flat = loaded_graph.get_tensor_by_name("movie_fc/Reshape:0")
    user_combine_layer_flat = loaded_graph.get_tensor_by_name("user_fc/Reshape:0")
    return uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, lr, dropout_keep_prob, inference, movie_combine_layer_flat, user_combine_layer_flat


# 指定用户和电影进行评分
# 这部分就是对网络做正向传播，计算得到预测的评分
def rating_movie(user_id_val, movie_id_val):
    loaded_graph = tf.Graph()  #
    with tf.Session(graph=loaded_graph) as sess:  #
        # Load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')  # 由于已经将模型保存在了 .meta 文件中，因此可使用tf.train.import()函数来重新创建网络
        # 使用别人已经训练好的模型来fine-tuning的第一步：此为创建网络Create the network
        loader.restore(sess, load_dir)  # 第二步：加载参数Load the parameters，调用restore函数来恢复网络的参数

        # Get Tensors from loaded model
        uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, lr, dropout_keep_prob, inference, _, __ = get_tensors(
            loaded_graph)  # loaded_graph

        categories = np.zeros([1, 18])
        categories[0] = movies.values[movieid2idx[movie_id_val]][2]

        titles = np.zeros([1, sentences_size])
        titles[0] = movies.values[movieid2idx[movie_id_val]][1]

        feed = {
            uid: np.reshape(users.values[user_id_val - 1][0], [1, 1]),
            user_gender: np.reshape(users.values[user_id_val - 1][1], [1, 1]),
            user_age: np.reshape(users.values[user_id_val - 1][2], [1, 1]),
            user_job: np.reshape(users.values[user_id_val - 1][3], [1, 1]),
            movie_id: np.reshape(movies.values[movieid2idx[movie_id_val]][0], [1, 1]),
            movie_categories: categories,  # x.take(6,1)
            movie_titles: titles,  # x.take(5,1)
            dropout_keep_prob: 1
        }

        # Get Prediction
        inference_val = sess.run([inference], feed)

        return (inference_val)


# 生成Movie特征矩阵
# 将训练好的电影特征组合成电影特征矩阵并保存到本地

loaded_graph = tf.Graph()  # 1、新建一个graph
movie_matrics = []
with tf.Session(graph=loaded_graph) as sess:  # 2、在session中引入这个图
    # Load saved model导入已经训练好的模型
    loader = tf.train.import_meta_graph(load_dir + '.meta')  # 由于模型已经保存在meta文件中，这里import该文件来创建网络
    loader.restore(sess, load_dir)  # 加载参数：通过调用restore函数来恢复网络的参数

    # Get Tensors from loaded model 要恢复这个网络，不仅需要恢复图（graph）和权重，也需要准备一个新的feed_dict，将新的训练数据喂给网络。
    # 我们可以通过使用graph.get_tensor_by_name()方法来获得已经保存的操作（operations）和placeholder variables。
    # 为后续feed做准备
    uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, lr, dropout_keep_prob, _, movie_combine_layer_flat, __ = get_tensors(
        loaded_graph)  # loaded_graph

    for item in movies.values:
        # item为
        # array([[1,
        #        list([106, 2958, 543, 543, 543, 543, 543, 543, 543, 543, 543, 543, 543, 543, 543]),
        #        list([15, 5, 9, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4])],......])

        categories = np.zeros([1, 18])  # 得到array([[ 18个0 ]])
        categories[0] = item.take(2)  # categories[0]= [ 18个0 ]，注意这里取得是array的第0个元素，而array中的一个元素是一个列表，观察括号个数
        # item为movies.values中的一个[1,list[电影名],list[电影类别]]
        # item.take(2)表示取电影类别这个list[电影类别]
        titles = np.zeros([1, sentences_size])
        titles[0] = item.take(1)  # item.take(1)表示取电影名称这个list[电影名称]

        feed = {
            movie_id: np.reshape(item.take(0), [1, 1]),
            movie_categories: categories,  # 前面训练模型的代码中是训练数据取x.take(6,1)
            movie_titles: titles,  # x.take(5,1)
            dropout_keep_prob: 1
        }

        movie_combine_layer_flat_val = sess.run([movie_combine_layer_flat],
                                                feed)  # 执行整个movie结构中的最后一个功能,完成全部的数据流动,得到输出的电影特征
        movie_matrics.append(movie_combine_layer_flat_val)  # 为每个movie生成一个电影特征矩阵，存储到movie_matrics列表中

# pickle.dump((np.array(movie_matrics).reshape(-1, 200)), open('matrix/movie_matrics.p', 'wb'))
movie_matrics = pickle.load(open('matrix/movie_matrics.p', mode='rb'))  # 将所有电影特征存到movie_matrics.p文件里

# 生成User特征矩阵
# 将训练好的用户特征组合成用户特征矩阵并保存到本地
loaded_graph = tf.Graph()  #
users_matrics = []
with tf.Session(graph=loaded_graph) as sess:  #
    # Load saved model
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    # Get Tensors from loaded model
    uid, user_gender, user_age, user_job, movie_id, movie_categories, movie_titles, targets, lr, dropout_keep_prob, _, __, user_combine_layer_flat = get_tensors(
        loaded_graph)  # loaded_graph

    for item in users.values:
        feed = {
            uid: np.reshape(item.take(0), [1, 1]),
            user_gender: np.reshape(item.take(1), [1, 1]),
            user_age: np.reshape(item.take(2), [1, 1]),
            user_job: np.reshape(item.take(3), [1, 1]),
            dropout_keep_prob: 1}

        user_combine_layer_flat_val = sess.run([user_combine_layer_flat], feed)
        users_matrics.append(user_combine_layer_flat_val)

# pickle.dump((np.array(users_matrics).reshape(-1, 200)), open('matrix/users_matrics.p', 'wb'))
users_matrics = pickle.load(open('matrix/users_matrics.p', mode='rb'))


# 开始推荐电影
# 使用生产的用户特征矩阵和电影特征矩阵做电影推荐，这里有三种方法，都可以在命令行进行调用来推荐

# 1、推荐同类型的电影
# 思路是计算当前看的电影特征向量与整个电影特征矩阵的余弦相似度，取相似度最大的top_k个
# 这里加了些随机选择在里面，保证每次的推荐稍稍有些不同。
def recommend_same_type_movie(movie_id_val, top_k=20):
    loaded_graph = tf.Graph()  #
    with tf.Session(graph=loaded_graph) as sess:  #
        # Load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        norm_movie_matrics = tf.sqrt(tf.reduce_sum(tf.square(movie_matrics), 1, keep_dims=True))
        # movie_matrics显示为 (3883, 200)
        # array([[-0.9784413 ,  0.97033578, -0.99996817, ..., -0.94367135,0.938721  ,  0.94092846],...])

        # tf.square()是对a里的每一个元素求平方i=(x,y)
        # tf.reduce_sum,注意参数表示在维度1(列)上进行求和,且维度不变 x^2+y^2
        # tf.sqrt计算x元素的平方根
        # 这里完成向量单位化
        # (3883, 1)
        normalized_movie_matrics = movie_matrics / norm_movie_matrics  # Python中的 // 与 / 的区别
        # / 表示浮点数除法,返回浮点结果
        # //表示整数除法,返回不大于结果的一个最大的整数
        # 单位化后的i=(  x/(x^2+y^2),y/(x^2+y^2)  )

        # 推荐同类型的电影
        probs_embeddings = (movie_matrics[movieid2idx[movie_id_val]]).reshape([1, 200])  # 用户输入已看过的电影，进行movieid2idx数字转化
        # movie_matrics[转化后的标记数值]得到对应的电影特征向量
        probs_similarity = tf.matmul(probs_embeddings,
                                     tf.transpose(normalized_movie_matrics))  # 矩阵乘法(x1,x2)和(y1,y2)可以得到x1y1+x2y2
        # 即得到输入的电影与各个电影的余弦相似性的值
        # (1,200)×(200,3883)
        sim = (probs_similarity.eval())  # 转化为字符串
        # sim [[ 13.49374485  13.48943233  13.51107979 ...,  13.50281906  13.49236774  13.49707603]]

        print("您看的电影是：{}".format(movies_orig[movieid2idx[movie_id_val]]))  # movies_orig原始未处理的电影数据，为输出用户可读
        print("以下是给您的推荐：")
        p = np.squeeze(sim)  # np.squeeze将表示向量的数组转换为秩为1的数组
        # p [ 13.49374485  13.48943233  13.51107979 ...,  13.50281906  13.49236774  13.49707603]

        p[np.argsort(p)[:-top_k]] = 0  # numpy.argsort()
        # x=np.array([1,4,3,-1,6,9])
        # 函数含义：首先将p中的元素从小到大排列后，得到[-1,1,3,4,6,9]
        #          按照所得的排好序的对应找其在原x中的索引值，如-1由x[3]得到；1由x[0]得到，所以索引值为[3,0,2,1,4,5]
        #          所以这个即为输出
        # np.argsort()[:-top_k]表示将np.argsort()得到的结果去掉后面20个后的前面所有值为0，因为我们只考虑最相似的20个
        # 这些值不为0，以便做后面的处理

        p = p / np.sum(p)  # sum函数对某一维度求和，这里表示全部元素求和,这里将p的值限制在0~1
        results = set()
        while len(results) != 5:  # 推荐5个
            c = np.random.choice(3883, 1, p=p)[0]  # 参数意思分别 是从a 中以概率P，随机选择3个,
            # p没有指定的时候表示同等概率会被取出，p指定时表示每个数会被取出的概率
            # replace代表的意思是抽样之后不放回，选出的三个数都不一样
            # a1 = np.random.choice(a=5, size=3, replace=False, p=None)
            results.add(c)  # results本身为set（可以完成剔除掉相同的推荐，虽然前面np.random.choice是不放回）
        for val in (results):
            print(val)  # 由于前面已经转换为字符串eval，所以可以直接输出
            print(movies_orig[val])
        return results


# recommend_same_type_movie(1401, 20)输出
# 您看的电影是：[1401 'Ghosts of Mississippi (1996)' 'Drama']
# 以下是给您的推荐：
# 3385
# [3454 'Whatever It Takes (2000)' 'Comedy|Romance']
# 707
# [716 'Switchblade Sisters (1975)' 'Crime']
# 2351
# [2420 'Karate Kid, The (1984)' 'Drama']
# 2189
# [2258 'Master Ninja I (1984)' 'Action']
# 2191
# [2260 'Wisdom (1986)' 'Action|Crime']


# 2、推荐您喜欢的电影
# 思路是使用用户特征向量与电影特征矩阵计算所有电影的评分，取评分最高的top_k个，同样加了些随机选择部分。
def recommend_your_favorite_movie(user_id_val, top_k=10):
    loaded_graph = tf.Graph()  #
    with tf.Session(graph=loaded_graph) as sess:  #
        # Load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        # 推荐您喜欢的电影
        probs_embeddings = (users_matrics[user_id_val - 1]).reshape([1, 200])  # ！！！这里变成用户特征，且前面没有余弦相似性的计算

        probs_similarity = tf.matmul(probs_embeddings, tf.transpose(movie_matrics))  # 这里计算后的结果就是预测分数，相当于模型中计算inference
        sim = (probs_similarity.eval())

        print("以下是给您的推荐：")
        p = np.squeeze(sim)
        p[np.argsort(p)[:-top_k]] = 0
        p = p / np.sum(p)
        results = set()
        while len(results) != 5:
            c = np.random.choice(3883, 1, p=p)[0]
            results.add(c)
        for val in (results):
            print(val)
            print(movies_orig[val])

        return results


recommend_your_favorite_movie(234, 10)
# 以下是给您的推荐：
# 1642
# [1688 'Anastasia (1997)' "Animation|Children's|Musical"]
# 994
# [1007 'Apple Dumpling Gang, The (1975)' "Children's|Comedy|Western"]
# 667
# [673 'Space Jam (1996)' "Adventure|Animation|Children's|Comedy|Fantasy"]
# 1812
# [1881 'Quest for Camelot (1998)' "Adventure|Animation|Children's|Fantasy"]
# 1898
# [1967 'Labyrinth (1986)' "Adventure|Children's|Fantasy"]


# 3、看过这个电影的人还看了（喜欢）哪些电影
# 首先选出喜欢某个电影的top_k个人，得到这几个人的用户特征向量。
# 然后计算这几个人对所有电影的评分
# 选择每个人评分最高的电影作为推荐
# 同样加入了随机选择
def recommend_other_favorite_movie(movie_id_val, top_k=20):
    loaded_graph = tf.Graph()  #
    with tf.Session(graph=loaded_graph) as sess:  #
        # Load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        probs_movie_embeddings = (movie_matrics[movieid2idx[movie_id_val]]).reshape([1, 200])  # 根据输入的电影得到这个电影的特征向量
        probs_user_favorite_similarity = tf.matmul(probs_movie_embeddings, tf.transpose(users_matrics))
        favorite_user_id = np.argsort(probs_user_favorite_similarity.eval())[0][-top_k:]  # 选出喜欢某个电影的top_k个人
        #     print(normalized_users_matrics.eval().shape)
        #     print(probs_user_favorite_similarity.eval()[0][favorite_user_id])
        #     print(favorite_user_id.shape)

        print("您看的电影是：{}".format(movies_orig[movieid2idx[movie_id_val]]))

        print("喜欢看这个电影的人是：{}".format(users_orig[favorite_user_id - 1]))
        probs_users_embeddings = (users_matrics[favorite_user_id - 1]).reshape([-1, 200])  # 计算这几个人的特征
        probs_similarity = tf.matmul(probs_users_embeddings, tf.transpose(movie_matrics))  # 计算这几个人对所有电影的评分
        sim = (probs_similarity.eval())
        #     results = (-sim[0]).argsort()[0:top_k]
        #     print(results)

        #     print(sim.shape)
        #     print(np.argmax(sim, 1))
        p = np.argmax(sim, 1)
        print("喜欢看这个电影的人还喜欢看：")

        results = set()
        while len(results) != 5:
            c = p[random.randrange(top_k)]
            results.add(c)
        for val in (results):
            print(val)
            print(movies_orig[val])

        return results

# recommend_other_favorite_movie(1401, 20)
# 您看的电影是：[1401 'Ghosts of Mississippi (1996)' 'Drama']
# 喜欢看这个电影的人是：[[5782 'F' 35 0]
# [5767 'M' 25 2]
# [3936 'F' 35 12]
# [3595 'M' 25 0]
# [1696 'M' 35 7]
# [2728 'M' 35 12]
# [763 'M' 18 10]
# [4404 'M' 25 1]
# [3901 'M' 18 14]
# [371 'M' 18 4]
# [1855 'M' 18 4]
# [2338 'M' 45 17]
# [450 'M' 45 1]
# [1130 'M' 18 7]
# [3035 'F' 25 7]
# [100 'M' 35 17]
# [567 'M' 35 20]
# [5861 'F' 50 1]
# [4800 'M' 18 4]
# [3281 'M' 25 17]]
# 喜欢看这个电影的人还喜欢看：
# 1779
# [1848 'Borrowers, The (1997)' "Adventure|Children's|Comedy|Fantasy"]
# 1244
# [1264 'Diva (1981)' 'Action|Drama|Mystery|Romance|Thriller']
# 1812
# [1881 'Quest for Camelot (1998)' "Adventure|Animation|Children's|Fantasy"]
# 1742
# [1805 'Wild Things (1998)' 'Crime|Drama|Mystery|Thriller']
# 2535
# [2604 'Let it Come Down: The Life of Paul Bowles (1998)' 'Documentary']

#######以上就是实现的常用的推荐功能，将网络模型作为回归问题进行训练，得到训练好的用户特征矩阵和电影特征矩阵进行推荐。

