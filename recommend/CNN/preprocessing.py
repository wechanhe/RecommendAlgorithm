#!/usr/bin/python 
# -*- coding: UTF-8 -*-

import codecs
import numpy as np
import pandas as pd
import os  # os 模块提供了非常丰富的方法用来处理文件和目录
import pickle  # 提供保存数据在本地的方法
import re  # 正则表达式
import zipfile  # 用来做zip格式编码的压缩和解压缩的
import hashlib  # 用来进行hash 或者md5 加密
from gensim.models import Word2Vec

w2v_path = 'word2vec/w2v.model'
# 嵌入矩阵的维度：一个单词或其他变量的特征表示
embed_dim = 32
batch_size = 512

def _unzip(save_path, _, database_name, data_path):
    """
    Unzip wrapper with the same interface as _ungzip使用与_ungzip相同的接口解压缩包装器
    :param save_path: gzip文件的路径
    :param database_name:数据库的名称
    :param data_path: 提取路径
    :param _: HACK - Used to have to same interface as _ungzip 用于与_ungzip具有相同的接口？？解压后的文件路径
    """
    print('Extracting {}...'.format(database_name))  # .format通过 {} 来代替字符串database_name
    with zipfile.ZipFile(save_path) as zf:  # ZipFile是zipfile包中的一个类，用来创建和读取zip文件
        zf.extractall(data_path)  # 类函数zipfile.extractall([path[, member[, password]]])
        # path解压缩目录，没什么可说的
        # member需要解压缩的文件名列表
        # password当zip文件有密码时需要该选项


def download_extract(database_name, data_path):
    """
    下载并提取数据库
    :param database_name: Database name
    data_path 这里为./表示当前目录
    save_path 下载后数据的保存路径即压缩文件的路径
    extract_path 解压后的文件路径
    """
    DATASET_ML1M = 'ml-1m'

    if database_name == DATASET_ML1M:
        url = 'http://files.grouplens.org/datasets/movielens/ml-1m.zip'
        hash_code = 'c4d9eecfca2ab87c1945afe126590906'
        extract_path = os.path.join(data_path, 'ml-1m')  # os.path.join将多个路径组合后返回，提取数据的路径
        save_path = os.path.join(data_path, 'ml-1m.zip')  # 要保存的路径
        extract_fn = _unzip

    if os.path.exists(extract_path):  # 指定路径（文件或者目录）是否存在
        print('Found {} Data'.format(database_name))
        return

    if not os.path.exists(data_path):  # 指定路径（文件或者目录）不存在，则递归创建目录data_path
        os.makedirs(data_path)

    if not os.path.exists(save_path):  # 指定路径（文件或者目录）不存在，则递归创建目录save_path
        with DLProgress(unit='B', unit_scale=True, miniters=1,
                        desc='Downloading {}'.format(database_name)) as pbar:  # 调用类，进度条显示相关，tqdm相关参数设置
            urlretrieve(url, save_path,
                        pbar.hook)  # urlretrieve()方法直接将远程数据下载到本地 rlretrieve(url, filename=None, reporthook=None, data=None)
            # filename指定了保存本地路径
            # reporthook是回调函数，当连接上服务器、以及相应的数据块传输完毕时会触发该回调，可利用回调函数显示当前下载进度。

    assert hashlib.md5(open(save_path, 'rb').read()).hexdigest() == hash_code, \
        '{} file is corrupted.  Remove the file and try again.'.format(save_path)
    # assert expression [, arguments]表示断言测试，如expression异常，则输出后面字符串信息
    # 能指出数据是否被篡改过，就是因为摘要函数是一个单向函数，计算f(data)很容易，但通过digest反推data却非常困难。而且，对原始数据做一个bit的修改，都会导致计算出的摘要完全不同。
    # 摘要算法应用：用户存储用户名密码，但在数据库不能以明文存储，而是用md5，当一个用户输入密码时，进行md5匹配，如果相同则可以登录
    # hashlib提供了常见的摘要算法，如MD5，SHA1等等。摘要算法又称哈希算法、散列算法。它通过一个函数，把任意长度的数据转换为一个长度固定的数据串（通常用16进制的字符串表示）。
    # hexdigest为md5后的结果

    os.makedirs(extract_path)
    try:
        extract_fn(save_path, extract_path, database_name, data_path)  # 解压
    except Exception as err:
        shutil.rmtree(extract_path)  # Remove extraction folder if there is an error表示递归删除文件夹下的所有子文件夹和子文件
        raise err  # 抛出异常

    print('Done.')
    # Remove compressed data


data_dir = '../../data/ml-1m/ml-1m'
# download_extract('ml-1m', data_dir)

# 实现数据预处理
def load_data():
    """
    Load Dataset from File
    """
    # 读取User数据-------------------------------------------------------------
    users_title = ['UserID', 'Gender', 'Age', 'JobID', 'Zip-code']

    users = pd.read_table(data_dir + '/users.dat', sep='::', header=None, names=users_title, engine='python')

    # 分隔符参数：sep
    # 是否读取文本数据的header，headers = None表示使用默认分配的列名，一般用在读取没有header的数据文件
    # 为文本的数据加上自定义列名： names
    # pandas.read_csv()从文件，URL，文件型对象中加载带分隔符的数据。默认分隔符为','
    # pandas.read_table()从文件，URL，文件型对象中加载带分隔符的数据。默认分隔符为'\t'

    users = users.filter(regex='UserID|Gender|Age|JobID')

    # 这里使用正则式进行过滤
    users_orig = users.values  # dataframe.values以数组的形式返回DataFrame的元素

    # 改变User数据中性别和年龄
    gender_map = {'F': 0, 'M': 1}
    users['Gender'] = users['Gender'].map(
        gender_map)  # map()函数可以用于Series对象或DataFrame对象的一列，接收函数或字典对象作为参数，返回经过函数或字典映射处理后的值。


    # age_map = {val: ii for ii, val in enumerate(set(users['Age']))}
    age_map = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}

    # enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列
    # 同时列出数据和数据下标，一般用在 for 循环当中
    # set() 函数创建一个无序不重复元素集
    users['Age'] = users['Age'].map(age_map)  # map接收的参数是函数


    # 读取Movie数据集---------------------------------------------------------
    movies_title = ['MovieID', 'Title', 'Genres']
    movies = pd.read_table(data_dir + '/movies.dat', sep='::', header=None, names=movies_title, engine='python')

    movies_orig = movies.values
    # 将Title中的年份去掉
    pattern = re.compile(
        r'^(.*)\((\d+)\)$')  # re.compile(strPattern[, flag]):把正则表达式的模式和标识转化成正则表达式对象。供match()和search()这两个函数使用
    # 第二个参数flag是匹配模式，取值可以使用按位或运算符'|'表示同时生效
    # r表示后面是一个正则表达式''
    # ^匹配开头,$匹配结尾,(.*)中的()表示匹配其中的任意正则表达式,.匹配任何字符,*代表可以重复0次或多次
    # \(和\)：表示对括号的转义，匹配文本中真正的括号
    # (\d+)表示匹配()内的任意字符,\d表示任何数字,+代表数字重复一次或者多次

    title_map = {val: pattern.match(val).group(1) for ii, val in enumerate(set(movies['Title']))}
    # 这里的ii是索引值，val是真正的列表中Title元素
    # pattern.match(val)使用Pattern匹配文本val，获得匹配结果，无法匹配时将返回None
    # group获得一个或多个分组截获的字符串；指定多个参数时将以元组形式返回，分组是按照()匹配顺序进行
    # 这里group(1)相当于只返回第一组，分组标号从1开始。不填则为返回全部结果
    # 这里即完成了将电影名称的时间去掉
    movies['Title'] = movies['Title'].map(title_map)  # title列的电影名转化为去掉名称后的电影名
    movies_name = movies['Title'].values

    # 电影title进行word2vec转换
    if not os.path.exists(w2v_path):
        print 'training word2vec model...'
        sents = []
        for name in movies_name:
            tmp = []
            for word in name.strip().split(' '):
                tmp.append(word)
            sents.append(tmp)
        model = Word2Vec(sentences=sents, size=embed_dim)
        model.save(w2v_path)
    model = Word2Vec.load(w2v_path)
    for i in range(movies['Title'].values.size):
        vec = np.zeros([1, embed_dim], float)
        for word in movies['Title'].values[i].split(' '):
            try:
                vec += model.wv[word]
            except:
                pass
        movies['Title'].values[i] = vec.reshape([-1, embed_dim])


    # 电影Title转数字字典
    title_set = set()  # set() 函数创建一个无序不重复元素集,返回一个可迭代对象
    # for val in movies['Title'].str.split():  # 对于电影名称按空格分，val为整个电影列表中全部单词
    #     # 注意string.split() 不带参数时，和 string.split(' ') 是有很大区别的
    #     # 不带参数的不会截取出空格，而带参数的只按一个空格去切分，多出来的空格会被截取出来
    #     # 参见https://code.ziqiangxuetang.com/python/att-string-split.html
    #     title_set.update(val)  # 添加新元素到集合当中，即完成出现电影中的新单词时，存下来
    #
    # title_set.add('<PAD>')  # 这里不是numpy.pad函数，只是一个填充表示，也为<PAD>进行编码
    # title2int = {val: ii for ii, val in enumerate(title_set)}  # 为全部单词进行像字典一样进行标注'描述电影的word：数字'格式,即数字字典
    # 将电影Title转成等长数字列表，长度是15
    title_count = 15
    # title_map = {val: [title2int[row] for row in val.split()] for ii, val in enumerate(set(movies['Title']))}
    # # for ii,val in enumerate(set(movies['Title']))得到ii索引值和其对应的不重复的一个电影字符串val(去掉月份的)
    # # val.split()得到全部被空格分开的电影名称字符串列表，row遍历电影集中一个电影的全部单词
    # # title_map得到的是字典，格式为'一个电影字符串：[描述这个电影的全部单词构成的一个对应的数值列表]'
    # for key in title_map:
    #     for cnt in range(title_count - len(title_map[key])):
    #         title_map[key].insert(len(title_map[key]) + cnt,
    #                               title2int['<PAD>'])  # insert(index, object) 在指定位置index前插入元素object
    #         # index                    ,object  电影key长度少于15就加填充符
    # movies['Title'] = movies['Title'].map(title_map)  # title字段的去掉名称后的电影名转化为对应的数字列表
    # 如电影集中的一行数据如下movieid，title，genre
    # array([1,
    # list([3001, 5100, 275, 275, 275, 275, 275, 275, 275, 275, 275, 275, 275, 275, 275]),
    # list([3, 6, 2, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17])], dtype=object)

    # 电影类型转数字字典-----
    genres_set = set()
    for val in movies['Genres'].str.split('|'):  # 对于一个电影的题材进行字符串转化，并用|分割遍历
        genres_set.update(val)  # set.update()方法用于修改当前集合，可以添加新的元素或集合到当前集合中，如果添加的元素在集合中已存在，则该元素只会出现一次，重复的会忽略。
        # 将描述不同题材的电影存入set
    genres_set.add('<PAD>')  # 集合add方法：是把要传入的元素做为一个整个添加到集合中，为<PAD>进行编码
    genres2int = {val: ii for ii, val in enumerate(genres_set)}  # 将类型转化为'字符串：数字'格式,即数字字典，同上面电影名称，一个word对应一个数字
    # 将电影类型转成等长数字列表，长度是18
    genres_map = {val: [genres2int[row] for row in val.split('|')] for ii, val in enumerate(set(movies['Genres']))}
    for key in genres_map:
        for cnt in range(max(genres2int.values()) - len(genres_map[key])):
            genres_map[key].insert(len(genres_map[key]) + cnt, genres2int['<PAD>'])
    movies['Genres'] = movies['Genres'].map(genres_map)

    # 读取评分数据集--------------------------------------------------------
    ratings_title = ['UserID', 'MovieID', 'ratings', 'timestamps']

    ratings = pd.read_table(data_dir + '/ratings.dat', sep='::', header=None, names=ratings_title, engine='python')

    ratings = ratings.filter(regex='UserID|MovieID|ratings')

    # 合并三个表
    data = pd.merge(pd.merge(ratings, users), movies)
    # 通过一个或多个键将两个数据集的行连接起来，类似于 SQL 中的 JOIN
    # 合并左dataframe和右datafram，默认为取交集，取交集作为索引键

    # 将数据分成X和y两张表
    target_fields = ['ratings']
    features_pd, targets_pd = data.drop(target_fields, axis=1), data[target_fields]
    # features_pd只删除rating作为x表；targets_pd只有rating作为y
    # 删除表中的某一行或者某一列使用drop，不改变原有的df中的数据，而是返回另一个dataframe来存放删除后的数据。

    features = features_pd.values
    targets_values = targets_pd.values

    print np.reshape(features.take(5, 1), [-1, 1])
    return movies_name, title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig
    # title_count电影名长度15
    # title_set {索引：去掉年份且不重复的电影名}
    # genres2int {题材字符串列表：数字}
    # features 去掉评分ratings列的三表合并信息，作为输入x。则列信息：userid,gender,age,occupation,movieid,title,genres
    # targets_values 评分，学习目标y,三表合并后的对应ratings
    # 返回处理后的ratings，users，movies表，pandas对象
    # 返回三表的合并表data
    # moives表中的原始数据值：movies_orig
    # users表中的原始数据值：users_orig

def train_w2v():
    movies['Title']
    # model = Word2Vec.train(sentences=)

if __name__ == '__main__':
    # ---------------------------------------------------------------------------
    # 加载数据并保存到本地
    movies_name, title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = load_data()

    pickle.dump((movies_name, title_count, title_set, genres2int,
                 features, targets_values, ratings,
                 users, movies, data, movies_orig, users_orig), open('feature/feature.p', 'wb'))
    # 保存数据到本地，便于后续的提取，以防后面用到这些数据还要进行一边上次的数据预处理过程
    # 从本地读取数据，下面这些代码可以用在核心代码中第一步数据读取
    # movies_name, title_count, title_set, genres2int, features, targets_values, ratings, users, movies, data, movies_orig, users_orig = pickle.load(
    #     open('./feature/feature.p', mode='rb'))
    # print features[0:10]

