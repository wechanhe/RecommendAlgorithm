#!/usr/bin/python 
# -*- coding: UTF-8 -*-

import os,sys,time
import numpy as np
import tensorflow as tf

import input
from model import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('embedding_size', 16, 'the size for embedding user and item.')
tf.app.flags.DEFINE_integer('epochs', 20, 'the number of epochs.')
tf.app.flags.DEFINE_string('optim', 'SGD', 'the optimization method.')
tf.app.flags.DEFINE_string('initializer', 'Xavier', 'the initializer method.')
tf.app.flags.DEFINE_string('activation', 'ReLU', 'the activation function.')
tf.app.flags.DEFINE_string('model_dir', 'model/', 'the dir for saving model.')
tf.app.flags.DEFINE_float('regularizer', 0.0, 'the regularizer rate.')
tf.app.flags.DEFINE_float('lr', 0.001, 'learning rate.')
tf.app.flags.DEFINE_float('dropout', 0.0, 'dropout rate.')

def train(train_data, test_data, user_size, item_size):
    with tf.Session() as sess:

        model = aSDAE(embed_size=FLAGS.embedding_size, user_size=user_size, item_size=item_size,
                      lr=FLAGS.lr, optim=FLAGS.optim, initializer=FLAGS.initializer,
                      activation_func=FLAGS.activation, regularizer_rate=FLAGS.regularizer,
                      dropout=FLAGS.dropout, is_training=True)

        model.build()

        ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
        if ckpt:
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print("Creating model with fresh parameters.")
            sess.run(tf.global_variables_initializer())

        count = 0
        for epoch in range(FLAGS.epochs):
            sess.run(model.iterator.make_initializer(train_data))
            model.is_training = True
            model.get_data()
            start_time = time.time()

            try:
                while True:
                    model.step(sess, count)
                    count += 1
            except tf.errors.OutOfRangeError:
                print("Epoch %d training " % epoch + "Took: " + time.strftime("%H: %M: %S",
                                                                              time.gmtime(time.time() - start_time)))



            # sess.run(model.iterator.make_initializer(test_data))
            # model.is_training = False
            # model.get_data()
            # start_time = time.time()
            # HR,MRR,NDCG = [],[],[]
            # prediction, label = model.step(sess, None)
            # try:
            #     while True:
            #         prediction, label = model.step(sess, None)
            #
            #         label = int(label[0]) # 用户最后一个交互项目
            #         HR.append(metrics.hit(label, prediction))
            #         MRR.append(metrics.mrr(label, prediction))
            #         NDCG.append(metrics.ndcg(label, prediction))
            # except tf.errors.OutOfRangeError:
            #     hr = np.array(HR).mean()
            #     mrr = np.array(MRR).mean()
            #     ndcg = np.array(NDCG).mean()
            #     print("Epoch %d testing  " % epoch + "Took: " + time.strftime("%H: %M: %S",
            #                                                                   time.gmtime(time.time() - start_time)))
            #     print("HR is %.3f, MRR is %.3f, NDCG is %.3f" % (hr, mrr, ndcg))

        ################################## SAVE MODEL ################################
        checkpoint_path = os.path.join(FLAGS.model_dir, "aSDAE.ckpt")
        model.saver.save(sess, checkpoint_path)


def main():
    rating_data, user_size, item_size = input.load_rating_data()

    print(rating_data.head())

    dataset = tf.data.Dataset.from_tensor_slices(rating_data.items)
    # dataset = dataset.shuffle(100000).batch(FLAGS.batch_size)

    train(dataset, rating_data, user_size, item_size)

if __name__ == '__main__':
    main()