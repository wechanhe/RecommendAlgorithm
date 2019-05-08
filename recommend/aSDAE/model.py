#!/usr/bin/python 
# -*- coding: UTF-8 -*-

import numpy as np
import pandas as pd
import tensorflow as tf

class aSDAE(object):
    def __init__(self, embed_size, user_size, item_size, lr, optim, initializer,
                 activation_func, regularizer_rate, dropout, iterator, is_training):
        '''
        :param embed_size: the middle embedding size for users and items
        :param lr: learning rate
        :param optim: optimizer
        :param initializer: the initialization method
        :param loss_func: Loss function, MSE
        :param activation_func: activation function
        :param reguration_rate: regularizer rate L1 or L2
        :param dropout: whether using dropout
        :param iterator: iterator to get batch samples
        :param is_training: whether training
        '''
        self.embed_size = embed_size
        self.user_size = user_size
        self.item_size = item_size
        self.lr = lr
        self.optim = optim
        self.initializer = initializer
        self.activation_func = activation_func
        self.regularizer_rate = regularizer_rate
        self.dropout = dropout
        self.iterator = iterator
        self.is_training = is_training

    def get_data(self):
        sample = self.iterator.get_next()
        self.user = sample['user']
        self.item = sample['item']
        self.rating = tf.cast(sample['rating'], tf.float16)

    def inferenct(self):
        """ Initialize important settings """
        self.k_regularizer = tf.keras.regularizers.l2(self.regularizer_rate)
        self.b_regularizer = tf.keras.regularizers.l1(self.regularizer_rate)

        if self.initializer == 'Normal':
            self.initializer = tf.truncated_normal_initializer(stddev=0.01)
        elif self.initializer == 'Xavier_Normal':
            self.initializer = tf.contrib.layers.xavier_initializer()
        else:
            self.initializer = tf.glorot_uniform_initializer()

        if self.activation_func == 'ReLU':
            self.activation_func = tf.nn.relu
        elif self.activation_func == 'Leaky_ReLU':
            self.activation_func = tf.nn.leaky_relu
        elif self.activation_func == 'ELU':
            self.activation_func = tf.nn.elu

        if self.optim == 'SGD':
            self.optim = tf.train.GradientDescentOptimizer(self.lr,
                                                           name='SGD')
        elif self.optim == 'RMSProp':
            self.optim = tf.train.RMSPropOptimizer(self.lr, decay=0.9,
                                                   momentum=0.0, name='RMSProp')
        elif self.optim == 'Adam':
            self.optim = tf.train.AdamOptimizer(self.lr, name='Adam')

    def create_model(self):
        with tf.name_scope('input'):
            self.user_input = tf.placeholder(tf.float16, [None, self.item_size], name='user_input')
            self.item_input = tf.placeholder(tf.float16, [None, self.user_size], name='item_input')

        with tf.name_scope('embedding'):
            self.user_embed = tf.layers.dense(self.user_input, self.embed_size,
                                              activation=self.activation_func,
                                              kernel_regularizer=self.k_regularizer,
                                              bias_regularizer=self.b_regularizer,
                                              name='user_embedding')
            self.item_embed = tf.layers.dense(self.item_input, self.embed_size,
                                              activation=self.activation_func,
                                              kernel_regularizer=self.k_regularizer,
                                              bias_regularizer=self.b_regularizer,
                                              name='item_embedding')

        with tf.name_scope('output'):
            self.user_output = tf.layers.dense(self.user_embed, self.user_size,
                                               kernel_regularizer=self.k_regularizer,
                                               bias_regularizer=self.b_regularizer,
                                               activation=None, name='user_output')
            self.item_output = tf.layers.dense(self.item_embed, self.item_size,
                                               kernel_regularizer=self.k_regularizer,
                                               bias_regularizer=self.b_regularizer,
                                               activation=None, name='item_output')

        with tf.name_scope('loss'):
            u_loss = tf.reduce_sum(self.user_output - self.user_input)
            i_loss = tf.reduce_sum(self.item_output - self.item_input)
            ui_loss = tf.reduce_sum(
                tf.subtract(
                    tf.multiply(self.user_embed, self.item_embed), self.rating
                )
            )
            self.loss = u_loss + i_loss + ui_loss

        with tf.name_scope('optimization'):
            self.optimizer = self.optim.minimize(self.loss)

        return self.loss

    def eval(self):
        pass

    def summary(self):
        """ Create summaries to write on tensorboard. """
        self.writer = tf.summary.FileWriter('./graphs/', tf.get_default_graph)
        tf.summary.scalar('loss', self.loss)
        tf.summary.histogram('histogram loss', self.loss)
        self.summary_op = tf.summary.merge_all()

    def build(self):
        self.get_data()
        self.inference()
        self.create_model()
        self.eval()
        self.summary()
        self.saver = tf.train.Saver(tf.global_variables())

    def step(self, session, step):
        """ Train the model step by step. """
        if self.is_training:
            loss, optim, summaries = session.run(
                [self.loss, self.optimizer, self.summary_op])
            self.writer.add_summary(summaries, global_step=step)
        else:
            loss, optim, summaries = session.run(
                [self.loss, self.optimizer, self.summary_op])
            return loss