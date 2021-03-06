# -*- coding:utf-8 -*-
import tensorflow.compat.v1 as tf
import numpy as np


class LeNet(object):
    """
    :param lr_rate:初始化学习率，可以使用change_lr_rate函数进行调整
    :type  lr_rate:float
    :param regular:L2惩罚系数
    """
    def __init__(self, lr_rate=0.01, regular=0.01, train=False):
        self.out = None
        self.input_x = None
        self.input_y = None
        self.loss = None
        self.out = None
        self.accuracy = None
        self.lr_rate=lr_rate
        self.train = train

        with tf.name_scope('input'):
            self.input_x = tf.placeholder(tf.float32, [None, 28, 28, 1], name='input_x')
            self.input_y = tf.placeholder(tf.float32, [None, 10], name='input_x')

        with tf.name_scope('conv1'):
            kernel1 = tf.Variable(tf.truncated_normal([5, 5, 1, 6], stddev=0.1, dtype=tf.float32), name='weight1',
                                  collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'loss'], trainable=train)
            bias1 = tf.Variable(tf.constant(0.0, shape=[6], dtype=tf.float32), trainable=train, name='bias')
            conv1 = tf.nn.conv2d(self.input_x, kernel1, strides=[1, 1, 1, 1], padding='VALID', name='conv')
            out1 = tf.nn.relu(conv1 + bias1)

        with tf.name_scope('maxpool1'):
            outpool1 = tf.nn.max_pool(out1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='maxpool')

        with tf.name_scope('conv2'):
            kernel2 = tf.Variable(tf.truncated_normal([5, 5, 6, 16], stddev=0.1, dtype=tf.float32),
                                  name='conv_weight2',collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'loss'],
                                  trainable=train)
            bias2 = tf.Variable(tf.constant(0.0, shape=[16], dtype=tf.float32), trainable=train, name='bias')
            conv2 = tf.nn.conv2d(outpool1, kernel2, strides=[1, 1, 1, 1], padding='VALID', name='conv2')
            out2 = tf.nn.relu(conv2 + bias2)

        with tf.name_scope('maxpool2'):
            outpool2 = tf.nn.max_pool(out2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='maxpool')

        with tf.name_scope('fc1'):
            weights1 = tf.Variable(tf.truncated_normal([int(np.prod(outpool2.get_shape()[1:])), 120],
                                                       dtype=tf.float32, stddev=1e-1),
                                   name='fc_weights1',
                                   collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'loss'],
                                   trainable=train)
            bias1_fc = tf.Variable(tf.constant(0.0, shape=[120], dtype=tf.float32),
                                   trainable=train, name='fc_biases1')
            fc1 = tf.nn.relu(
                tf.matmul(tf.reshape(outpool2, [-1, int(np.prod(outpool2.get_shape()[1:]))]), weights1) + bias1_fc)
            if train:
                out3 = tf.nn.dropout(fc1, rate=0.5)
            else:
                out3 = fc1

        with tf.name_scope('fc2'):
            weights2 = tf.Variable(tf.truncated_normal([120, 84],
                                                       dtype=tf.float32,
                                                       stddev=1e-1),
                                   name='fc_weights2',
                                   collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'loss'],
                                   trainable=train)
            bias2_fc = tf.Variable(tf.constant(0.0, shape=[84], dtype=tf.float32),
                                   trainable=train, name='fc_biases2')
            fc2 = tf.nn.relu(tf.matmul(out3, weights2) + bias2_fc)
            if train:
                out4 = tf.nn.dropout(fc2, rate=0.5)
            else:
                out4 = fc2

        with tf.name_scope('outlayer'):
            weights3 = tf.Variable(tf.truncated_normal([84, 10],
                                                       dtype=tf.float32,
                                                       stddev=1e-1),
                                   name='fc_weights3',
                                   collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'loss'],
                                   trainable=train)
            bias3_fc = tf.Variable(tf.constant(0.0, shape=[10], dtype=tf.float32),
                                   trainable=train,
                                   name='fc_biases3')
            self.out = tf.matmul(out4, weights3) + bias3_fc

        with tf.name_scope('loss'):
            regulation_loss = 0
            for i in tf.get_collection('loss'):
                tensor = tf.get_default_graph().get_tensor_by_name(i.name)
                regulation_loss += tf.nn.l2_loss(tensor)
            regulation_loss = tf.nn.l2_loss(weights1) + tf.nn.l2_loss(weights2) + tf.nn.l2_loss(weights3)
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_y, logits=self.out))\
                + regulation_loss * regular
        if self.train:
            self.train_loss_op = tf.train.GradientDescentOptimizer(self.lr_rate).minimize(self.loss)

        with tf.name_scope('accuracy'):
            self.accuracy = tf.reduce_mean(
                             tf.cast(
                              tf.equal(tf.argmax(self.out, axis=1), tf.argmax(self.input_y, axis=1)), tf.float32))

    def change_lr_rate(self, lr_rate):
        self.lr_rate = lr_rate
        self.train_loss_op = tf.train.GradientDescentOptimizer(lr_rate).minimize(self.loss)

    def make_train(self, train_x=None, train_y=None, session=None):
        """train_y需要已经转成onehot"""
        if not self.train:
            raise AttributeError('pleease set model.train=True while training')
        session.run(self.train_loss_op, feed_dict={self.input_x: train_x,
                                                   self.input_y: train_y})

    def compute_loss(self, test_x, test_y, session=None):
        return session.run(self.loss, feed_dict={self.input_x: test_x,
                                                 self.input_y: test_y})

    def compute_accuracy(self, data_x, data_y, session=None):
        return session.run(self.accuracy, feed_dict={self.input_x: data_x,
                                                     self.input_y: data_y})


