# -*- coding:utf-8 -*-
import tensorflow.compat.v1 as tf
import numpy as np


class AlexNet(object):
    def __init__(self, lr_rate=0.001, regular=0.005):
        self.parameter = []
        with tf.name_scope('input_layer'):
            self.input_x = tf.placeholder(dtype=tf.float32, shape=[None, 224, 224, 3], name='input_x')
            self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, 1000], name='input_y')

        with tf.name_scope('first_conv_layer'):
            kernel = tf.Variable(tf.truncated_normal(shape=[11, 11, 3, 96], stddev=0.01, dtype=tf.float32),
                                 name='kernel_weight', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'loss'])
            bias = tf.Variable(tf.constant(value=1, shape=[96], dtype=tf.float32), name='kernel_bias')
            conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.input_x, kernel, strides=[1, 4, 4, 1],
                                              padding='VALID'), bias))
        self.parameter.append([kernel, bias])
        with tf.name_scope('first_maxpoll_layer'):
            maxpool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        with tf.name_scope('second_conv_layer'):
            kernel2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 96, 256], stddev=0.01, dtype=tf.float32),
                                  name='kernel_weight', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'loss'])
            bias2 = tf.Variable(tf.constant(value=1, shape=[256], dtype=tf.float32), name='kernel_bias')
            conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(maxpool1, kernel2, strides=[1, 1, 1, 1],
                                              padding='SAME'), bias2))
        self.parameter.append([kernel2, bias2])
        with tf.name_scope('second_maxpool_layer'):
            maxpool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        with tf.name_scope('third_conv_layer'):
            kernel3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 384], stddev=0.01, dtype=tf.float32),
                                  name='kernel_weight', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'loss'])
            bias3 = tf.Variable(tf.constant(value=1, shape=[384], dtype=tf.float32), name='kernel_bias')
            conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(maxpool2, kernel3, strides=[1, 1, 1, 1],
                                              padding='SAME'), bias3))
        self.parameter.append([kernel3, bias3])
        with tf.name_scope('fourth_conv_layer'):
            kernel4 = tf.Variable(tf.truncated_normal(shape=[3, 3, 384, 384], stddev=0.01, dtype=tf.float32),
                                  name='kernel_weight', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'loss'])
            bias4 = tf.Variable(tf.constant(value=1, shape=[384], dtype=tf.float32), name='kernel_bias')
            conv4 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv3, kernel4, strides=[1, 1, 1, 1],
                                              padding='SAME'), bias4))
        self.parameter.append([kernel4, bias4])
        with tf.name_scope('fifth_conv_layer'):
            kernel5 = tf.Variable(tf.truncated_normal(shape=[3, 3, 384, 256], stddev=0.01, dtype=tf.float32),
                                  name='kernel_weight', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'loss'])
            bias5 = tf.Variable(tf.constant(value=1, shape=[256], dtype=tf.float32), name='kernel_bias')
            conv5 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv4, kernel5, strides=[1, 1, 1, 1],
                                              padding='SAME'), bias5))
        self.parameter.append([kernel5, bias5])
        dim_list = conv5.get_shape().as_list()[1:]
        shape_dim = np.prod(dim_list)
        reshaped = tf.reshape(conv5, [-1, shape_dim])

        with tf.name_scope('first_fc_layer'):
            weight1 = tf.Variable(tf.truncated_normal(shape=[shape_dim, 4096], stddev=0.01, dtype=tf.float32),
                                  name='fc_weight', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'loss'])
            bias6 = tf.Variable(tf.constant(shape=[4096], value=1, dtype=tf.float32), name='fc_bias')
            fc1_out = tf.nn.relu(tf.nn.bias_add(tf.matmul(reshaped, weight1), bias6))
            drop1 = tf.nn.dropout(fc1_out, rate=0.5)
        self.parameter.append([weight1, bias6])

        with tf.name_scope('second_fc_layer'):
            weight2 = tf.Variable(tf.truncated_normal(shape=[4096, 4096], stddev=0.01, dtype=tf.float32),
                                  name='fc_weight', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'loss'])
            bias7 = tf.Variable(tf.constant(shape=[4096], value=1, dtype=tf.float32), name='fc_bias')
            fc2_out = tf.nn.relu(tf.nn.bias_add(tf.matmul(drop1, weight2), bias7))
            drop2 = tf.nn.dropout(fc2_out, rate=0.5)
        self.parameter.append([weight2, bias7])
        with tf.name_scope('thrid_fc_layer'):
            weight3 = tf.Variable(tf.truncated_normal(shape=[4096, 1000], stddev=0.01), dtype=tf.float32,
                                  name='fc_weight', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'loss'])
            bias8 = tf.Variable(tf.constant(shape=[1000], value=1, dtype=tf.float32), name='fc_bias')
            self.out = tf.nn.softmax(tf.nn.bias_add(tf.matmul(drop2, weight3), bias8))
        self.parameter.append([weight3, bias8])
        with tf.name_scope('loss'):
            regulation_loss = 0
            for i in tf.get_collection('loss'):
                tensor = tf.get_default_graph().get_tensor_by_name(i.name)
                regulation_loss += tf.nn.l2_loss(tensor)
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_y, logits=self.out))\
                + regulation_loss * regular
            self.train_loss_op = tf.train.GradientDescentOptimizer(lr_rate).minimize(self.loss)

        with tf.name_scope('accuracy'):
            self.accuracy = tf.reduce_mean(
                             tf.cast(
                              tf.equal(tf.argmax(self.out, axis=1),
                                       tf.argmax(self.input_y, axis=1)), tf.float32))

    def change_lr_rate(self, lr_rate):
        self.train_loss_op = tf.train.GradientDescentOptimizer(lr_rate).minimize(self.loss)

    def train(self, train_x=None, train_y=None, session=None):
        """train_y需要已经转成onehot"""
        session.run(self.train_loss_op, feed_dict={self.input_x: train_x,
                                                   self.input_y: train_y})

    def compute_loss(self, test_x, test_y, session=None):
        return session.run(self.loss, feed_dict={self.input_x: test_x,
                                                 self.input_y: test_y})

    def compute_accuracy(self, data_x, data_y, session=None):
        return session.run(self.accuracy, feed_dict={self.input_x: data_x,
                                                     self.input_y: data_y})


class AlexNet2(object):
    def __init__(self, lr_rate=0.001, regular=0.005, trainable=False):
        self.parameter = []
        with tf.name_scope('input_layer'):
            self.input_x = tf.placeholder(dtype=tf.float32, shape=[None, 227, 227, 3], name='input_x')
            self.input_y = tf.placeholder(dtype=tf.float32, shape=[None, 1000], name='input_y')

        with tf.name_scope('first_conv_layer_part1'):
            kernel1_1 = tf.Variable(tf.truncated_normal(shape=[11, 11, 3, 48], stddev=0.01, dtype=tf.float32),
                                    name='kernel_weight', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'loss'])
            bias1_1 = tf.Variable(tf.constant(value=0, shape=[48], dtype=tf.float32), name='kernel_bias')
            conv1_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.input_x, kernel1_1, strides=[1, 4, 4, 1],
                                                             padding='VALID'), bias1_1))
            lrn1_1 = tf.nn.local_response_normalization(conv1_1, depth_radius=2, bias=1, alpha=2e-05, beta=0.75)
        self.parameter.append([kernel1_1, bias1_1])

        with tf.name_scope('fisrt_conv_layer_part2'):
            kernel1_2 = tf.Variable(tf.truncated_normal(shape=[11, 11, 3, 48], stddev=0.01, dtype=tf.float32),
                                    name='kernel_weight', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'loss'])
            bias1_2 = tf.Variable(tf.constant(value=0, shape=[48], dtype=tf.float32), name='kernel_bias')
            conv1_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.input_x, kernel1_2, strides=[1, 4, 4, 1],
                                                             padding='VALID'), bias1_2))
            lrn1_2 = tf.nn.local_response_normalization(conv1_2, depth_radius=2, bias=1, alpha=2e-05, beta=0.75)
        self.parameter.append([kernel1_2, bias1_2])

        with tf.name_scope('first_maxpool_layer_part1'):
            maxpool1_1 = tf.nn.max_pool(lrn1_1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        with tf.name_scope('fisrt_maxpool_layer_part2'):
            maxpool1_2 = tf.nn.max_pool(lrn1_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        with tf.name_scope('second_conv_layer_part1'):
            kernel2_1 = tf.Variable(tf.truncated_normal(shape=[5, 5, 48, 128], stddev=0.01, dtype=tf.float32),
                                    name='kernel_weight', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'loss'])
            bias2_1 = tf.Variable(tf.constant(value=1, shape=[128], dtype=tf.float32), name='kernel_bias')
            conv2_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(maxpool1_1, kernel2_1, strides=[1, 1, 1, 1],
                                                             padding='SAME'), bias2_1))
            lrn2_1 = tf.nn.local_response_normalization(conv2_1, depth_radius=2, bias=1, alpha=2e-05, beta=0.75)
        self.parameter.append([kernel2_1, bias2_1])

        with tf.name_scope('second_conv_layer_part2'):
            kernel2_2 = tf.Variable(tf.truncated_normal(shape=[5, 5, 48, 128], stddev=0.01, dtype=tf.float32),
                                    name='kernel_weight', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'loss'])
            bias2_2 = tf.Variable(tf.constant(value=0, shape=[128], dtype=tf.float32), name='kernel_bias')
            conv2_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(maxpool1_2, kernel2_2, strides=[1, 1, 1, 1],
                                                             padding='SAME'), bias2_2))
            lrn2_2 = tf.nn.local_response_normalization(conv2_2, depth_radius=2, bias=1, alpha=2e-05, beta=0.75)
        self.parameter.append([kernel2_2, bias2_2])

        with tf.name_scope('second_maxpool_layer_part1'):
            maxpool2_1 = tf.nn.max_pool(lrn2_1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        with tf.name_scope('second_maxpool_layer_part2'):
            maxpool2_2 = tf.nn.max_pool(lrn2_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        '''in paper conv3 have four conv kernels so there have four,
           in many codes for alexnet they have only one kernel have shape=[3, 3, 256, 384]
           they are same,because 128X2=256, 192X2=384'''
        with tf.name_scope('third_conv_layer_part1'):
            kernel3_1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 192], stddev=0.01, dtype=tf.float32),
                                    name='kernel_weight1', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'loss'])
            conv3_1 = tf.nn.conv2d(maxpool2_1, kernel3_1, strides=[1, 1, 1, 1], padding='SAME')
            kernel3_2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 192], stddev=0.01, dtype=tf.float32),
                                    name="kernel_weight2", collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'loss'])
            conv3_2 = tf.nn.conv2d(maxpool2_1, kernel3_2, strides=[1, 1, 1, 1], padding='SAME')
        self.parameter.append([kernel3_1, kernel3_2])

        with tf.name_scope('third_conv_layer_part2'):
            kernel3_3 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 192], stddev=0.01, dtype=tf.float32),
                                    name='kernel_weight1', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'loss'])
            conv3_3 = tf.nn.conv2d(maxpool2_2, kernel3_3, strides=[1, 1, 1, 1], padding='SAME')
            kernel3_4 = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 192], stddev=0.01, dtype=tf.float32),
                                    name='kernel_weight2', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'loss'])
            conv3_4 = tf.nn.conv2d(maxpool2_2, kernel3_4, strides=[1, 1, 1, 1], padding='SAME')
        self.parameter.append([kernel3_3, kernel3_4])

        with tf.name_scope('make_two_as_one'):
            bias3_1 = tf.Variable(tf.constant(value=1, shape=[192], dtype=tf.float32), name='bias3_1')
            bias3_2 = tf.Variable(tf.constant(value=1, shape=[192], dtype=tf.float32), name='bias3_1')
            conv3_out1 = tf.nn.bias_add(tf.nn.relu(conv3_1 + conv3_3), bias3_1)
            conv3_out2 = tf.nn.bias_add(tf.nn.relu(conv3_2 + conv3_4), bias3_2)
        self.parameter.append([bias3_1, bias3_2])

        with tf.name_scope('fourth_conv_layer_part1'):
            kernel4_1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 192, 192], stddev=0.01, dtype=tf.float32),
                                    name='kernel_weight', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'loss'])
            bias4_1 = tf.Variable(tf.constant(value=1, shape=[192], dtype=tf.float32), name='kernel_bias')
            conv4_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv3_out1, kernel4_1, strides=[1, 1, 1, 1],
                                                             padding='SAME'), bias4_1))
        self.parameter.append([kernel4_1, bias4_1])

        with tf.name_scope('fourth_conv_layer_part2'):
            kernel4_2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 192, 192], stddev=0.01, dtype=tf.float32),
                                    name='kernel_weight', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'loss'])
            bias4_2 = tf.Variable(tf.constant(value=1, shape=[192], dtype=tf.float32), name='kernel_bias')
            conv4_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv3_out2, kernel4_2, strides=[1, 1, 1, 1],
                                                             padding='SAME'), bias4_2))
        self.parameter.append([kernel4_2, bias4_2])

        with tf.name_scope('fifth_conv_layer_part1'):
            kernel5_1 = tf.Variable(tf.truncated_normal(shape=[3, 3, 192, 128], stddev=0.01, dtype=tf.float32),
                                    name='kernel_weight', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'loss'])
            bias5_1 = tf.Variable(tf.constant(value=1, shape=[128], dtype=tf.float32), name='kernel_bias')
            conv5_1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv4_1, kernel5_1, strides=[1, 1, 1, 1],
                                                padding='SAME'), bias5_1))
        self.parameter.append([kernel5_1, bias5_1])

        with tf.name_scope('fifth_conv_layer_part2'):
            kernel5_2 = tf.Variable(tf.truncated_normal(shape=[3, 3, 192, 128], stddev=0.01, dtype=tf.float32),
                                    name='kernel_weight', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'loss'])
            bias5_2 = tf.Variable(tf.constant(value=1, shape=[128], dtype=tf.float32), name='kernel_bias')
            conv5_2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv4_2, kernel5_2, strides=[1, 1, 1, 1],
                                                             padding='SAME'), bias5_2))
        self.parameter.append([kernel5_2, bias5_2])
        with tf.name_scope('fifth_maxpool_layer_part1'):
            maxpool5_1 = tf.nn.max_pool(conv5_1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        with tf.name_scope('fifth_maxpool_layer_part2'):
            maxpool5_2 = tf.nn.max_pool(conv5_2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='VALID')

        conv_out = tf.concat([maxpool5_1, maxpool5_2], 3)
        dim_list = conv_out.get_shape().as_list()[1:]
        shape_dim = np.prod(dim_list)
        reshaped = tf.reshape(conv_out, [-1, shape_dim])

        with tf.name_scope('first_fc_layer'):
            weight1 = tf.Variable(tf.truncated_normal(shape=[shape_dim, 4096], stddev=0.01, dtype=tf.float32),
                                  name='fc_weight', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'loss'])
            bias6 = tf.Variable(tf.constant(shape=[4096], value=1, dtype=tf.float32), name='fc_bias')
            fc1_out = tf.nn.relu(tf.nn.bias_add(tf.matmul(reshaped, weight1), bias6))
            if trainable:
                fc1_out = tf.nn.dropout(fc1_out, rate=0.5)
        self.parameter.append([weight1, bias6])

        with tf.name_scope('second_fc_layer'):
            weight2 = tf.Variable(tf.truncated_normal(shape=[4096, 4096], stddev=0.01, dtype=tf.float32),
                                  name='fc_weight', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'loss'])
            bias7 = tf.Variable(tf.constant(shape=[4096], value=1, dtype=tf.float32), name='fc_bias')
            fc2_out = tf.nn.relu(tf.nn.bias_add(tf.matmul(fc1_out, weight2), bias7))
            if trainable:
                fc2_out = tf.nn.dropout(fc2_out, rate=0.5)
        self.parameter.append([weight2, bias7])

        with tf.name_scope('thrid_fc_layer'):
            weight3 = tf.Variable(tf.truncated_normal(shape=[4096, 1000], stddev=0.01), dtype=tf.float32,
                                  name='fc_weight', collections=[tf.GraphKeys.GLOBAL_VARIABLES, 'loss'])
            bias8 = tf.Variable(tf.constant(shape=[1000], value=1, dtype=tf.float32), name='fc_bias')
            self.out = tf.nn.softmax(tf.nn.bias_add(tf.matmul(fc2_out, weight3), bias8))
        self.parameter.append([weight3, bias8])
        with tf.name_scope('loss'):
            regulation_loss = 0
            for i in tf.get_collection('loss'):
                tensor = tf.get_default_graph().get_tensor_by_name(i.name)
                regulation_loss += tf.nn.l2_loss(tensor)
            self.loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.input_y, logits=self.out))\
                + regulation_loss * regular
            self.train_loss_op = tf.train.GradientDescentOptimizer(lr_rate).minimize(self.loss)

        with tf.name_scope('accuracy'):
            self.accuracy = tf.reduce_mean(
                             tf.cast(
                              tf.equal(tf.argmax(self.out, axis=1),
                                       tf.argmax(self.input_y, axis=1)), tf.float32))

    def change_lr_rate(self, lr_rate):
        self.train_loss_op = tf.train.GradientDescentOptimizer(lr_rate).minimize(self.loss)

    def train(self, train_x=None, train_y=None, session=None):
        """train_y需要已经转成onehot"""
        session.run(self.train_loss_op, feed_dict={self.input_x: train_x,
                                                   self.input_y: train_y})

    def compute_loss(self, test_x, test_y, session=None):
        return session.run(self.loss, feed_dict={self.input_x: test_x,
                                                 self.input_y: test_y})

    def compute_accuracy(self, data_x, data_y, session=None):
        return session.run(self.accuracy, feed_dict={self.input_x: data_x,
                                                     self.input_y: data_y})
