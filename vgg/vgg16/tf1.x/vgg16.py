# -*- coding:utf-8 -*-

import tensorflow.compat.v1 as tf


class vgg16(object):
    def __init__(self):
        """定义vgg16模型"""
        self.parameter = []
        with tf.name_scope('input'):
            self.input_x = tf.placeholder(shape=[None, 224, 224, 3], dtype=tf.float32, name='input_x')
            self.input_y = tf.placeholder(shape=[None, 1000], dtype=tf.float32, name='input_y')

        with tf.name_scope('conv1_1'):
            conv11_weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 3, 64], dtype=tf.float32,
                                                             stddev=1e-1),
                                         name='weights',
                                         trainable=True)
            conv11_biases = tf.Variable(tf.constant(value=0, shape=[64], dtype=tf.float32),
                                        name='baises',
                                        trainable=True)
            conv11_out = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(self.input_x,
                                                                conv11_weights,
                                                                strides=[1, 1, 1, 1],
                                                                padding='SAME'),
                                                   conv11_biases))
            self.parameter += [conv11_weights, conv11_biases]
        with tf.name_scope('conv1_2'):
            conv12_weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 64], dtype=tf.float32,
                                                             stddev=1e-1),
                                         name='weights',
                                         trainable=True)
            conv12_biases = tf.Variable(tf.constant(value=0, shape=[64], dtype=tf.float32),
                                        name='baises',
                                        trainable=True)
            conv12_out = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv11_out,
                                                                conv12_weights,
                                                                strides=[1, 1, 1, 1],
                                                                padding='SAME'),
                                                   conv12_biases))
            self.parameter += [conv12_weights, conv12_biases]
        with tf.name_scope('maxpool1'):
            maxpool1_out = tf.nn.max_pool(conv12_out,
                                         ksize=[1, 2, 2, 1],
                                         strides=[1, 2, 2, 1],
                                         padding='SAME',
                                         name='pool1')
        with tf.name_scope('conv2_1'):
            conv21_weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 64, 128], dtype=tf.float32,
                                                             stddev=1e-1),
                                         name='weights',
                                         trainable=True)
            conv21_biases = tf.Variable(tf.constant(value=0, shape=[128], dtype=tf.float32),
                                        name='baises',
                                        trainable=True)
            conv21_out = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(maxpool1_out,
                                                                conv21_weights,
                                                                strides=[1, 1, 1, 1],
                                                                padding='SAME'),
                                                   conv21_biases))
            self.parameter += [conv21_weights, conv21_biases]
        with tf.name_scope('conv2_2'):
            conv22_weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 128], dtype=tf.float32,
                                                             stddev=1e-1),
                                         name='weights',
                                         trainable=True)
            conv22_biases = tf.Variable(tf.constant(value=0, shape=[128], dtype=tf.float32),
                                        name='biases',
                                        trainable=True)
            conv22_out = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv21_out,
                                                                conv22_weights,
                                                                strides=[1, 1, 1, 1],
                                                                padding='SAME'),
                                                   conv22_biases))
            self.parameter += [conv22_weights, conv22_biases]
        with tf.name_scope('maxpool2'):
            maxpool2_out = tf.nn.max_pool(conv22_out,
                                         ksize=[1, 2, 2, 1],
                                         strides=[1, 2, 2, 1],
                                         padding='SAME',
                                         name='pool2')
        with tf.name_scope('conv3_1'):
            conv31_weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 128, 256],
                                                             dtype=tf.float32,
                                                             stddev=1e-1),
                                         name='weights',
                                         trainable=True)
            conv31_biases = tf.Variable(tf.constant(value=0,
                                                    shape=[256],
                                                    dtype=tf.float32),
                                        name='biases',
                                        trainable=True)
            conv31_out = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(maxpool2_out,
                                                                conv31_weights,
                                                                strides=[1, 1, 1, 1],
                                                                padding='SAME'),
                                                   conv31_biases))
            self.parameter += [conv31_weights, conv31_biases]
        with tf.name_scope('conv3_2'):
            conv32_weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 256],
                                                             dtype=tf.float32,
                                                             stddev=1e-1),
                                         name='weighs',
                                         trainable=True)
            conv32_biases = tf.Variable(tf.constant(value=0,
                                                    shape=[256],
                                                    dtype=tf.float32),
                                        name='biases',
                                        trainable=True)
            conv32_out = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv31_out,
                                                                conv32_weights,
                                                                strides=[1, 1, 1, 1],
                                                                padding='SAME'),
                                                   conv32_biases))
            self.parameter += [conv32_weights, conv32_biases]
        with tf.name_scope('conv3_3'):
            conv33_weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 256],
                                                             stddev=1e-1,
                                                             dtype=tf.float32),
                                         name='weights',
                                         trainable=True)
            conv33_biases = tf.Variable(tf.constant(value=0,
                                                    shape=[256],
                                                    dtype=tf.float32),
                                        name='biases',
                                        trainable=True)
            conv33_out = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv32_out,
                                                                conv33_weights,
                                                                strides=[1, 1, 1, 1],
                                                                padding='SAME'),
                                                   conv33_biases))
            self.parameter += [conv33_weights, conv33_biases]
        with tf.name_scope('maxpool3'):
            maxpool3_out = tf.nn.max_pool(conv33_out,
                                         ksize=[1, 2, 2, 1],
                                         strides=[1, 2, 2, 1],
                                         padding='SAME')
        with tf.name_scope('conv4_1'):
            conv41_weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 256, 512],
                                                             stddev=1e-1,
                                                             dtype=tf.float32),
                                         name='weights',
                                         trainable=True)
            conv41_biases = tf.Variable(tf.constant(value=0,
                                                    shape=[512],
                                                    dtype=tf.float32),
                                        name='biases',
                                        trainable=True)
            conv41_out = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(maxpool3_out,
                                                                conv41_weights,
                                                                strides=[1, 1, 1, 1],
                                                                padding='SAME'),
                                                   conv41_biases))
            self.parameter += [conv41_weights, conv41_biases]
        with tf.name_scope('conv4_2'):
            conv42_weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512],
                                                             stddev=1e-1,
                                                             dtype=tf.float32),
                                         name='weights',
                                         trainable=True)
            conv42_biases = tf.Variable(tf.constant(value=0,
                                                    shape=[512],
                                                    dtype=tf.float32),
                                        name='biases',
                                        trainable=True)
            conv42_out = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv41_out,
                                                                conv42_weights,
                                                                strides=[1, 1, 1, 1],
                                                                padding='SAME'),
                                                   conv42_biases))
            self.parameter += [conv42_weights, conv42_biases]
        with tf.name_scope('conv4_3'):
            conv43_weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512],
                                                             stddev=1e-1,
                                                             dtype=tf.float32),
                                         name='weights',
                                         trainable=True)
            conv43_biases = tf.Variable(tf.constant(value=0,
                                                    shape=[512],
                                                    dtype=tf.float32),
                                        name='biases',
                                        trainable=True)
            conv43_out = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv42_out,
                                                                conv43_weights,
                                                                strides=[1, 1, 1, 1],
                                                                padding='SAME'),
                                                   conv43_biases))
            self.parameter += [conv43_weights, conv43_biases]
        with tf.name_scope('maxpool4'):
            maxpool4_out = tf.nn.max_pool(conv43_out,
                                         ksize=[1, 2, 2, 1],
                                         strides=[1, 2, 2, 1],
                                         padding='SAME')
        with tf.name_scope('conv5_1'):
            conv51_weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512],
                                                             stddev=1e-1,
                                                             dtype=tf.float32),
                                         name='weights',
                                         trainable=True)
            conv51_biases = tf.Variable(tf.constant(value=0,
                                                    shape=[512],
                                                    dtype=tf.float32),
                                        name='biases',
                                        trainable=True)
            conv51_out = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(maxpool4_out,
                                                                conv51_weights,
                                                                strides=[1, 1, 1, 1],
                                                                padding='SAME'),
                                                   conv51_biases))
            self.parameter += [conv51_weights, conv51_biases]
        with tf.name_scope('conv5_2'):
            conv52_weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512],
                                                             stddev=1e-1,
                                                             dtype=tf.float32),
                                         name='weights',
                                         trainable=True)
            conv52_biases = tf.Variable(tf.constant(value=0,
                                                    shape=[512],
                                                    dtype=tf.float32),
                                        name='biases',
                                        trainable=True)
            conv52_out = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv51_out,
                                                                conv52_weights,
                                                                strides=[1, 1, 1, 1],
                                                                padding='SAME'),
                                                   conv52_biases))
            self.parameter += [conv52_weights, conv52_biases]
        with tf.name_scope('conv5_3'):
            conv53_weights = tf.Variable(tf.truncated_normal(shape=[3, 3, 512, 512],
                                                             stddev=1e-1,
                                                             dtype=tf.float32),
                                         name='weights',
                                         trainable=True)
            conv53_biases = tf.Variable(tf.constant(value=0,
                                                    shape=[512],
                                                    dtype=tf.float32),
                                        name='biases',
                                        trainable=True)
            conv53_out = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv52_out,
                                                                conv53_weights,
                                                                strides=[1, 1, 1, 1],
                                                                padding='SAME'),
                                                   conv53_biases))
            self.parameter += [conv53_weights, conv53_biases]
        with tf.name_scope('maxpool5'):
            maxpool5_out = tf.nn.max_pool(conv53_out,
                                         ksize=[1, 2, 2, 1],
                                         strides=[1, 2, 2, 1],
                                         padding='SAME')
        with tf.name_scope('flatten'):
            flatten = tf.layers.flatten(maxpool5_out)
        with tf.name_scope('fc1'):
            print(flatten.get_shape().as_list()[1])
            fc1_weights = tf.Variable(tf.truncated_normal(shape=[flatten.get_shape().as_list()[1], 4096],
                                                          stddev=1e-1,
                                                          dtype=tf.float32),
                                      name='weights',
                                      trainable=True)
            fc1_biases = tf.Variable(tf.constant(value=0,
                                                 shape=[4096],
                                                 dtype=tf.float32),
                                     name='biases',
                                     trainable=True)
            fc1_out = tf.nn.relu(tf.nn.bias_add(tf.matmul(flatten,
                                                          fc1_weights),
                                                fc1_biases))
            self.parameter += [fc1_weights, fc1_biases]
        with tf.name_scope('fc2'):
            fc2_weights = tf.Variable(tf.truncated_normal(shape=[4096, 4096],
                                                          stddev=1e-1,
                                                          dtype=tf.float32),
                                      name='weights',
                                      trainable=True)
            fc2_biases = tf.Variable(tf.constant(value=0,
                                                 shape=[4096],
                                                 dtype=tf.float32),
                                     name='biases',
                                     trainable=True)
            fc2_out = tf.nn.relu(tf.nn.bias_add(tf.matmul(fc1_out,
                                                          fc2_weights),
                                                fc2_biases))
            self.parameter += [fc2_weights, fc2_biases]
        with tf.name_scope('fc3'):
            fc3_weights = tf.Variable(tf.truncated_normal(shape=[4096, 1000],
                                                          stddev=1e-1,
                                                          dtype=tf.float32),
                                      name='weights',
                                      trainable=True)
            fc3_biases = tf.Variable(tf.constant(value=0,
                                                 shape=[1000],
                                                 dtype=tf.float32),
                                     name='biases',
                                     trainable=True)
            fc3_out = tf.nn.bias_add(tf.matmul(fc2_out,
                                               fc3_weights),
                                     fc3_biases)
            self.parameter += [fc3_weights, fc3_biases]


