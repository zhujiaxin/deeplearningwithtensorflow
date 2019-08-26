# -*- coding:utf-8 -*-
import tensorflow as tf

def model():
    inputs = tf.keras.Input((28, 28, 1))
    convlay1 = tf.keras.layers.Conv2D(filters=6, kernel_size=[5, 5], strides=[1, 1], activation='relu')(inputs)
    maxpool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2])(convlay1)
    convlay2 = tf.keras.layers.Conv2D(filters=16, kernel_size=[5, 5], strides=[1, 1], activation='relu')(maxpool1)
    maxpool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=[2, 2])(convlay2)
    flatten = tf.keras.layers.Flatten()(maxpool2)
    fc1 = tf.keras.layers.Dense(120, activation='relu')(flatten)
    fc2 = tf.keras.layers.Dense(84, activation='relu')(fc1)
    fc3 = tf.keras.layers.Dense(10, activation='softmax')(fc2)
    net = tf.keras.Model(inputs=inputs, outputs=fc3)
    return net


