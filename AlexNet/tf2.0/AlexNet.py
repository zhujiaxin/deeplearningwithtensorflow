# -*- coding:utf-8 -*-
import tensorflow as tf
from utils.costumelayers import MyDenseLayer

def alexnet():
    input = tf.keras.Input([227, 227, 3], dtype=tf.float32)
    regular = tf.keras.regularizers.l1(0.005)
    out = tf.keras.layers.Conv2D(filters=96, kernel_size=[11, 11], activation='relu', padding='valid',
                                 strides=[4, 4], kernel_regularizer=regular)(input)
    out = tf.keras.layers.MaxPool2D(pool_size=[3, 3], strides=[2, 2], padding='valid')(out)
    out = tf.keras.layers.Conv2D(filters=256, kernel_size=[5, 5], activation='relu', padding='same',
                                 strides=[1, 1], kernel_regularizer=regular)(out)
    out = tf.keras.layers.MaxPool2D(pool_size=[3, 3], strides=[2, 2], padding='valid')(out)
    out = tf.keras.layers.Conv2D(filters=384, kernel_size=[3, 3], activation='relu', padding='same',
                                 strides=[1, 1], kernel_regularizer=regular)(out)
    out = tf.keras.layers.Conv2D(filters=384, kernel_size=[3, 3], activation='relu', padding='same',
                                 strides=[1, 1], kernel_regularizer=regular)(out)
    out = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], activation='relu', padding='same',
                                 strides=[1, 1], kernel_regularizer=regular)(out)
    out = tf.keras.layers.MaxPool2D(pool_size=[3, 3], strides=[2, 2], padding='valid')(out)
    out = tf.keras.layers.Flatten()(out)
    out = tf.keras.layers.Dense(4096, activation='relu', kernel_regularizer=regular)(out)
    out = tf.keras.layers.Dropout(0.5)(out)
    out = tf.keras.layers.Dense(4096, activation='relu', kernel_regularizer=regular)(out)
    out = tf.keras.layers.Dropout(0.5)(out)
    out = tf.keras.layers.Dense(1000, activation='softmax', kernel_regularizer=regular)(out)
    model = tf.keras.Model(inputs=input, outputs=out)
    return model


def alexnet2():
    input = tf.keras.Input([227, 227, 3], dtype=tf.float32)
    regular = tf.keras.regularizers.l1(0.005)
    out = tf.keras.layers.Conv2D(filters=96, kernel_size=[11, 11], activation='relu', padding='valid',
                                 strides=[4, 4], kernel_regularizer=regular)(input)
    out = tf.keras.layers.MaxPool2D(pool_size=[3, 3], strides=[2, 2], padding='valid')(out)
    out = tf.keras.layers.Conv2D(filters=256, kernel_size=[5, 5], activation='relu', padding='same',
                                 strides=[1, 1], kernel_regularizer=regular)(out)
    out = tf.keras.layers.MaxPool2D(pool_size=[3, 3], strides=[2, 2], padding='valid')(out)
    out = tf.keras.layers.Conv2D(filters=384, kernel_size=[3, 3], activation='relu', padding='same',
                                 strides=[1, 1], kernel_regularizer=regular)(out)
    out = tf.keras.layers.Conv2D(filters=384, kernel_size=[3, 3], activation='relu', padding='same',
                                 strides=[1, 1], kernel_regularizer=regular)(out)
    out = tf.keras.layers.Conv2D(filters=256, kernel_size=[3, 3], activation='relu', padding='same',
                                 strides=[1, 1], kernel_regularizer=regular)(out)
    out = tf.keras.layers.MaxPool2D(pool_size=[3, 3], strides=[2, 2], padding='valid')(out)
    out = tf.keras.layers.Flatten()(out)
    weights1 = tf.ones(shape=[9216, 4096], dtype=tf.float32)
    biases1 = tf.ones(shape=[4096], dtype=tf.float32)
    out = MyDenseLayer(weights=weights1, biases=biases1, activation='relu', kernel_regularizer=regular)(out)
    out = tf.keras.layers.Dropout(0.5)(out)
    weights2 = tf.ones(shape=[4096, 4096])
    biases2 = tf.ones(shape=[4096])
    out = MyDenseLayer(weights=weights2, biases=biases2, activation='relu', kernel_regularizer=regular)(out)
    out = tf.keras.layers.Dropout(0.5)(out)
    weights3 = tf.ones(shape=[4096, 1000])
    biases3 = tf.ones(shape=[1000])
    out = MyDenseLayer(weights=weights3, biases=biases3, activation='softmax', kernel_regularizer=regular)(out)
    model = tf.keras.Model(inputs=input, outputs=out)
    return model


a = alexnet2()
print(a.summary())

b = alexnet()
print(b.summary())