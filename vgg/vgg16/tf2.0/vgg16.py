# -*- coding:utf-8 -*-
import tensorflow.keras as keras
import tensorflow.compat.v2 as tf


def vgg16():
    input = keras.Input([224, 224, 3])
    x = keras.layers.Conv2D(filters=64,
                            kernel_size=[3, 3],
                            strides=[1, 1],
                            padding='same',
                            activation='relu')(input)
    for _ in range(1):
        x = keras.layers.Conv2D(filters=64,
                                kernel_size=[3, 3],
                                strides=[1, 1],
                                padding='same',
                                activation='relu')(x)
    x = keras.layers.MaxPool2D(pool_size=[2, 2],
                               strides=[2, 2],
                               padding='same')(x)
    for _ in range(2):
        x = keras.layers.Conv2D(filters=128,
                                kernel_size=[3, 3],
                                strides=[1, 1],
                                padding='same',
                                activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=[2, 2],
                                  strides=[2, 2],
                                  padding='same')(x)
    for _ in range(3):
        x = keras.layers.Conv2D(filters=256,
                                kernel_size=[3, 3],
                                strides=[1, 1],
                                padding='same',
                                activation='relu')(x)
    x = keras.layers.MaxPool2D(pool_size=[2, 2],
                               strides=[2, 2],
                               padding='same')(x)
    for _ in range(3):
        x = keras.layers.Conv2D(filters=512,
                                kernel_size=[3, 3],
                                strides=[1, 1],
                                padding='same',
                                activation='relu')(x)
    x = keras.layers.MaxPool2D(pool_size=[2, 2],
                               strides=[2, 2],
                               padding='same')(x)
    for _ in range(3):
        x = keras.layers.Conv2D(filters=512,
                                kernel_size=[3, 3],
                                strides=[1, 1],
                                padding='same',
                                activation='relu')(x)
    x = keras.layers.MaxPool2D(pool_size=[2, 2],
                               strides=[2, 2],
                               padding='same')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(4096,
                              activation='relu')(x)
    x = tf.keras.layers.Dense(4096,
                              activation='relu')(x)
    x = tf.keras.layers.Dense(1000,
                              activation='softmax')(x)
    model = keras.Model(inputs=input,
                               outputs=x)
    return model


a = vgg16()
print(a.summary())
