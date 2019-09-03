# -*- coding:utf-8 -*-
import tensorflow.compat.v1 as tf
import numpy as np
import vgg16

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.8
sess = tf.Session(config=config)
model = vgg16.vgg16()
data = np.load('vgg16_weights.npz')
key = sorted(data.keys())
for i, j in enumerate(key):
    sess.run(model.parameter[i].assign(data[j]))

saver = tf.train.Saver()
saver.save(sess, r'ckpt/vgg-ckpt')
