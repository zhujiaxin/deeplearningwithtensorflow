# -*- coding:utf-8 -*-
import tensorflow.compat.v1 as tf
import lenet
import utils.read_mnist as read_mnist
import time
import os
import numpy as np


train_x, train_y, test_x, test_y = read_mnist.read_mnist(one_hot=True, standard=True)
index_train = np.random.permutation(train_x.shape[0])
train_x, train_y = train_x[index_train], train_y[index_train]

lr_rate = 0.01
batch_size = 64
sess = tf.Session()
model = lenet.LeNet(lr_rate=0.001, regular=0.0005, train=True)
sess.run(tf.global_variables_initializer())
tensorboard_dir = r"tensorboardlog/"
if os.path.exists(tensorboard_dir):
    for file in os.listdir(tensorboard_dir):
        path_file = os.path.join(tensorboard_dir, file)
        os.remove(path_file)
file_writer = tf.summary.FileWriter(tensorboard_dir, sess.graph)
tf.summary.scalar("loss of test data", model.loss)
tf.summary.scalar("accuracy on test data", model.accuracy)
tf.summary.scalar('learning rate ', tf.reduce_mean(model.lr_rate))
merge = tf.summary.merge_all()
print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
epochs = 100
saver = tf.train.Saver()
for epoch in range(epochs):
    start = time.time()
    batch_sofar = 0
    for j in range(train_x.shape[0] // batch_size + 1):
        model.make_train(train_x=train_x[batch_sofar:batch_sofar + batch_size, :, :, :],
                         train_y=train_y[batch_sofar:batch_sofar + batch_size, :],
                         session=sess)
        batch_sofar += batch_size
        if batch_sofar > train_x.shape[0]:
            model.make_train(train_x=train_x[batch_sofar - batch_size:, :, :, :],
                             train_y=train_y[batch_sofar - batch_size:, :],
                             session=sess)
    if (epoch + 1) % 20 == 0:
        lr_rate = lr_rate / 2
        model.change_lr_rate(lr_rate)
    print('lr_rate', model.lr_rate)
    rs = sess.run(merge, feed_dict={model.input_x: test_x, model.input_y: test_y})
    file_writer.add_summary(rs, epoch)
    print('{:.0f} epoch use {:.2f} ms'.format(epoch, (time.time()-start)*1000))
    print('loss on test data', model.compute_loss(train_x, train_y, session=sess))
    print('accuracy', model.compute_accuracy(train_x, train_y, session=sess))
saver.save(sess, r'ckpt/lenet-ckpt', global_step=epoch)
sess.close()

