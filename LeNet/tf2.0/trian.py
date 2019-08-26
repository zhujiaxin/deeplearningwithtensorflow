# -*- coding:utf-8 -*-

import tensorflow as tf
import lenet
import utils.read_mnist as read_mnist
import datetime as datetime


train_x, train_y, test_x, test_y = read_mnist.read_mnist(one_hot=True, z_score=True)

model = lenet.model()
model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])


logdir=r"trainlogs2\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
#logdir = "testlogs"
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
model.fit(x=train_x, y=train_y,
          batch_size=256,
          epochs=6,
          #validation_data=(test_x, test_y),
          shuffle=True,
          callbacks=[tensorboard_callback])


