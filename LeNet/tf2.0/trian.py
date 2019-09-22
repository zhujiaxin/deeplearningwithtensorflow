# -*- coding:utf-8 -*-

import tensorflow as tf
import lenet
import utils.read_mnist as read_mnist

train_x, train_y, test_x, test_y = read_mnist.read_mnist(one_hot=True, standard=True)

model = lenet.model(regular=0)
model.compile(optimizer=tf.keras.optimizers.SGD(),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=['accuracy'])

#date = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#trainlogdir=r"trainlogs3\\"
#scalarlogdir = r'trainlogs3\\'
#tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=trainlogdir)
#file_writer = tf.summary.create_file_writer(scalarlogdir)
#file_writer.set_as_default()


def lr_schedule(epoch):
    """Returns a custom learning rate that decreases as epochs progress.    """
    learning_rate = 0.02
    if epoch > 2:
        learning_rate = 0.0002
    if epoch > 5:
        learning_rate = 0.0001
    if epoch > 8:
        learning_rate = 0.00005
    if epoch > 12:
        learning_rate = 0.00001

    #tf.summary.scalar('learning rate', data=learning_rate, step=epoch)
    return learning_rate


lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)
earlystops = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3)
logger = tf.keras.callbacks.CSVLogger('alnet.csv')
callbacks = [#tensorboard_callback,
             earlystops, lr_callback, logger]

history = model.fit(x=train_x, y=train_y,
                    batch_size=256,
                    epochs=20,
                    validation_data=(test_x, test_y),
                    shuffle=True,
                    callbacks=callbacks)


print(history.history)
