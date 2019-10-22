import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras import activations


class MyDenseLayer(keras.layers.Layer):
    def __init__(self, weights=None, biases=None, activation=None, **kwargs):
        super(MyDenseLayer, self).__init__(**kwargs)
        self.kernel = tf.Variable(weights)
        self.bias = tf.Variable(biases)
        self.activation = activations.get(activation)

    def call(self, inputs):
        out = tf.nn.bias_add(tf.matmul(inputs,
                                       self.kernel),
                             self.bias)
        if self.activation is None:
            return out
        else:
            return self.activation(out)


class MyConv2DLayer(keras.layers.Layer):
    def __init__(self, kernels=None, biases=None, padding='SAME', strides=(1, 1), activation=None,  **kwargs):
        super(MyConv2DLayer, self).__init__(**kwargs)
        self.kernel = tf.Variable(kernels)
        self.bias = tf.Variable(biases)
        self.padding = padding
        self.strides = strides
        self.activation = activations.get(activation)

    def call(self, inputs):
        out = tf.nn.conv2d(inputs, self.kernel, padding=self.padding, strides=self.strides)
        if self.activation is None:
            return out
        else:
            return self.activation(out)

