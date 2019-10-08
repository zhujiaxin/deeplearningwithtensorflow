import tensorflow.compat.v2 as tf
import tensorflow.keras as keras
from tensorflow.python.keras import activations


class MyDenseLayer(keras.layers.Layer):
    def __init__(self, strides, padding, activation=None, weights=None, biases=None, **kwargs):
        super(MyDenseLayer, self).__init__(
            activation=activations.get(activation),
            **kwargs)
        self.weights = tf.constant(weights)
        self.biases = tf.constant(biases)
        self.strides = strides
        self.padding = padding

    def call(self, input):
        return tf.nn.bias_add(tf.nn.conv2D(input,
                                           self.weights,
                                           padding=self.padding,
                                           strides=self.strides),
                              self.biases)

