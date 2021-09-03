import tensorflow.keras as K
import tensorflow as tf
import numpy as np
import random
import tensorflow_addons as tfa
WEIGHT_DECAY=5e-4
import math


def l2Distance(X):
    x1, x2 = X
    dis = x1 - x2
    return dis
def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))



class ConvBlock(K.models.Model):
    def __init__(self, filters):
        super(ConvBlock, self).__init__()
        self.dense = K.layers.Conv2D(filters, 3, padding="same", use_bias=True)
        self.batch = K.layers.BatchNormalization()
        self.relu = K.layers.ReLU()
        self.max = K.layers.MaxPool2D(pool_size=(2, 2), strides=2)


    def call(self, inputs,  training=None, mask=None):
        x = self.batch(self.dense(inputs), training=training)
        x = self.max(self.relu(x))
        return x




class FewShotModel(K.models.Model):

    def __init__(self, filters=64, z_dim=64):
        super(FewShotModel, self).__init__()

        self.conv_1 = ConvBlock(filters=filters)
        self.conv_2 = ConvBlock(filters=filters)
        self.conv_3 = ConvBlock(filters=filters)
        self.conv_4 = ConvBlock(filters=filters)

        #projector
        self.dense = K.layers.Dense(z_dim, activation=None, use_bias=False)
        self.normalize = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))
        self.flat = K.layers.Flatten()
        self.relu = tf.keras.layers.ReLU()
        self.batch = K.layers.BatchNormalization()



    def call(self, inputs, training=None, mask=None):
        z = self.conv_1(inputs, training=training)
        z = self.conv_2(z, training=training)
        z = self.conv_3(z, training=training)
        z = self.conv_4(z, training=training)
        z =  self.batch(self.flat(z), training=training)
        z = self.normalize(self.dense(z))
        return z

    def forward_pos(self, inputs, training=None):
        x = self.data_aug_pos(inputs)
        z = self.call(x, training=training)
        return z
    def forward_normalize(self, inputs):
        z = self.normalize(self.call(inputs, training=False))
        return z
    def forward_neg(self, inputs, training=None):
        x = self.data_aug_neg(inputs)
        z = self.call(x, training=training)

        return z



class DeepMetric(K.models.Model):

    def __init__(self, filters=256, output=1):
        super(DeepMetric, self).__init__()

        self.dist = K.layers.Lambda(euclidean_distance)
        self.batch = K.layers.BatchNormalization()
        self.dense_sigmoid = K.layers.Dense(units=1, activation="sigmoid")

    def call(self, inputs, training=None, mask=None):
        '''
        :param inputs:
        :param training:
        :param mask:
        :return:
        '''
        x = self.batch(self.dist(inputs))
        z = self.dense_sigmoid(x)

        return z
