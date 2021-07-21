import tensorflow.keras as K
import tensorflow as tf
import numpy as np
import random
import tensorflow_addons as tfa
WEIGHT_DECAY=5e-4
import math
class ConvBlock(K.layers.Layer):
    def __init__(self, filters):
        super(ConvBlock, self).__init__()
        self.dense = K.layers.Conv2D(filters, 3, padding="same", use_bias=False)
        self.batch = K.layers.BatchNormalization()
        self.relu = K.layers.ReLU()
        self.max = K.layers.MaxPool2D(pool_size=(2, 2))

    def call(self, inputs, **kwargs):
        x = self.batch(self.dense(inputs))
        x = self.max(self.relu(x))
        return x


class ProjectBlock(K.layers.Layer):
    def __init__(self, filters):
        super(ProjectBlock, self).__init__()
        self.dense = K.layers.Dense(filters,  activation=None, use_bias=False)
        self.batch = K.layers.BatchNormalization()
        self.relu = K.layers.ReLU()


    def call(self, inputs, **kwargs):
        x = self.batch(self.dense(inputs))
        x = self.relu(x)
        return x

class FewShotModel(K.models.Model):


    def __init__(self, filters=64, z_dim=64):
        super(FewShotModel, self).__init__()

        self.conv_1 = ConvBlock(filters=filters)
        self.conv_2 = ConvBlock(filters=filters)
        self.conv_3 = ConvBlock(filters=filters)
        self.conv_4 = ConvBlock(filters=filters)

        #projector
        self.project = ProjectBlock(filters=1024)
        self.dense = K.layers.Dense(z_dim, activation=None, use_bias=False)
        self.normalize = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))
        self.flat = K.layers.Flatten()
        self.relu = tf.keras.layers.ReLU()

        self.random_zoomout = tf.keras.layers.experimental.preprocessing.RandomZoom((-0.1, 0.1))
        self.random_zoomout_neg = tf.keras.layers.experimental.preprocessing.RandomZoom((-0.5, 0.5))




    def call(self, inputs, training=None, mask=None):
        z = self.conv_1(inputs)
        z = self.conv_2(z)
        z = self.conv_3(z)
        z = self.conv_4(z)
        z = self.project(self.flat(z))
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


    def data_aug_pos(self, x):
        deg = tf.random.uniform([len(x)], .1, 15.)
        optional = tf.random.uniform([len(x)], 0, 1, dtype=tf.int32)
        rad = ((tf.cast(optional, tf.float32) * 345.) + deg) * (math.pi / 180)
        x = tfa.image.rotate(x, rad, fill_mode="nearest")

        return x

    def data_aug_neg(self, x):
        deg =tf.random.uniform([len(x)], 90., 180.) * (math.pi / 180)
        x = tfa.image.rotate(x, deg, fill_mode="nearest")
        x = tf.image.random_crop(x, (len(x), 28, 28, 1))
        x = self.random_zoomout_neg(x)

        return x

class DeepMetric(K.models.Model):

    def __init__(self, filters=128, dropout_rate=0.3, output=1):
        super(DeepMetric, self).__init__()


        self.dense_logit = K.layers.Dense(output, activation=None)

    def call(self, inputs, training=None, mask=None):
        '''
        :param inputs:
        :param training:
        :param mask:
        :return:
        '''
        X = K.backend.concatenate(inputs, -1)

        z = self.dense_logit(X)

        return z