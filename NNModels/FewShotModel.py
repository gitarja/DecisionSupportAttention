import tensorflow.keras as K
import tensorflow as tf
import numpy as np
import random
import tensorflow_addons as tfa

WEIGHT_DECAY = 5e-4
import math




class ConvBlock(K.models.Model):
    def __init__(self, filters):
        super(ConvBlock, self).__init__()
        self.dense = K.layers.Conv2D(filters, 3, padding="same", use_bias=True)
        self.batch = K.layers.BatchNormalization()
        self.relu = K.layers.ReLU()
        self.max = K.layers.MaxPool2D(pool_size=(2, 2), strides=2)

    def call(self, inputs, training=None, mask=None):
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

        # projector
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
        z = self.batch(self.flat(z))
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


