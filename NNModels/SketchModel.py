import tensorflow.keras as K
import tensorflow as tf


class SketchModel(K.models.Model):

    def __init__(self, z_dim=64):
        super(SketchModel, self).__init__()

        self.base = K.applications.ResNet50V2(
            include_top=False,
            weights="imagenet",
            pooling="avg",
            input_tensor=None)
        self.flat = K.layers.Flatten()
        self.dense = K.layers.Dense(z_dim, activation=None, use_bias=True)
        self.normalize = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))

    def call(self, inputs, training=None, mask=None):
        '''
        :param inputs:
        :param training:
        :param mask:
        :return:
        '''
        z = self.flat(self.base(inputs))
        z = self.normalize(self.dense(z))

        return z


class DomainDiscriminator(K.models.Model):

    def __init__(self, output=1):
        super(DomainDiscriminator, self).__init__()

        self.batch = K.layers.BatchNormalization()
        self.dense_sigmoid = K.layers.Dense(units=output, activation=None)

    def call(self, inputs, training=None, mask=None):
        '''
        :param inputs:
        :param training:
        :param mask:
        :return:
        '''
        x = self.batch(inputs)
        z = self.dense_sigmoid(x)

        return z
