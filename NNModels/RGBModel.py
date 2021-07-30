import tensorflow.keras as K
import tensorflow as tf


class RGBModel(K.models.Model):

    def __init__(self, z_dim=64, model="person"):
        super(RGBModel, self).__init__()
        if model == "person":
            self.base = K.applications.ResNet50V2(
                include_top=False,
                weights="imagenet",
                pooling="avg",
                input_tensor=None)


        self.flat = K.layers.Flatten()
        self.dense = K.layers.Dense(z_dim, activation=None, use_bias=True)
        self.normalize = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))
        self.data_augment = K.models.Sequential([
            K.layers.experimental.preprocessing.Resizing(256, 128),
            # K.layers.experimental.preprocessing.RandomZoom(0.1),
            K.layers.experimental.preprocessing.RandomFlip("horizontal"),
            # K.layers.experimental.preprocessing.RandomRotation(0.1)
        ])



    def call(self, inputs, training=None, mask=None):
        '''
        :param inputs:
        :param training:
        :param mask:
        :return:
        '''
        images = self.data_augment(inputs, training=training)
        z = self.flat(self.base(images))
        z = self.normalize(self.dense(z))

        return z



class DomainDiscriminator(K.models.Model):

    def __init__(self, output=1):
        super(DomainDiscriminator, self).__init__()

        self.batch = K.layers.BatchNormalization()
        self.dense = K.layers.Dense(units=128, activation=None)
        self.dense_sigmoid = K.layers.Dense(units=output, activation=None)
        self.relu = K.layers.ReLU

    def call(self, inputs, training=None, mask=None):
        '''
        :param inputs:
        :param training:
        :param mask:
        :return:
        '''
        x = self.dense(inputs)
        x = self.batch(x)
        z = self.dense_sigmoid(x)

        return z
