import tensorflow.keras as K
import tensorflow as tf

class ConvBlock(K.layers.Layer):
    def __init__(self, filters):
        super(ConvBlock, self).__init__()
        self.dense = K.layers.Conv2D(filters, 3, padding="same")
        self.batch =  K.layers.BatchNormalization()
        self.relu = K.layers.ReLU()
        self.max = K.layers.MaxPool2D(pool_size=(2, 2))

    def call(self, inputs, **kwargs):
        x = self.batch(self.dense(inputs))
        x = self.max(self.relu(x))
        return x



class FewShotModel(K.models.Model):


    def __init__(self, filters=64, z_dim=64):
        super(FewShotModel, self).__init__()

        self.block_1 = ConvBlock(filters)
        self.block_2 = ConvBlock(filters)
        self.block_3 = ConvBlock(filters)
        self.block_4 = ConvBlock(z_dim)
        self.dense = K.layers.Dense(z_dim, activation=None)
        self.normalize = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))


        self.flat = K.layers.Flatten()




    def call(self, inputs, training=None, mask=None):
        z = self.block_1(inputs)
        z = self.block_2(z)
        z = self.block_3(z)
        z = self.block_4(z)
        z = self.dense(self.flat(z))
        z = self.normalize(z)
        return z