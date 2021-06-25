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

        self.base = K.applications.ResNet50(include_top=False)

        self.dense = K.layers.Dense(z_dim, activation=None)
        self.normalize = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))


        self.flat = K.layers.Flatten()




    def call(self, inputs, training=None, mask=None):
        z = self.base(inputs)
        z = self.dense(self.flat(z))
        z = self.normalize(z)
        return z