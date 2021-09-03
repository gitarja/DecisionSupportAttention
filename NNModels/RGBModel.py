import tensorflow.keras as K
import tensorflow as tf
from tensorflow.python.keras.applications import inception_resnet_v2

_RGB_MEAN = [123.68, 116.78, 103.94]

class ResnetIdentityBlock(tf.keras.Model):
  def __init__(self, kernel_size, filters, kernel_size2=1):
    super(ResnetIdentityBlock, self).__init__(name='')
    filters1, filters2, filters3 = filters

    self.conv2a = tf.keras.layers.Conv2D(filters1, (kernel_size2, kernel_size2))
    self.bn2a = tf.keras.layers.BatchNormalization()

    self.conv2b = tf.keras.layers.Conv2D(filters2, kernel_size, padding='same')
    self.bn2b = tf.keras.layers.BatchNormalization()

    self.conv2c = tf.keras.layers.Conv2D(filters3, (kernel_size2, kernel_size2))
    self.bn2c = tf.keras.layers.BatchNormalization()

  def call(self, input_tensor, training=False):
    x = self.conv2a(input_tensor)
    x = self.bn2a(x, training=training)
    x = tf.nn.leaky_relu(x, 0.3)

    x = self.conv2b(x)
    x = self.bn2b(x, training=training)
    x = tf.nn.leaky_relu(x, 0.3)

    x = self.conv2c(x)
    x = self.bn2c(x, training=training)

    x += input_tensor
    return tf.nn.leaky_relu(x, 0.3)

class LuNet(K.models.Model):
    def __init__(self):
        super(LuNet, self).__init__()
        self.conv1 = K.layers.Conv2D(filters=128, kernel_size=7, strides=3, activation=None, padding="same")
        self.conv2 = K.layers.Conv2D(filters=256, kernel_size=1, strides=1, activation=None, padding="same")
        self.conv3 = K.layers.Conv2D(filters=512, kernel_size=1, strides=1, activation=None, padding="same")
        self.res_11 = ResnetIdentityBlock(3, [128, 32, 128])

        #block2
        self.res_21 = ResnetIdentityBlock(3, [128, 32, 128])
        self.res_22 = ResnetIdentityBlock(3, [128, 32, 128])
        self.res_23 = ResnetIdentityBlock(3, [128, 64, 128])

        #block3
        self.res_31 = ResnetIdentityBlock(3, [256, 64, 256])
        self.res_32 = ResnetIdentityBlock(3, [256, 64, 256])

        #block4
        self.res_41 = ResnetIdentityBlock(3, [256, 64, 256])
        self.res_42 = ResnetIdentityBlock(3, [256, 64, 256])
        self.res_43 = ResnetIdentityBlock(3, [256, 128, 256])

        # block5
        self.res_51 = ResnetIdentityBlock(3, [512, 128, 512])
        self.res_52 = ResnetIdentityBlock(3, [512, 128, 512])

        # block6
        self.res_61 = ResnetIdentityBlock(3, [512, 128, 512], 1)

        self.max_pool = K.layers.MaxPooling2D(pool_size=3, strides=2, padding="same")

        #batchnorm
        self.batch_norm1 = K.layers.BatchNormalization()
        self.batch_norm2 = K.layers.BatchNormalization()
        self.batch_norm3 = K.layers.BatchNormalization()



    def call(self, inputs, training=None, mask=None):
        #restblock 1
        x = self.batch_norm1(self.conv1(inputs))
        x = tf.nn.leaky_relu(x, 0.3)
        x = self.res_11(x)
        x = self.max_pool(x)

        #restblock2
        x = self.res_21(x)
        x = self.res_22(x)
        x = self.res_23(x)
        x = self.max_pool(x)
        x = self.batch_norm2(self.conv2(x))
        x = tf.nn.leaky_relu(x, 0.3)

        # restblock3
        x = self.res_31(x)
        x = self.res_32(x)
        x = self.max_pool(x)

        #restblock4
        x = self.res_41(x)
        x = self.res_42(x)
        x = self.res_43(x)
        x = self.max_pool(x)
        x = self.batch_norm3(self.conv3(x))
        x = tf.nn.leaky_relu(x, 0.3)

        # restblock5
        x = self.res_51(x)
        x = self.res_52(x)
        x = self.max_pool(x)

        # restblock5
        z = self.res_61(x)

        return z

class RGBModel(K.models.Model):

    def __init__(self, z_dim=64, model="res_net"):
        super(RGBModel, self).__init__()
        if model == "res_net":
            base_model = K.applications.InceptionResNetV2(
                include_top=False,
                weights=None,
                pooling=None,
                input_tensor=None)

        else:
            base_model = LuNet()

        self.base = base_model
        self.flat = K.layers.Flatten()
        # self.dense0 = K.layers.Dense(1024, activation=None, use_bias=True)
        # self.batch0 = K.layers.BatchNormalization()

        self.dense = K.layers.Dense(z_dim, activation=None, use_bias=True)
        self.normalize = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=-1))

        self.relu = K.layers.ReLU()
        # self.data_augment = K.models.Sequential([
        #     K.layers.experimental.preprocessing.RandomFlip("horizontal"),
        #     K.layers.experimental.preprocessing.Resizing(int(256 * 1.125), int(128 * 1.125)),
        #     K.layers.experimental.preprocessing.RandomCrop(256, 128)
        # ])
        #
        # self.resize = K.layers.experimental.preprocessing.Resizing(256, 128)





    def call(self, inputs, training=None, mask=None):
        '''
        :param inputs:
        :param training:
        :param mask:
        :return:
        '''
        # if training:
        #     images = self.data_augment(inputs, training=training)
        # else:
        #     images = self.resize(inputs)
        # images = images - tf.constant(_RGB_MEAN, dtype=tf.float32, shape=(1, 1, 1, 3))
        images = inception_resnet_v2.preprocess_input(inputs)
        z = self.flat(self.base(images))
        # z = self.relu(self.batch0(self.dense0(z)))
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