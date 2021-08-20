import tensorflow.keras as K
import tensorflow as tf
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class BaseLine(K.models.Model):
    def build_model(self, hp):

        model = K.models.Sequential()
        model.add(K.layers.Dense(hp.Int('units_1',
                                        min_value=32,
                                        max_value=128,
                                        step=32), activation=None, name="dense1"))
        model.add(K.layers.BatchNormalization())
        model.add(K.layers.ELU())
        model.add(K.layers.Dense(hp.Int('units_2',
                                        min_value=32,
                                        max_value=128,
                                        step=32), activation=None, name="dense2"))
        model.add(K.layers.BatchNormalization())
        model.add(K.layers.ELU())
        model.add(K.layers.Dense(hp.Int('units_3',
                                        min_value=32,
                                        max_value=256,
                                        step=32), activation=None, name="dense3"))
        model.add(K.layers.BatchNormalization())
        model.add(K.layers.ELU())

        model.add(K.layers.Dense(units=1, activation="sigmoid"))
        model.compile(
            optimizer=K.optimizers.Adamax(
                hp.Choice('learning_rate',
                          values=[1e-2, 1e-3, 1e-4, 1e-5])),
            loss='binary_crossentropy',
            metrics=['accuracy'])

        return model

class ProjectBlock(K.layers.Layer):
    def __init__(self, filters):
        super(ProjectBlock, self).__init__()
        self.dense = K.layers.Dense(filters,  activation=None, use_bias=False)
        self.batch = K.layers.BatchNormalization()
        self.relu = K.layers.ELU()


    def call(self, inputs, **kwargs):
        x = self.batch(self.dense(inputs))
        x = self.relu(x)
        return x
class AttentionModel(K.models.Model):


    def __init__(self, filters=64, z_dim=64):
        super(AttentionModel, self).__init__()

        # self.dense1 = K.layers.Dense(units=filters, activation="elu")
        # self.dense2 = K.layers.Dense(units=filters, activation="elu")
        self.dense1 = ProjectBlock(filters=filters)
        self.dense2 = ProjectBlock(filters=filters)



        self.dense3 = K.layers.Dense(units=z_dim, activation=None)
        self.normalize = K.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))


        #dropout
        self.flat = K.layers.Flatten()
        self.noise = K.layers.GaussianDropout(0.3)
        #anchor
        self.typical_anchor = None
        self.asd_anchor = None
        self.adult_anchor = None



    def call(self, inputs, training=None, mask=None):
        # inputs = self.noise(inputs, training)
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.normalize(x)

        return x

    def computeDistance(self, inputs, support):
        z_inputs = self(inputs, training=False) #compute embedding distance
        cosine_sim = cosine_similarity(z_inputs.numpy(), support)
        cosine_sim[cosine_sim<-1] = -1
        cosine_sim[cosine_sim>1] = 1
        angular_sim = 1 - (tf.acos(cosine_sim) / math.pi)
        return angular_sim


    def predictClass(self, inputs):

        pos = self.computeDistance(inputs, self.typical_anchor)
        neg = self.computeDistance(inputs, self.asd_anchor)
        if self.adult_anchor is None:
            return np.average(pos.numpy()), np.average(neg.numpy())
        else:
            adl = self.computeDistance(inputs, self.adult_anchor)
            return np.average(pos.numpy()), np.average(neg.numpy()), np.average(adl.numpy())





    def setSupports(self, typical, asd, adult=None):
        self.typical_anchor = typical.numpy()
        self.asd_anchor = asd.numpy()
        self.adult_anchor = adult