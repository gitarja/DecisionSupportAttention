from tensorflow import keras as K
import tensorflow as tf
import math
import numpy as np
class BaseLine(K.models.Model):
    def build_model(self, hp):

        model = K.models.Sequential()
        model.add(K.layers.Dense(hp.Int('units_1',
                                        min_value=32,
                                        max_value=128,
                                        step=32), activation="elu", name="dense1"))
        model.add(K.layers.Dropout(hp.Choice('dropout_rate1',
                      values=[0.1, 0.35, 0.5, 0.75])))
        model.add(K.layers.Dense(hp.Int('units_2',
                                        min_value=32,
                                        max_value=128,
                                        step=32), activation="elu", name="dense2"))
        model.add(K.layers.Dropout(hp.Choice('dropout_rate2',
                                             values=[0.1, 0.35, 0.5, 0.75])))
        model.add(K.layers.Dense(hp.Int('units_3',
                                        min_value=64,
                                        max_value=256,
                                        step=32), activation="elu", name="dense3"))
        model.add(K.layers.Dropout(hp.Choice('dropout_rate3',
                                             values=[0.1, 0.35, 0.5, 0.75])))

        model.add(K.layers.Dense(units=1, activation="sigmoid"))
        model.compile(
            optimizer=K.optimizers.Adamax(
                hp.Choice('learning_rate',
                          values=[1e-2, 1e-3, 1e-4])),
            loss='binary_crossentropy',
            metrics=['accuracy'])

        return model
class SiameseModel(K.models.Model):


    def __init__(self, output=2):
        super(SiameseModel, self).__init__()

        self.dense1 = K.layers.Dense(units=128, activation="elu")
        self.dense2 = K.layers.Dense(units=128, activation="elu")
        self.dense3 = K.layers.Dense(units=64, activation=None)
        self.normalize = K.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))


        #dropout
        self.dropout = K.layers.Dropout(0.35)

        #metrics
        self.cosine_sim = K.losses.CosineSimilarity(reduction=K.losses.Reduction.NONE)


        #anchor
        self.typical_anchor = None
        self.asd_anchor = None



    def call(self, inputs, training=None, mask=None):

        x = self.dense1(inputs)
        x = self.dropout(x, training=training)
        x = self.dense2(x)
        x = self.dropout(x, training=training)
        x = self.dense3(x)
        x = self.normalize(x)

        return x

    def computeDistance(self, inputs, anchor):
        z_inputs = self(inputs, training=False) #compute embedding distance
        sim = self.cosine_sim(z_inputs, anchor)
        rep_val = tf.constant(-1.0, dtype=tf.float32)
        sim = tf.where(sim < -1.0, rep_val, sim)
        return self.angularSimilarity(sim)


    def predictClass(self, inputs):

        pos = self.computeDistance(inputs, self.typical_anchor)
        neg = self.computeDistance(inputs, self.asd_anchor)

        return np.max(pos.numpy()), np.max(neg.numpy())


    def angularSimilarity(self, cosine_sim):
        angular_dist = tf.acos(cosine_sim) / math.pi
        return angular_dist
        # angular_sim = angular_dist
        # return angular_sim

    def computeWeight(self, anchor_x, positive_x, negative_x, margin=0.5):
        anchor = self(anchor_x, training=False)
        positive = self(positive_x, training=False)
        negative = self(negative_x, training=False)

        d_pos = tf.reduce_sum(tf.square(anchor - positive), 1)
        d_neg = tf.reduce_sum(tf.square(anchor - negative), 1)

        weights = tf.ones(anchor.shape[0]) * 1e-1
        one_val = tf.constant(1., dtype=tf.float32)
        final_weights = tf.where(tf.abs(d_pos - d_neg) <= margin, one_val, weights)

        return final_weights

    def tripletOffline(self, anchor_output, positive_output, negative_output, margin=0.5, sample_weights=None):
        d_pos = tf.reduce_sum(tf.square(anchor_output - positive_output), 1)
        d_neg = tf.reduce_sum(tf.square(anchor_output - negative_output), 1)


        loss = tf.maximum(0.0, d_pos - d_neg + margin)
        if sample_weights is None:
            loss = tf.reduce_mean(loss)
        else:
            loss = tf.reduce_mean(loss * sample_weights)

        return loss

    def doubleTriplet(self, anchor_output, positive_output, negative_anchor, negative_output, margin=0.5):
        d_pos = tf.reduce_sum(tf.square(anchor_output - positive_output), 1)
        d_neg = tf.reduce_sum(tf.square(anchor_output - negative_anchor), 1)
        d_neg_neg = tf.reduce_sum(tf.square(negative_anchor - negative_output), 1)


        loss = tf.maximum(0.0, (d_pos + d_neg_neg) - d_neg+margin)
        loss = tf.reduce_mean(loss)

        return loss


    def setAnchor(self, typical, asd):
        self.typical_anchor = typical
        self.asd_anchor = asd