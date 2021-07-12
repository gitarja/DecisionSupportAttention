import tensorflow as tf
import tensorflow.keras as K
from tensorflow.python.framework import ops


class DoubleTriplet():

    def __init__(self, margin=1., squared=False, soft=False):
        super(DoubleTriplet, self).__init__()
        self.margin = margin
        self.squared = squared
        self.soft = soft

    def __call__(self, anchor_positive, pair_positive, anchor_negative, pair_negative):
        ap = ops.convert_to_tensor_v2(anchor_positive, name="anchor_positive")
        pp = ops.convert_to_tensor_v2(pair_positive, name="pair_positive")
        an = ops.convert_to_tensor_v2(anchor_negative, name="anchor_negative")
        pn = ops.convert_to_tensor_v2(pair_negative, name="pair_negative")

        # comver tensor
        ap = tf.cast(ap, tf.float32)
        pp = tf.cast(pp, tf.float32)
        an = tf.cast(an, tf.float32)
        pn = tf.cast(pn, tf.float32)

        d_pos = tf.reduce_sum(tf.square(ap - pp), 1)  # distance between positive anchor and pair
        d_neg = tf.reduce_sum(tf.square(an - pn), 1)  # distance between negative anchor and pair
        d_pos_neg = tf.reduce_sum(tf.square(ap - an), 1)  # distance between positive and negative anchor
        d_pos_neg_pair = tf.reduce_sum(tf.square(ap - an), 1)  # distance between positive and negative anchor

        pull_in = d_neg + d_pos
        push_away = d_pos_neg + d_pos_neg_pair
        if self.soft:
            triplet_loss = tf.math.log1p(tf.math.exp((d_neg + d_pos) - (d_pos_neg + d_pos_neg_pair)))
        else:
            triplet_loss = tf.maximum(0.0, (self.margin + pull_in) - (push_away))
        # Get final mean triplet loss
        # triplet_loss = tf.reduce_mean(triplet_loss, axis=-1)

        return triplet_loss


class EntropyDoubleAnchor(K.losses.Loss):

    def __init__(self, margin=1., soft=False, reduction=tf.keras.losses.Reduction.AUTO, name='EntropyDoubleAnchorLoss'):
        super().__init__(reduction=reduction, name=name)
        self.margin = margin
        self.soft = soft

    def call(self, y_true, y_pred):
        '''
        :param y_true: pull in (d_pos_neg + d_pos_neg_pair)
        :param y_pred: (d_neg + d_pos)
        :return:
        '''
        y_pull_in = ops.convert_to_tensor_v2(y_true, name="y_pull_in")
        y_push_away = ops.convert_to_tensor_v2(y_pred, name="y_push_away")

        y_pull_in = tf.cast(y_pull_in, tf.float32)
        y_push_away = tf.cast(y_push_away, tf.float32)

        if self.soft:
            return tf.reduce_mean(tf.math.log1p(tf.math.exp(self.margin + y_push_away - y_pull_in)), -1)
        else:
            return tf.reduce_mean(tf.math.maximum(self.margin + y_pull_in - y_push_away, 0), -1)


class TripletBarlow(K.losses.Loss):
    def __init__(self, alpha=1., positive=True, reduction=tf.keras.losses.Reduction.AUTO,
                 name='TripletBarlow'):
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.positive = positive

    def call(self, y_true, y_pred):
        '''
        :param y_true: anchor
        :param y_pred: projection
        :return:
        '''
        y_true = ops.convert_to_tensor_v2(y_true, name="y_true")
        y_pred = ops.convert_to_tensor_v2(y_pred, name="y_pred")

        y_true = tf.cast(y_true, tf.float32)
        y_true_norm = (y_true - tf.reduce_mean(y_true, 0)) / tf.math.reduce_std(y_true, 0)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred_norm = (y_pred - tf.reduce_mean(y_pred, 0)) / tf.math.reduce_std(y_pred, 0)
        N, D = y_true_norm.shape
        c = tf.transpose(y_true_norm) @ y_pred_norm
        c = c / N
        I = tf.eye(D)

        if self.positive:
            c_diff = tf.square(c - I)
        else:
            c_diff = tf.square(c + I)

        c_diff = tf.where(I != 1, c_diff * self.alpha, c_diff)

        loss = tf.reduce_sum(c_diff)

        return loss

    def OffDiag(self, x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        x = tf.reshape(x, -1)[:-1]
        x = tf.reshape(x, (n - 1, n + 1))[:, 1:]
        x = tf.reshape(x, -1)
        return x


if __name__ == '__main__':
    import numpy as np

    a = np.random.normal(size=(2, 3))

    b = np.random.uniform(size=(2, 3))

    # cl = EntropyDoubleAnchor(soft=True, margin=1., reduction=tf.keras.losses.Reduction.NONE)
    cl = TripletBarlow(alpha=0.0, positive=False, reduction=tf.keras.losses.Reduction.NONE)

    print(cl(a, a))
