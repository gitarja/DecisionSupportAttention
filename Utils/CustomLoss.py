import tensorflow as tf
import tensorflow.keras as K
from tensorflow.python.framework import ops


def off_diagonal(x):
    n = tf.shape(x)[0]
    flattened = tf.reshape(x, [-1])[:-1]
    off_diagonals = tf.reshape(flattened, (n-1, n+1))[:, 1:]
    return tf.reshape(off_diagonals, [-1])
def normalize(x):
        return (x - tf.reduce_mean(x, 0)) / tf.math.reduce_std(x, 0)

def lip_dist(x1, x2):
    r = x1 - x2
    return -tf.reduce_sum(tf.math.log((4 * tf.math.exp(r)) / tf.square(1+tf.math.exp(r))), 1)
class DoubleTriplet():

    def __init__(self, margin=1., soft=False, squared=False):
        super(DoubleTriplet, self).__init__()
        self.margin = margin
        self.soft = soft
        self.squared = squared


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
        # d_pos_neg = tf.reduce_sum(tf.square(ap-an), 1)  # distance between positive and negative anchor
        d_pos_neg = tf.reduce_sum(tf.square(((ap + pp)/2.) - ((an + pn)/2.)), 1)  # distance between positive and negative anchor


        pull_in = (d_neg + d_pos)/2.
        push_away = d_pos_neg

        if self.soft:
            triplet_loss = tf.math.log1p(tf.math.exp(pull_in - push_away))
        elif self.squared:
            triplet_loss = tf.square(tf.maximum(0.0, (self.margin + pull_in) - (push_away)))
        else:
            triplet_loss = tf.maximum(0.0, (self.margin + pull_in) - (push_away))




        return triplet_loss


class TripletBarlow():

    def __init__(self, margin=1.,alpha=1e-1):
        super(TripletBarlow, self).__init__()
        self.margin = margin
        self.alpha = alpha


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

        # normalize data
        N,D = ap.shape
        ap_norm = normalize(ap)
        pp_norm = normalize(pp)
        an_norm = normalize(an)
        pn_norm = normalize(pn)
        I= tf.eye(D)

        pos_c = (tf.transpose(ap_norm) @ pp_norm) / N
        neg_c = (tf.transpose(an_norm) @ pn_norm) / N
        pos_neg = (tf.transpose((an_norm + pp_norm) / 2.) @ (an_norm + pn_norm)/2) / N

        c_diff = tf.square(pos_c - I) + tf.square(neg_c - I) + tf.square(pos_neg)
        c_diff = tf.where(I != 1, c_diff * self.alpha, c_diff)
        #
        # d_pos_neg = tf.reduce_sum(tf.square(((ap + pp) / 2.) - ((an + pn) / 2.)),
        #                           1)  # distance between positive and negative anchor
        #
        # push_away = d_pos_neg
        #
        # triplet_loss = tf.square(tf.maximum(0.0, self.margin - (push_away)))

        return  tf.reduce_sum(c_diff)


class TripletOffline():

    def __init__(self, margin=1., squared=False, soft=False):
        super(TripletOffline, self).__init__()
        self.margin = margin
        self.squared = squared
        self.soft = soft



    def __call__(self, anchor_positive, pair_positive, anchor_negative):
        ap = ops.convert_to_tensor_v2(anchor_positive, name="anchor_positive")
        pp = ops.convert_to_tensor_v2(pair_positive, name="pair_positive")
        an = ops.convert_to_tensor_v2(anchor_negative, name="anchor_negative")

        # comver tensor
        ap = tf.cast(ap, tf.float32)
        pp = tf.cast(pp, tf.float32)
        an = tf.cast(an, tf.float32)


        d_pos = tf.reduce_sum(tf.square(ap - pp), 1)  # distance between positive anchor and pair
        d_pos_neg = tf.reduce_sum(tf.square(ap - an), 1)  # distance between positive and negative anchor


        pull_in = d_pos
        push_away = d_pos_neg

        triplet_loss = tf.maximum(0.0, (self.margin + pull_in) - (push_away))
        # Get final mean triplet loss
        triplet_loss = tf.reduce_mean(triplet_loss, axis=-1)

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

class  DoubleBarlow:
        def __init__(self, alpha=5e-3,
                     name='DoubleBarlow'):
            super(DoubleBarlow, self).__init__()
            self.alpha = alpha




        def __call__(self, anchor, positive, negative):
            '''
            :param y_true: anchor
            :param y_pred: projection
            :return:
            '''
            anchor = ops.convert_to_tensor_v2(anchor, name="anchor")
            positive = ops.convert_to_tensor_v2(positive, name="positive")
            negative = ops.convert_to_tensor_v2(negative, name="negative")

            anchor = tf.cast(anchor, tf.float32)
            anchor_norm = normalize(anchor)
            positive = tf.cast(positive, tf.float32)
            positive_norm = normalize(positive)
            negative = tf.cast(negative, tf.float32)
            negative_norm = normalize(negative)


            N, D = anchor.shape
            c_ap = (tf.transpose(anchor_norm) @ positive_norm) / N
            c_an = (tf.transpose(anchor_norm) @ negative_norm) / N
            I = tf.eye(D)


            #positive loss
            c_diff = tf.math.log(tf.math.cosh(c_ap - I)) + tf.math.log(tf.math.cosh(c_an))
            c_diff = tf.where(I != 1, c_diff * self.alpha, c_diff)

            #negative loss
            # d_neg = tf.reduce_sum(tf.square(anchor - negative), 1)
            # negative_loss = tf.reduce_mean(tf.math.log1p(tf.math.exp(-d_neg)), -1)
            loss = tf.reduce_mean(c_diff)

            return loss

class BarlowTwins(K.losses.Loss):
    def __init__(self, alpha=5e-3, beta=3., positive=True, reduction=tf.keras.losses.Reduction.AUTO,
                 name='BarlowTwins'):
        super().__init__(reduction=reduction, name=name)
        self.alpha = alpha
        self.beta = beta
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
        y_true_norm = normalize(y_true)
        y_pred = tf.cast(y_pred, tf.float32)
        y_pred_norm = normalize(y_pred)
        N, D = y_true_norm.shape
        c = tf.transpose(y_true_norm) @ y_pred_norm
        c = c / N
        I = tf.eye(D)

        c_diff = tf.square(c - I)
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

    a = np.random.normal(size=(5, 3))

    b = np.random.normal(size=(5, 3))

    # cl = EntropyDoubleAnchor(soft=True, margin=1., reduction=tf.keras.losses.Reduction.NONE)
    cl = DoubleBarlow()

    print(cl(a, a, b))
