import tensorflow as tf
import tensorflow.keras as K
from tensorflow.python.framework import ops
import tensorflow_probability as tfp


def off_diagonal(x):
    n = tf.shape(x)[0]
    flattened = tf.reshape(x, [-1])[:-1]
    off_diagonals = tf.reshape(flattened, (n - 1, n + 1))[:, 1:]
    return tf.reshape(off_diagonals, [-1])


def normalize(x):
    return (x - tf.reduce_mean(x, 0)) / tf.math.reduce_std(x, 0)

def l2_dis(x1, x2, w=None):
    if w is not None:
        return tf.math.sqrt(tf.reduce_sum(tf.math.square(x1 - x2)/w, -1))
    else:
        return tf.sqrt(tf.reduce_sum(tf.multiply(x1, x1), -1) + tf.reduce_sum(tf.multiply(x2, x2), -1) - 2 * tf.reduce_sum(tf.multiply(x1, x2), -1))

def l1_dis(x1, x2):
    return tf.reduce_sum(tf.math.abs(x1 - x2), -1)

def minc_dis(x1, x2, p):
    return tf.math.pow(tf.reduce_sum(tf.math.pow(tf.math.abs(x1 - x2), p), -1), 1./p)


def log_cosh_dis(x1, x2):
    return tf.reduce_sum(tf.math.log(tf.math.cosh(x1 - x2)), -1)


class CentroidTripletSketch():

    def __init__(self, margin=0.5, soft=False, n_shots=5, mean=False):
        super(CentroidTripletSketch, self).__init__()
        self.margin = margin
        self.n_shots = n_shots
        self.soft = soft
        self.mean = mean

    def __call__(self, sketch_embd, embed, n_class=5):
        sketch_embd = ops.convert_to_tensor_v2(sketch_embd, name="sketch_embd")
        embed = ops.convert_to_tensor_v2(embed, name="embed")
        # comver tensor
        embed = tf.cast(embed, tf.float32)
        N_s, D_s = sketch_embd.shape
        N_e, D_e = embed.shape
        N = N_s + N_e
        # res_embed = tf.concat([sketch_embd, tf.reshape(embed, (n_class, self.n_shots, D))], axis=-1)
        res_embed = tf.concat([tf.expand_dims(sketch_embd, 1), tf.reshape(embed, (n_class, self.n_shots, D_s))], axis=1)

        # centroids = tf.reduce_mean(res_embed, 1, keepdims=True)
        shots_add = self.n_shots + 1
        if self.mean:
            centroids = tf.reduce_mean(res_embed, 1, keepdims=True)
            dist_to_cens = tf.reshape(tf.reduce_sum(tf.square(tf.expand_dims(res_embed, 1) - centroids), -1),
                                      (n_class, n_class * self.n_shots))
        else:
            centroids = tfp.stats.percentile(res_embed, 50.0, 1, keepdims=True)
            if self.soft:
                dist_to_cens = tf.reshape(
                    tf.reduce_sum(tf.math.log(tf.math.cosh(tf.expand_dims(res_embed, 1) - centroids)), -1),
                    (n_class, n_class * shots_add))

            else:
                dist_to_cens = tf.reshape(tf.reduce_sum(tf.square(tf.expand_dims(res_embed, 1) - centroids), -1),
                                          (n_class, n_class * shots_add))

            d = tf.reshape(tf.repeat(tf.eye(n_class), shots_add), (n_class, n_class * shots_add))
            inner_dist = tf.reshape(tf.boolean_mask(dist_to_cens, d == 1), (N, -1))
            extra_dist = tf.reduce_min(
                tf.reshape(tf.boolean_mask(dist_to_cens, d == 0), (n_class, n_class - 1, shots_add)), axis=1)
            triplet_loss = tf.maximum(0., self.margin + inner_dist - tf.reshape(extra_dist, (N, -1)))
        return triplet_loss


class CentroidTriplet():

    def __init__(self, margin=0.5, soft=False, n_shots=5, mean=False):
        super(CentroidTriplet, self).__init__()
        self.margin = margin
        self.n_shots = n_shots
        self.soft = soft
        self.mean = mean

    def __call__(self, embed, n_class=5):
        embed = ops.convert_to_tensor_v2(embed, name="embed")
        # comver tensor
        embed = tf.cast(embed, tf.float32)
        N, D = embed.shape

        res_embed = tf.reshape(embed, (n_class, self.n_shots, D))

        # centroids = tf.reduce_mean(res_embed, 1, keepdims=True)
        if self.mean:
            centroids = tf.reduce_mean(res_embed, 1, keepdims=True)
        else:
            centroids = tfp.stats.percentile(res_embed, 50.0, 1, keepdims=True)

        dist_to_cens = tf.reshape(l2_dis(tf.expand_dims(res_embed, 1), centroids),
                                  (n_class, n_class * self.n_shots))

        d = tf.reshape(tf.repeat(tf.eye(n_class), self.n_shots), (n_class, n_class * self.n_shots))
        inner_dist = tf.reshape(tf.boolean_mask(dist_to_cens, d == 1), (N, -1))
        extra_dist = tf.reshape(tf.reduce_min(
            tf.reshape(tf.boolean_mask(dist_to_cens, d == 0), (n_class, n_class - 1, self.n_shots)), axis=1),  (N, -1))


        if self.soft:
            triplet_loss = tf.math.log1p(tf.math.exp(inner_dist - extra_dist))

        else:
            triplet_loss = tf.maximum(0., self.margin + inner_dist - extra_dist)

        return triplet_loss


class NucleusTriplet():

    def __init__(self, margin=0.5, soft=False, n_shots=5, mean=False):
        super(NucleusTriplet, self).__init__()
        self.margin = margin
        self.n_shots = n_shots
        self.soft = soft
        self.mean = mean

    def __call__(self, embed, n_class=5):
        embed = ops.convert_to_tensor_v2(embed, name="embed")
        # comver tensor
        embed = tf.cast(embed, tf.float32)
        N, D = embed.shape
        res_embed = tf.reshape(embed, (n_class, self.n_shots, D))
        # K = n_class - 1
        # NK = (self.n_shots * n_class) - K
        if self.mean:
            centroids = tf.reduce_mean(res_embed, 1, keepdims=True)
            dist_to_cens = tf.reshape(l2_dis(tf.expand_dims(centroids, 1), res_embed),
                                      (n_class, n_class * self.n_shots))
            cens_to_cens =  tf.reshape(l2_dis(tf.expand_dims(centroids, 1), centroids), (n_class, n_class))

        else:
            centroids = tfp.stats.percentile(res_embed, 50.0, 1, keepdims=True)
            dist_to_cens = tf.reshape(l1_dis(tf.expand_dims(centroids, 1), res_embed),
                                          (n_class, n_class * self.n_shots))


        d = tf.reshape(tf.repeat(tf.eye(n_class), self.n_shots), (n_class, n_class * self.n_shots))
        d_cens = tf.eye(n_class)
        inner_dist = tf.reduce_max(tf.reshape(tf.boolean_mask(dist_to_cens, d == 1), (n_class, self.n_shots)), -1)
        extra_dist = tf.reduce_min(tf.reshape(tf.boolean_mask(dist_to_cens, d == 0), (n_class, (n_class - 1) * self.n_shots)), axis=-1)
        inter_dist =  tf.reduce_min(tf.reshape(tf.boolean_mask(cens_to_cens, d_cens == 0), (n_class, (n_class - 1))), axis=-1)



        if self.soft:
            neg_cluster = 0.5 * (inter_dist + extra_dist)
            triplet_loss = tf.math.log1p(tf.math.exp(inner_dist- neg_cluster ))

        else:
            triplet_loss = tf.maximum(0., self.margin + inner_dist - extra_dist)

        return triplet_loss


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

        centroid_pos = (ap + pp) / 2.
        centroid_neg = (an + pn) / 2.

        # d_pos = l2_dis(ap, pp) # distance between positive anchor and pair
        # d_neg = l2_dis(an, pn)  # distance between negative anchor and pair
        # d_pos_neg = l2_dis(centroid_pos, centroid_neg)  # distance between positive and negative centroids

        d_pos = l2_dis(ap, pp)  # distance between positive anchor and pair
        d_neg = l2_dis(an, pn)  # distance between negative anchor and pair
        d_pos_neg = l2_dis(centroid_pos, centroid_neg)  # distance between positive and negative centroids

        pull_in = (d_neg + d_pos) / 2.
        push_away = d_pos_neg

        if self.soft:
            triplet_loss = tf.square(pull_in)
        elif self.squared:
            triplet_loss = tf.square(tf.maximum(0.0, self.margin - push_away + pull_in))

        else:
            triplet_loss = tf.maximum(0.0, (self.margin + pull_in) - (push_away))

        return triplet_loss


class DoubleBarlow:
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

        # positive loss
        c_diff = tf.math.log(tf.math.cosh(c_ap - I)) + tf.math.log(tf.math.cosh(c_an))
        c_diff = tf.where(I != 1, c_diff * self.alpha, c_diff)

        # negative loss
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

    # b = tf.transpose(tf.Variable([[1., 2., 3., 4., 5., 6.]]))

    n_class = 3
    n_shots = 2
    # a = tf.ones((n_shots * n_class, 1)) * b
    a =  tf.random.normal((n_class * n_shots, 10))
    a = tf.math.l2_normalize(a, -1)
    loss = NucleusTriplet(n_shots=n_shots, mean=True, soft=True)
    loss(a, n_class=n_class)
