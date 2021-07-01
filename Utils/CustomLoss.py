import tensorflow as tf
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

        #comver tensor
        ap = tf.cast(ap, tf.float32)
        pp = tf.cast(pp, tf.float32)
        an = tf.cast(an, tf.float32)
        pn = tf.cast(pn, tf.float32)

        d_pos = tf.reduce_sum(tf.square(ap - pp), 1) #distance between positive anchor and pair
        d_neg = tf.reduce_sum(tf.square(an - pn), 1) #distance between negative anchor and pair
        d_pos_neg = tf.reduce_sum(tf.square(ap - an), 1) #distance between positive and negative anchor

        if self.soft:
            triplet_loss = tf.math.log1p(tf.math.exp((d_neg + d_pos) - d_pos_neg))
        else:
            triplet_loss = tf.maximum(0.0, (self.margin + d_neg + d_pos) - d_pos_neg)
        # Get final mean triplet loss
        triplet_loss = tf.reduce_mean(triplet_loss, axis=-1)

        return triplet_loss


if __name__ == '__main__':
    import numpy as np
    X = tf.Variable(np.random.normal(size=(25, 64)), dtype=tf.float32)
    labels = tf.Variable(np.random.randint(1, 5, size=(25, )), dtype=tf.float32)
    cl = DoubleTriplet(soft=True)
    loss = cl(X, X, X, X)
    print(loss)