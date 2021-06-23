import tensorflow as tf
import math


def euclidianMetric(query, references, ref_num=1):
    n = query.shape[0]
    q = tf.expand_dims(query, 1) # n-class * k-shot
    r = tf.expand_dims(tf.reshape(references, (n, ref_num, -1)), 1)
    logits = tf.reduce_mean(tf.reduce_sum(tf.sqrt((q - r) ** 2), axis=-1), -1)
    return logits

def computeACC(metrics, labels):
    pred = tf.argmin(metrics, -1)

    return tf.reduce_mean(tf.cast(pred== labels, tf.float32))