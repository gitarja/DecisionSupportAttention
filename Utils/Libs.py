import tensorflow as tf
import math
import numpy as np
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
from sklearn.metrics import euclidean_distances

def euclidianMetric(query, references, labels, ref_num=1):
    n = query.shape[0]
    q = query# n-class * k-shot
    r = references
    disc = euclidean_distances(q, r)
    disc = np.reshape(disc, (n, n,ref_num))
    pred = tf.argmin(np.mean(disc, -1), 1)
    return pred == labels

def computeACC(metrics, labels):
    pred = tf.argmax(metrics, -1)
    return tf.reduce_mean(tf.cast(pred== labels, tf.float32))


def cosineSimilarity(query, references, ref_num=1):
    metrics = tf.losses.CosineSimilarity(reduction=tf.losses.Reduction.NONE)
    n = query.shape[0]
    q = query  # n-class * k-shot
    r = tf.reduce_mean(tf.expand_dims(tf.reshape(references, (n, ref_num, -1)), 1), 1)
    logits = -metrics(q, r)

    return logits

def predict_class(x):
    idx, c = np.unique(x, return_counts=True)
    c_max = np.argmax(c)
    return idx[c_max]
def kNN(query, q_labels, references, ref_labels, ref_num=1, th=0.5, return_pred=True):
    X = references
    q = query
    y = ref_labels + 1
    classifier = KNeighborsClassifier(n_neighbors=ref_num, metric="euclidean", algorithm="kd_tree")
    classifier.fit(X, y)


    q_y = q_labels + 1

    if return_pred:
        return classifier.predict(q)
    else:
        return classifier.predict(q) == q_y




