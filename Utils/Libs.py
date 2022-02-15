import tensorflow as tf
import math
import numpy as np
from sklearn.neighbors import KNeighborsClassifier,RadiusNeighborsClassifier
from sklearn.metrics import euclidean_distances


def l2_dis(x1, x2):
     return tf.sqrt(tf.reduce_sum(tf.multiply(x1, x1), -1) + tf.reduce_sum(tf.multiply(x2, x2), -1) - 2 * tf.reduce_sum(tf.multiply(x1, x2), -1))

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
    X_train = references
    X_test = query
    y = ref_labels + 1
    classifier = KNeighborsClassifier(n_neighbors=ref_num, metric="euclidean", algorithm="kd_tree")
    classifier.fit(X_train, np.unique(y))


    q_y = q_labels + 1

    if return_pred:
        return classifier.predict(X_test)
    else:
        return classifier.predict(X_test) == q_y

def classify(q_logits, labels, ref_logits, n_class=5, n_shots=5, mean=True):
    N, _ = q_logits.shape
    _, D = ref_logits.shape

    centroids = tf.reduce_mean(tf.reshape(ref_logits, (n_class, n_shots, D)), 1) # mean

    distances = l2_dis(tf.expand_dims(q_logits, 1), centroids)

    preds = tf.argmin(distances, -1)

    return preds == tf.cast(labels, dtype=preds.dtype)


