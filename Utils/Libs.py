import tensorflow as tf
import math
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def euclidianMetric(query, references, ref_num=1):
    n = query.shape[0]
    q = query# n-class * k-shot
    r = tf.reshape(references, (n, ref_num, -1))
    logits = -tf.reduce_sum(tf.sqrt((q - r) ** 2), axis=-1)
    return logits

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
def kNN(query, q_labels, references, ref_labels, ref_num=1, th=0.5):
    X = references.numpy()
    y = ref_labels + 1
    classifier = KNeighborsClassifier(n_neighbors=ref_num)
    classifier.fit(X, y)

    q = query.numpy()
    q_y = q_labels + 1

    # #identify outliers or novel class
    # dist, index = classifier.kneighbors(q, ref_num, True)
    # masked_pred = y[index] * (dist <= th)
    # predictions = []
    # for i in range(len(masked_pred)):
    #     predictions.append(predict_class(masked_pred[i,:]))
    # return np.average(predictions==q_y)

    #identify outliers or novel class
    return classifier.score(q, q_y)




