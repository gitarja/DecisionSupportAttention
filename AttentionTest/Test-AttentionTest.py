from AttentionTest.Conf import TENSOR_BOARD_PATH
from AttentionTest.DataGenerator import Dataset
from NNModels.SiameseModel import AttentionModel
import tensorflow as tf
from Utils.Libs import kNN
import numpy as np
from sklearn.metrics import matthews_corrcoef
import random
import math
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import euclidean_distances
random.seed(2021)

def cos_dist(q, support):
    dist = 1 - np.arccos(cosine_similarity(q,support)) / math.pi
    return np.average(dist), np.std(dist)
def knn_class(q_logits, labels, ref_logits, ref_labels, ):
    q_logits = q_logits.numpy()
    preds = kNN(q_logits, labels, ref_logits, ref_labels, ref_num=1, return_pred=True)
    return preds

generator_train = Dataset(mode="train_val", val_frac=0.2)
generator_test = Dataset(mode="test")
#prepare dataset
x, y, val_x, val_y = generator_train.fetch_all(validation=True)

test_x, test_y = generator_test.fetch_all()


model = AttentionModel(filters=32, z_dim=32)

checkpoint_path = TENSOR_BOARD_PATH + "centroid" + "\\model\\"
checkpoint = tf.train.Checkpoint(step=tf.Variable(1), siamese_model=model)

manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=10)

checkpoint.restore(manager.latest_checkpoint)
print(manager.latest_checkpoint)

q_logit = model(test_x, training=False)
ref_logits = model(x, training=False)

preds = knn_class(q_logit, test_y, ref_logits, y)
mcc_score = matthews_corrcoef(test_y+1, preds)
print(mcc_score)
print(np.average(preds == test_y+1))
# print(preds)
# print(test_y+1)

#
#
# # add new class
# outlier_data = Dataset("outlier")
# x_out = outlier_data.data[1:]
# out_logits =  model(x_out, training=False)
# y_out = np.ones(len(x_out)) * outlier_data.labels
#
# support_outliers = out_logits[3:]
# support_y_outliers = y_out[3:]
#
# q_outliers = out_logits[:3]
# q_y_outliers = y_out[:3]
#
#
# # add to ref
# ref_logits = tf.concat([ref_logits, support_outliers], 0)
# y = np.concatenate([y, support_y_outliers])
#
# # add to query
# q_logit = tf.concat([q_logit, q_outliers], 0)
# test_y = np.concatenate([test_y, q_y_outliers])
#
#
# #combined
# preds = knn_class(q_outliers, q_y_outliers, ref_logits, y)
# print(np.average(preds == q_y_outliers+1))
#
#
# preds = knn_class(q_logit, test_y, ref_logits, y)
# print(np.average(preds == test_y+1))
# mcc_score = matthews_corrcoef(test_y+1, preds)
# print(mcc_score)
#
# out_distances = 1 - np.arccos(cosine_similarity(out_logits, ref_logits[y==0])) / math.pi
# out_distances_pos = euclidean_distances(out_logits, ref_logits[y == 0])
# out_distances_neg = euclidean_distances(out_logits, ref_logits[y == 1])
# print(np.average(out_distances_pos, -1))
# print(np.average(out_distances_neg, -1))
# print(np.average(out_distances, -1))