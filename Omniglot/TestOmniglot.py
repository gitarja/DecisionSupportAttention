import tensorflow as tf

tf.keras.backend.set_floatx('float32')
from Omniglot.DataGenerator import Dataset
from Omniglot.Conf import TENSOR_BOARD_PATH
import datetime
from NNModels.FewShotModel import FewShotModel, DeepMetric
from Utils.Libs import kNN, euclidianMetric, computeACC, cosineSimilarity
import numpy as np
import random


def knn_class(q_logits, labels, ref_logits, ref_labels):
    acc = kNN(q_logits, labels, ref_logits, ref_labels, ref_num=1)
    return acc

def metric_class(q_logits, labels, ref_logits, ref_labels, deep_metric):

    pred_labels = np.zeros_like(labels)
    for i in range(len(q_logits)):
        query = tf.tile(tf.expand_dims(q_logits[i],0), (len(ref_logits), 1))
        logits = deep_metric([query, ref_logits])
        pred_labels[i] = ref_labels[tf.argmax(logits).numpy()]

    return np.average(labels==pred_labels)



#checkpoint
checkpoint_path = "D:\\usr\\pras\\result\\Siamese\\OmniGlot\\barlow\\20210712-094810_double\\model\\"
metric = False
#model
model = FewShotModel(filters=64, z_dim=64)
deep_metric = DeepMetric(filters=64)
# check point
if metric:
    checkpoint = tf.train.Checkpoint(step=tf.Variable(1), siamese_model=model, deep_metric=deep_metric)
else:
    checkpoint = tf.train.Checkpoint(step=tf.Variable(1), siamese_model=model)

manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=3)
checkpoint.restore(manager.latest_checkpoint)

# dataset
test_dataset = Dataset(mode="test")
shots = 5


random.seed(2)
for way in [5, 20, 50, 100]:
    acc_avg = []
    for i in range(1000):
        query, labels, references, ref_labels = test_dataset.get_batches(shots=shots, num_classes=way)

        q_logits = model(query, training=False)
        ref_logits = model(references, training=False)

        if metric:
            acc = metric_class(q_logits, labels, ref_logits, ref_labels, deep_metric)
        else:

            acc = knn_class(q_logits, labels, ref_logits, ref_labels)

            # dist = cosineSimilarity(val_logits, ref_logits, ref_num=5)
            # acc = computeACC(dist, labels)
        acc_avg.append(acc)

    print(np.average(acc_avg))


