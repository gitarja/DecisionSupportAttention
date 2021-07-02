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
    q_logits = tf.tile(q_logits, (len(ref_logits), 1))
    pred_labels = np.zeros_like(labels)
    for i in range(len(q_logits)):
        logits = tf.sigmoid(deep_metric([q_logits[i], ref_logits]))
        pred_labels[i] = ref_labels[tf.argmax(logits)]

    return np.average(labels==pred_labels)



#checkpoint
checkpoint_path = "D:\\usr\\pras\\result\\Siamese\\OmniGlot\\20210701-222437_double\\model\\"
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
shots = 1


random.seed(2)
for way in [5, 20, 50, 100]:
    acc_avg = []
    for i in range(1000):
        query, labels, references, ref_labels = test_dataset.get_batches(shots=shots, num_classes=way)

        q_logits = model(query, training=False)
        ref_logits = model(references, training=False)

        if metric:
            acc = knn_class(q_logits, labels, ref_logits, ref_labels)
        else:
            acc = metric_class(q_logits, labels, ref_logits, ref_labels, deep_metric)

            # dist = cosineSimilarity(val_logits, ref_logits, ref_num=5)
            # acc = computeACC(dist, labels)
        acc_avg.append(acc)

    print(np.average(acc_avg))


