import tensorflow as tf

tf.keras.backend.set_floatx('float32')
from Omniglot.DataGenerator import Dataset
from Omniglot.Conf import TENSOR_BOARD_PATH
import datetime
from NNModels.FewShotModel import FewShotModel, DeepMetric
from Utils.Libs import kNN, euclidianMetric, computeACC, cosineSimilarity
from Utils.CustomLoss import CentroidTriplet
import numpy as np
import random
import tensorflow_probability as tfp
import glob


def knn_class(q_logits, labels, ref_logits, ref_labels, shots=5):
    N, D = ref_logits.shape
    # ref_logits = tf.reduce_mean(tf.reshape(ref_logits, (len(labels), shots, D)), 1)
    # ref_logits = tfp.stats.percentile(tf.reshape(ref_logits, (len(labels), shots, D)), 50.0, axis=1)
    acc = kNN(q_logits, labels, ref_logits, ref_labels, ref_num=1)
    return acc

def metric_class(q_logits, labels, ref_logits, ref_labels, deep_metric, ref_num=5):

    pred_labels = np.zeros_like(labels)
    for i in range(len(q_logits)):
        query = tf.tile(tf.expand_dims(q_logits[i],0), (len(ref_logits), 1))
        disc = deep_metric([query, ref_logits])
        # disc = tf.reduce_mean(tf.reshape(disc, shape=(len(labels), ref_num)), -1)
        pred_labels[i] = ref_labels[tf.argmax(disc).numpy()]

    return np.average(labels==pred_labels)

def correlation_class(q_logits, labels, ref_logits, ref_labels):

    q_logits = (q_logits - tf.reduce_mean(q_logits, 0)) /  tf.math.reduce_std(q_logits, 0)
    ref_logits = (ref_logits - tf.reduce_mean(ref_logits, 0)) / tf.math.reduce_std(ref_logits, 0)

    corr = q_logits @ tf.transpose(ref_logits)

    preds = ref_labels[tf.argmax(corr, -1).numpy()]

    return np.mean(preds == labels)



# dataset
test_dataset = Dataset(mode="test")
shots = 5

# inputs = test_dataset.get_mini_offline_batches(3,
#                                                            shots=3,
#                                                            )
# triplet_loss = CentroidTriplet(margin=1., margin_2=1., n_shots=3)
# embd = model(inputs)
# triplet_loss(embd, n_class=3)

#checkpoint
models_path = "/mnt/data1/users/pras/result/Siamese/OmniGlot/*_double"
random.seed(2021)
num_classes = 5
data_test = [test_dataset.get_batches(shots=shots, num_classes=num_classes) for i in range(1000)]

for models in sorted(glob.glob(models_path)):
    checkpoint_path = models + "/model"
    print(models)

    correlation = False
    #model
    model = FewShotModel(filters=64, z_dim=64)
    deep_metric = DeepMetric()
    # check point

    checkpoint = tf.train.Checkpoint(step=tf.Variable(1), siamese_model=model, deep_metric_model=deep_metric)

    manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=100)
    all_acc = []
    for j in range(1, 16, 1):
        checkpoint.restore(manager.checkpoints[-j])

        acc_avg = []
        for i in range(len(data_test)):
            query, labels, references, ref_labels = data_test[i]
            q_logits = model(query)
            ref_logits = model(references)

            if correlation:
                acc = correlation_class(q_logits, labels, ref_logits, ref_labels)
            else:
                # acc = euclidianMetric(q_logits, ref_logits, labels, ref_num=shots)
                acc = knn_class(q_logits, labels, ref_logits, ref_labels, shots)
                # if np.sum(acc) != 5:
                #     print("wrong")
                # acc = metric_class(q_logits, labels, ref_logits, ref_labels, deep_metric, ref_num=shots)

                # dist = cosineSimilarity(val_logits, ref_logits, ref_num=5)
                # acc = computeACC(dist, labels)
            acc_avg.append(acc)
        all_acc.append((np.average(acc_avg)))

    print(np.max(all_acc))
