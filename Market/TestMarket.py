import tensorflow as tf

tf.keras.backend.set_floatx('float32')
from Market.DataGenerator import Dataset
from Market.Conf import TENSOR_BOARD_PATH
import datetime
from NNModels.RGBModel import RGBModel
from Utils.Libs import kNN, euclidianMetric, computeACC, cosineSimilarity
from Utils.CustomLoss import CentroidTriplet
import numpy as np
import random
import tensorflow_probability as tfp
import glob
import os
import pandas as pd
from sklearn.metrics import euclidean_distances

os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
gpus = tf.config.experimental.list_logical_devices('GPU')


def knn_class(q_logits, labels, ref_logits, ref_labels, shots=5):
    N, D = ref_logits.shape
    # compute cent
    # all_ref_labels = np.unique(ref_labels)
    # new_ref_logits = np.zeros((len(all_ref_labels), D))
    # new_ref_labels = np.zeros((len(all_ref_labels)))

    # for i in range(len(all_ref_labels)):
    #     new_ref_logits[i] = np.mean(ref_logits[ref_labels==all_ref_labels[i]], 0)
    #     new_ref_labels[i] = all_ref_labels[i]
    # normalize
    # mean = np.mean(np.concatenate([q_logits, ref_logits], 0), 0)
    # std = np.std(np.concatenate([q_logits, ref_logits], 0), 0)
    # q_logits = (q_logits - mean) / std
    # new_ref_logits = (ref_logits - mean) / std
    # ref_logits = tf.reduce_mean(tf.reshape(ref_logits, (len(labels), shots, D)), 1)
    # ref_logits = tfp.stats.percentile(tf.reshape(ref_logits, (len(labels), shots, D)), 50.0, axis=1)
    q_logits = q_logits.numpy()
    acc = kNN(q_logits, labels, ref_logits, ref_labels, ref_num=1)
    return acc


# dataset
test_list = pd.read_csv("test_list.csv")
test_dataset = Dataset(mode="test")
shots = 5
random.seed(2021)
dim = 128
models_path = "/mnt/data1/users/pras/result/Siamese/Market-1501/n-class/"
# checkpoint
for index, row in test_list.iterrows():

    models = models_path + row["path"]

    data_test = [test_dataset.get_batches()]
    div = 6

    checkpoint_path = models + "/model"
    print(models)

    correlation = False
    # model
    model = RGBModel(z_dim=dim)
    # check point

    checkpoint = tf.train.Checkpoint(step=tf.Variable(1), siamese_model=model)

    manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=100)
    all_acc = []

    checkpoint.restore(manager.checkpoints[-row["index"]])

    acc_avg = []
    for i in range(len(data_test)):
        query, labels, references, ref_labels = data_test[i]
        ref_logits = np.zeros((len(references),
                               dim))  # len ref = 19732, to prevent memory overflow we split the task of extracting the embedding
        q_logits = model(query, False)
        step_ref = int(len(references) / div)
        res = len(references) % div
        for i in range(div):
            ref_logits[i * step_ref:(i + 1) * step_ref] = model(references[i * step_ref:(i + 1) * step_ref], False)
        if res > 0:
            ref_logits[(i + 1) * step_ref:div + ((i + 1) * step_ref)] = model(
                references[(i + 1) * step_ref:div + ((i + 1) * step_ref)], False)
        acc = knn_class(q_logits, labels, ref_logits, ref_labels, shots)

        dist = euclidean_distances(q_logits, ref_logits)
        y_true = np.tile(np.expand_dims(ref_labels, 0), [len(labels), 1]) == np.tile(np.expand_dims(labels, -1), len(ref_labels))
        y_pred = dist
        np.save("y_true.npy", y_true)
        np.save("y_pred.npy", y_pred)
    #     acc_avg.append(acc)
    # # print(np.average(acc_avg))
    # all_acc.append((np.average(acc_avg)))
    # df = pd.DataFrame(np.concatenate(acc_avg).astype(np.float), columns=["acc"])
    # df.to_csv(row["path"] + ".csv")
    # print(str(np.max(all_acc)) + "," + str(np.argmax(all_acc) + 1))
