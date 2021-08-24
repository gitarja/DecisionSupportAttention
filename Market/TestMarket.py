import tensorflow as tf

tf.keras.backend.set_floatx('float32')
from Market.DataGenerator import DataTest
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
from Utils.CustomMetrics import eval_func, average_precision_score_market

os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
gpus = tf.config.experimental.list_logical_devices('GPU')


def knn_class(q_logits, labels, ref_logits, ref_labels, shots=5):
    N, D = ref_logits.shape

    q_logits = q_logits.numpy()
    acc = kNN(q_logits, labels, ref_logits, ref_labels, ref_num=1, return_pred=False)
    return acc


# dataset
test_list = pd.read_csv("test_list.csv")
test_dataset = DataTest()
shots = 5
random.seed(2021)
dim = 128

# load data
q_images, q_labels, q_cams, ref_images, ref_labels, ref_cams = test_dataset.getTestData()

# ref_images = ref_images[ref_labels >= 0]
# ref_cams = ref_cams[ref_labels >= 0]
# ref_labels = ref_labels[ref_labels >= 0]

models_path = "/mnt/data1/users/pras/result/Siamese/Market-1501/"
# checkpoint
for index, row in test_list.iterrows():

    models = models_path + row["path"]

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

    ref_logits = np.zeros((len(ref_images),
                           dim))  # len ref = 19732, to prevent memory overflow we split the task of extracting the embedding
    q_logits = model(q_images, False)
    step_ref = int(len(ref_images) / div)
    res = len(ref_images) % div
    for i in range(div):
        ref_logits[i * step_ref:(i + 1) * step_ref] = model(ref_images[i * step_ref:(i + 1) * step_ref], False)
    if res > 0:
        ref_logits[(i + 1) * step_ref:] = model(ref_images[(i + 1) * step_ref:], False)

    acc = knn_class(q_logits, q_labels, ref_logits, ref_labels, shots)

    dist = euclidean_distances(q_logits, ref_logits)
    all_cmc, mAP = eval_func(dist, q_labels, ref_labels, q_cams, ref_cams)
    print(np.average(acc))
    print(all_cmc)
    print(mAP)

    # y_true = np.tile(np.expand_dims(ref_labels, 0), [len(labels), 1]) == np.tile(np.expand_dims(labels, -1), len(ref_labels))
    # y_pred = dist
    # np.save("y_true.npy", y_true)
    # np.save("y_pred.npy", y_pred)
#     acc_avg.append(acc)
# # print(np.average(acc_avg))
# all_acc.append((np.average(acc_avg)))
# df = pd.DataFrame(np.concatenate(acc_avg).astype(np.float), columns=["acc"])
# df.to_csv(row["path"] + ".csv")
# print(str(np.max(all_acc)) + "," + str(np.argmax(all_acc) + 1))
