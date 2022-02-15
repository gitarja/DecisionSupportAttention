import tensorflow as tf

tf.keras.backend.set_floatx('float32')
from MiniImageNet.DataGenerator import Dataset
from MiniImageNet.Conf import TENSOR_BOARD_PATH
import datetime
from NNModels.FewShotModel import FewShotModel
from NNModels.RGBModel import RGBModel
from Utils.Libs import kNN, classify

import numpy as np
import random

import pandas as pd



def knn_class(q_logits, labels, ref_logits, ref_labels, shots=5, n_class=5):
    N, D = ref_logits.shape
    q_logits = q_logits.numpy()
    ref_logits = ref_logits.numpy()
    ref_logits = tf.reduce_mean(tf.reshape(ref_logits, (n_class, shots, D)), 1) # mean
    acc = kNN(q_logits, labels, ref_logits, ref_labels, ref_num=1, return_pred=False)
    return acc


# dataset
test_dataset = Dataset(mode="test")
num_classes = 5
shots = 5

# checkpoint
models_path = TENSOR_BOARD_PATH + ""
random.seed(2021)

test_list = pd.read_csv("test_list.csv")
data_test = [test_dataset.get_batches(shots=shots, num_classes=num_classes, num_query=1) for i in range(600)]

for index, row in test_list.iterrows():
    models = models_path + row["path"]
    checkpoint_path = models + "/model"
    print(models)

    # model
    model = FewShotModel(z_dim=64)

    # check point

    checkpoint = tf.train.Checkpoint(step=tf.Variable(1), siamese_model=model)

    manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=100)
    all_acc = []
    all_std = []
    for j in range(1, 10):
    # checkpoint.restore(manager.checkpoints[-row["index"]])
        checkpoint.restore(manager.checkpoints[-j])

        acc_avg = []
        for i in range(len(data_test)):
            query, labels, references, ref_labels = data_test[i]
            q_logits = model(query, False)
            ref_logits = model(references, False)
            acc = knn_class(q_logits, labels, ref_logits, ref_labels, shots, n_class=num_classes)

            # if np.sum(acc) < num_classes:
            #     print("wrong")
            acc_avg.append(np.average(acc))
        all_acc.append(np.average(acc_avg))


    print(str(np.max(all_acc)) + "," + str(np.argmax(all_acc) + 1))
