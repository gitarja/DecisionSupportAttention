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
import pandas as pd
from sklearn.metrics import euclidean_distances

def knn_class(q_logits, labels, ref_logits, ref_labels, shots=5):
    N, D = ref_logits.shape
    q_logits = q_logits.numpy()
    ref_logits = ref_logits.numpy()
    # ref_logits = tf.reduce_mean(tf.reshape(ref_logits, (len(labels), shots, D)), 1)
    # ref_logits = tfp.stats.percentile(tf.reshape(ref_logits, (len(labels), shots, D)), 50.0, axis=1)
    # mean  = np.mean(np.concatenate([q_logits, ref_logits], 0), 0)
    # std = np.std(np.concatenate([q_logits, ref_logits], 0), 0)
    # q_logits = (q_logits - mean) / std
    # ref_logits = (ref_logits - mean) / std
    acc = kNN(q_logits, labels, ref_logits, ref_labels, ref_num=1, return_pred=False)
    return acc





# dataset
test_dataset = Dataset(mode="test")
num_classes = 5
shots = 5

#checkpoint
models_path = "/mnt/data1/users/pras/result/Siamese/OmniGlot/n_class_08/"
random.seed(2021)

test_list = pd.read_csv("test_list.csv")
data_test = [test_dataset.get_batches(shots=shots, num_classes=num_classes) for i in range(1000)]


for index, row in test_list.iterrows():
    models = models_path + row["path"]
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
    all_std = []
    for j in range(10):
        # checkpoint.restore(manager.checkpoints[-row["index"]])
        checkpoint.restore(manager.checkpoints[-j])

        acc_avg = []
        for i in range(len(data_test)):
            query, labels, references, ref_labels = data_test[i]
            q_logits = model(query, False)
            ref_logits = model(references, False)

            acc = knn_class(q_logits, labels, ref_logits, ref_labels, shots)
            # if np.sum(acc) < num_classes:
            #     print("wrong")
            acc_avg.append(np.average(acc))
        all_acc.append(np.average(acc_avg))
        all_std.append(np.std(acc_avg))
    # df = pd.DataFrame(np.concatenate(acc_avg).astype(np.float), columns=["acc"])
    # df = pd.DataFrame(acc_avg, columns=["acc"])
    # df.to_csv(str(num_classes)+"-way-"+str(shots)+"-shots.csv")
    # df.to_csv( row["path"] + ".csv")

    print(str(np.max(all_acc)) + "," + str(all_std[np.argmax(all_acc)]) + "," + str(np.argmax(all_acc) + 1))
