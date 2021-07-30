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
os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"
gpus = tf.config.experimental.list_logical_devices('GPU')


def knn_class(q_logits, labels, ref_logits, ref_labels, shots=5):
    N, D = ref_logits.shape
    # compute cent
    # all_ref_labels = np.unique(ref_labels)
    # new_ref_logits = np.zeros((len(all_ref_labels), D))
    # new_ref_labels = np.zeros((len(all_ref_labels)))
    #
    # for i in range(len(all_ref_labels)):
    #     new_ref_logits[i] = np.median(ref_logits[ref_labels==all_ref_labels[i]], 0)
    #     new_ref_labels[i] = all_ref_labels[i]
    # ref_logits = tf.reduce_mean(tf.reshape(ref_logits, (len(labels), shots, D)), 1)
    # ref_logits = tfp.stats.percentile(tf.reshape(ref_logits, (len(labels), shots, D)), 50.0, axis=1)
    acc = kNN(q_logits.numpy(), labels, ref_logits, ref_labels, ref_num=1)
    return acc




# dataset
test_dataset = Dataset(mode="test")
shots = 5


#checkpoint
models_path = "/mnt/data1/users/pras/result/Siamese/Market-1501/*_double"
random.seed(2021)
num_classes = 5
data_test = [test_dataset.get_batches()]
div = 6
for models in sorted(glob.glob(models_path)):
    checkpoint_path = models + "/model"
    print(models)
    dim=128
    correlation = False
    #model
    model = RGBModel(z_dim=dim)
    # check point

    checkpoint = tf.train.Checkpoint(step=tf.Variable(1), siamese_model=model)

    manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=100)
    all_acc = []

    for j in range(1, 10, 1):
        checkpoint.restore(manager.checkpoints[-j])

        acc_avg = []

        for i in range(len(data_test)):
            query, labels, references, ref_labels = data_test[i]
            ref_logits = np.zeros((len(references), dim)) #len ref = 19732
            q_logits = model(query)
            step_ref = int(len(references) / div)
            res  = len(references) % div
            for i in range(div):
                ref_logits[i*step_ref:(i+1)*step_ref] = model(references[i*step_ref:(i+1)*step_ref] )
            if res>0:
                ref_logits[(i + 1) * step_ref:div + ((i + 1) * step_ref)] = model(references[(i + 1) * step_ref:div + ((i + 1) * step_ref)])
            acc = knn_class(q_logits, labels, ref_logits, ref_labels, shots)

            acc_avg.append(acc)
        all_acc.append((np.average(acc_avg)))

    print(np.max(all_acc))
