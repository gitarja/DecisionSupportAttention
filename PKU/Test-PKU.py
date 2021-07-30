import tensorflow as tf
# tf.keras.backend.set_floatx('float32')
from PKU.DataGenerator import Dataset
from PKU.Conf import TENSOR_BOARD_PATH
import datetime
from NNModels.RGBModel import RGBModel, DomainDiscriminator
from NNModels.FewShotModel import FewShotModel, FewShotModelSmall
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



# dataset
test_dataset = Dataset(mode="test")
shots = 1


#checkpoint
models_path = "/mnt/data1/users/pras/result/Siamese/PKU-SketchReID/*_double"
random.seed(2021)
num_classes = 50
data_test = [test_dataset.get_batches(shots=shots, num_classes=num_classes) for i in range(1)]

for models in sorted(glob.glob(models_path)):
    checkpoint_path = models + "/model"
    print(models)
    z_dim = 64
    correlation = False
    #model
    sketch_model = FewShotModelSmall(z_dim=z_dim)
    rgb_model = FewShotModelSmall(z_dim=z_dim)
    disc_model = DomainDiscriminator()
    # check point

    checkpoint = tf.train.Checkpoint(step=tf.Variable(1), sketch_model=sketch_model, rgb_model=rgb_model, disc_model=disc_model)

    manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=100)
    all_acc = []
    for j in range(1, 2, 1):
        checkpoint.restore(manager.checkpoints[-j])

        acc_avg = []
        for i in range(len(data_test)):
            query, labels, references, ref_labels = data_test[i]
            q_logits = sketch_model(query, training=False)
            ref_logits = rgb_model(references, training=False)

            acc = knn_class(q_logits, labels, ref_logits, ref_labels, shots)
            acc_avg.append(acc)

        all_acc.append(np.average(acc_avg))

    print(np.max(all_acc))
