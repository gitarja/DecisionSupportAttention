import tensorflow as tf

tf.keras.backend.set_floatx('float32')
from Omniglot.DataGenerator import Dataset
from Omniglot.Conf import TENSOR_BOARD_PATH
import datetime
from NNModels.FewShotModel import FewShotModel
from Utils.Libs import kNN, euclidianMetric, computeACC, cosineSimilarity
import numpy as np

#checkpoint
checkpoint_path = "D:\\usr\\pras\\result\\Siamese\\OmniGlot\\adv\\20210628-183701\\model\\"

#model
model = FewShotModel(filters=64, z_dim=64)

# check point
checkpoint = tf.train.Checkpoint(step=tf.Variable(1), siamese_model=model)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=3)
checkpoint.restore(manager.latest_checkpoint)

# dataset
test_dataset = Dataset(mode="test")
way = 5
shots = 5

acc_avg = []
for i in range(4):
    test_data = test_dataset.get_batches(shots=shots, num_classes=way)
    for _, (query, labels, references, ref_labels) in enumerate(test_data):
        val_logits = model(query, training=False)
        ref_logits = model(references, training=False)

        acc = kNN(val_logits, labels, ref_logits, ref_labels, ref_num=1)
        # dist = cosineSimilarity(val_logits, ref_logits, ref_num=5)
        # acc = computeACC(dist, labels)
        acc_avg.append(acc)
print(np.average(acc_avg))

