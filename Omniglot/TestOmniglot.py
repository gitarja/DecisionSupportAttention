import tensorflow as tf

tf.keras.backend.set_floatx('float32')
from Omniglot.DataGenerator import Dataset
from Omniglot.Conf import TENSOR_BOARD_PATH
import datetime
from NNModels.FewShotModel import FewShotModel
from Utils.Libs import kNN

#checkpoint
checkpoint_path = TENSOR_BOARD_PATH + "adv\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "\\model"

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
test_data = test_dataset.get_batches(shots=shots, num_classes=way)
acc_avg = []
for _, (query, labels, references, ref_labels) in enumerate(test_data):
    val_logits = model(query, training=False)
    ref_logits = model(references, training=False)

    acc = kNN(val_logits, labels, ref_logits, ref_labels, ref_num=5)
    print(acc)

