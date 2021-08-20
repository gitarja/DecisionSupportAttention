from AttentionTest.Conf import TENSOR_BOARD_PATH
from AttentionTest.DataGenerator import Dataset
from NNModels.SiameseModel import AttentionModel
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import math
import shap
from Utils.CustomLoss import CentroidTriplet
import datetime
#make TF reproducible
seed = 2021
tf.random.set_seed(seed)
np.random.seed(seed)

train_class = 2
shots = 10


#setting
epochs = 50
eval_interval = 1
val_loss_th = 1e+3
margin = 0.35
early_th = 10
lr = 1e-5
#prepare model and loss
model = AttentionModel(filters=32, z_dim=32)

optimizer = tf.optimizers.Adam(learning_rate=lr)
loss = tfa.losses.TripletHardLoss(margin=margin)

#checkpoint
checkpoint_path = TENSOR_BOARD_PATH + "triplet" + "/model"
checkpoint = tf.train.Checkpoint(step=tf.Variable(1), siamese_model=model)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=early_th)
#prepare dataset
generator_train = Dataset(mode="train_val", val_frac=0.2)
x, y, val_x, val_y = generator_train.fetch_all(validation=True)
anchor_val, labels_val = generator_train.get_mini_offline_batches(val_x, val_y, n_class=train_class, shots=shots)
for epoch in range(epochs):
    anchor, labels = generator_train.get_mini_offline_batches(x, y, n_class=train_class, shots=shots)
    with tf.GradientTape() as siamese_tape, tf.GradientTape():
        logit = model(anchor, training=True)
        loss_train = tf.reduce_mean(loss(labels, logit))

    siamese_grads = siamese_tape.gradient(loss_train, model.trainable_weights)
    optimizer.apply_gradients(zip(siamese_grads, model.trainable_weights))

    if epoch % eval_interval == 0:
        logit_val = model(anchor_val, training=False)
        loss_test = tf.reduce_mean(loss(labels_val, logit_val))

        print("Training loss=%f, validation loss=%f" % (
            np.mean(loss_train.numpy()),  np.mean(loss_test.numpy())))  # print train and val losses
        val_loss = np.mean(loss_test.numpy())

        if (val_loss_th >= val_loss):
            val_loss_th = val_loss
            manager.save()
            early_idx = 0
















