import tensorflow as tf
tf.keras.backend.set_floatx('float32')
from Omniglot.DataGenerator import Dataset
from NNModels.FewShotModel import FewShotModel
import tensorflow_addons as tfa

from Utils.Libs import euclidianMetric, computeACC
import numpy as np


#set up GPUs
gpus = tf.config.list_physical_devices('GPU')
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)

train_dataset = Dataset(training=True)
test_dataset = Dataset(training=False)

eval_interval = 1
train_shots = 15
classes = 20
inner_batch_size = 25
inner_iters = 4
n_buffer = 100
ref_num = 5

#training setting
epochs = 50
lr = 3e-3

#loss
triplet_loss = tfa.losses.TripletSemiHardLoss()
optimizer = tf.optimizers.Adamax(lr=lr)

model = FewShotModel()

for epoch in range(epochs):
    # dataset
    mini_dataset, test_images, test_labels, ref_images = train_dataset.get_mini_batches(n_buffer,
                                                  inner_batch_size, inner_iters, train_shots, classes, split=True, ref_num=ref_num
                                                  )
    loss_avg = []
    acc_avg = []
    for images, labels in mini_dataset:


        with tf.GradientTape() as tape:
            train_logits = model(images, training=True)
            loss = triplet_loss(labels, train_logits)


            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))


    if epoch % eval_interval == 0:
        val_logits = model(test_images, training=False)
        ref_logits = model(ref_images, training=False)
        loss = triplet_loss(test_labels, val_logits)
        dist_metrics = euclidianMetric(val_logits, ref_logits, ref_num=ref_num)
        val_acc = computeACC(dist_metrics, test_labels)
        acc_avg.append(val_acc)
        loss_avg.append(loss)

    print(np.average(acc_avg))





# dataset
shots = 5
test_data = test_dataset.get_batches(shots=shots, num_classes=5)
acc_avg = []
for _, (query, labels, references) in enumerate(test_data):
    val_logits = model(query, training=False)
    ref_logits = model(references, training=False)
    dist_metrics = euclidianMetric(val_logits, ref_logits, ref_num=shots)
    val_acc = computeACC(dist_metrics, labels)
    acc_avg.append(val_acc)
    # print(val_acc)

print(np.average(acc_avg))