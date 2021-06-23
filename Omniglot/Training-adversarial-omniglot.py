import tensorflow as tf
tf.keras.backend.set_floatx('float32')
from Omniglot.DataGenerator import Dataset
from NNModels.FewShotModel import FewShotModel
from NNModels.AdversarialModel import DiscriminatorModel
import tensorflow_addons as tfa
from Utils.PriorFactory import GaussianMixture, Gaussian
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

cross_tower_ops = tf.distribute.HierarchicalCopyAllReduce(num_packs=3)
strategy = tf.distribute.MirroredStrategy(cross_device_ops=cross_tower_ops)

train_dataset = Dataset(training=True)
test_dataset = Dataset(training=False)

eval_interval = 1
train_shots = 20
classes = 5
inner_batch_size = 25
inner_iters = 4
n_buffer = 100
ref_num = 1

ALL_BATCH_SIZE = inner_batch_size * strategy.num_replicas_in_sync

#training setting
epochs = 2000
lr = 3e-3

#siamese and discriminator hyperparameter values
z_dim = 64

with strategy.scope():

    #optimizer
    siamese_optimizer = tf.optimizers.SGD(lr=lr)
    discriminator_optimizer = tf.optimizers.SGD(lr=lr/5)
    generator_optimizer = tf.optimizers.SGD(lr=lr)

    model = FewShotModel(filters=64, z_dim=z_dim)
    disc_model = DiscriminatorModel(n_hidden=z_dim, n_output=1, dropout_rate=0.5)
#losses
with strategy.scope():

    #loss
    triplet_loss = tfa.losses.TripletSemiHardLoss(reduction=tf.keras.losses.Reduction.NONE)
    binary_loss = tf.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    def compute_binary_loss(labels, predictions, global_batch_size):
        per_example_loss = binary_loss(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)

    def compute_triplet_loss(labels, predictions, global_batch_size):
        per_example_loss = triplet_loss(labels, predictions)
        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)

with strategy.scope():
    def train_step(inputs, GLOBAL_BATCH_SIZE):
        for images, labels in inputs:
            # sample from gaussian mixture
            samples = Gaussian(len(images), z_dim, n_labels=classes * 4)

            with tf.GradientTape() as siamese_tape, tf.GradientTape() as discriminator_tape, tf.GradientTape() as generator_tape:
                train_logits = model(images, training=True)
                embd_loss = compute_triplet_loss(labels, train_logits, GLOBAL_BATCH_SIZE)  # triplet loss

                # generative

                z_fake = disc_model(train_logits, training=True)
                z_true = disc_model(samples, training=True)

                # discriminator loss
                D_loss_fake = compute_binary_loss(z_fake, tf.zeros_like(z_fake), GLOBAL_BATCH_SIZE)
                D_loss_real = compute_binary_loss(z_true, tf.ones_like(z_true), GLOBAL_BATCH_SIZE)
                D_loss = D_loss_real + D_loss_fake

                # generator loss
                G_loss = compute_binary_loss(z_fake, tf.ones_like(z_fake), GLOBAL_BATCH_SIZE)

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
            siamese_grads = siamese_tape.gradient(embd_loss, model.trainable_weights)
            discriminator_grads = discriminator_tape.gradient(D_loss, disc_model.trainable_weights)
            generator_grads = generator_tape.gradient(G_loss, disc_model.trainable_weights)

            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            siamese_optimizer.apply_gradients(zip(siamese_grads, model.trainable_weights))
            discriminator_optimizer.apply_gradients(zip(discriminator_grads, disc_model.trainable_weights))
            generator_optimizer.apply_gradients(zip(generator_grads, disc_model.trainable_weights))




with strategy.scope():
    # `experimental_run_v2` replicates the provided computation and runs it
    # with the distributed input.
    @tf.function
    def distributed_train_step(dataset_inputs):
        per_replica_losses = strategy.experimental_run_v2(train_step,
                                                          args=(dataset_inputs,))
        return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                               axis=None)


for epoch in range(epochs):
    # dataset
    mini_dataset = train_dataset.get_mini_batches(n_buffer,
                                                  ALL_BATCH_SIZE, inner_iters, train_shots, classes, split=False, ref_num=ref_num
                                                  )
    with strategy.scope():
        distributed_train_step(mini_dataset)

    loss_avg = []
    acc_avg = []
    if (epoch+1) % eval_interval == 0:
        _, test_images, test_labels, ref_images = train_dataset.get_mini_batches(n_buffer,
                                                                                            inner_batch_size,
                                                                                            inner_iters, train_shots,
                                                                                            classes, split=True,
                                                                                            ref_num=ref_num
                                                                                            )
        val_logits = model(test_images, training=False)
        ref_logits = model(ref_images, training=False)
        # loss = triplet_loss(test_labels, val_logits)
        dist_metrics = euclidianMetric(val_logits, ref_logits, ref_num=ref_num)
        val_acc = computeACC(dist_metrics, test_labels)
        acc_avg.append(val_acc)
        # loss_avg.append(loss)

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