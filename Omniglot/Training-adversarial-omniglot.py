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

train_dataset = Dataset(training=True)
test_dataset = Dataset(training=False)

eval_interval = 100
train_shots = 20
classes = 5
inner_batch_size = 25
inner_iters = 4
n_buffer = 100
ref_num = 5

#training setting
epochs = 2000
lr = 3e-3

#siamese and discriminator hyperparameter values
z_dim = 64

#loss
triplet_loss = tfa.losses.TripletSemiHardLoss()
binary_loss = tf.losses.BinaryCrossentropy(from_logits=True)

#optimizer
siamese_optimizer = tf.optimizers.SGD(lr=lr)
discriminator_optimizer = tf.optimizers.SGD(lr=lr/5)
generator_optimizer = tf.optimizers.SGD(lr=lr)

model = FewShotModel(filters=64, z_dim=z_dim)
disc_model = DiscriminatorModel(n_hidden=z_dim, n_output=1, dropout_rate=0.5)

for epoch in range(epochs):
    # dataset
    mini_dataset = train_dataset.get_mini_batches(n_buffer,
                                                  inner_batch_size, inner_iters, train_shots, classes, split=False, ref_num=ref_num
                                                  )
    loss_avg = []
    acc_avg = []
    for images, labels in mini_dataset:
        #sample from gaussian mixture
        samples = Gaussian(len(images), z_dim, n_labels=classes*4)

        with tf.GradientTape() as siamese_tape, tf.GradientTape() as discriminator_tape, tf.GradientTape() as generator_tape:
            train_logits = model(images, training=True)
            embd_loss = triplet_loss(labels, train_logits) #triplet loss

            #generative

            z_fake = disc_model(train_logits, training=True)
            z_true = disc_model(samples, training=True)

            #discriminator loss
            D_loss_fake = binary_loss(z_fake, tf.zeros_like(z_fake))
            D_loss_real = binary_loss(z_true, tf.ones_like(z_true))
            D_loss = D_loss_real + D_loss_fake

            #generator loss
            G_loss = binary_loss(z_fake, tf.ones_like(z_fake))




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


    if (epoch+1) % eval_interval == 0:
        _, test_images, test_labels, ref_images = train_dataset.get_mini_batches(n_buffer,
                                                                                            inner_batch_size,
                                                                                            inner_iters, train_shots,
                                                                                            classes, split=True,
                                                                                            ref_num=ref_num
                                                                                            )
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