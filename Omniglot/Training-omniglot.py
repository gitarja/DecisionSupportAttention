import tensorflow as tf

tf.keras.backend.set_floatx('float32')
from Omniglot.DataGenerator import Dataset
from NNModels.FewShotModel import FewShotModel
from NNModels.AdversarialModel import DiscriminatorModel
import tensorflow_addons as tfa
from Utils.PriorFactory import GaussianMixture, Gaussian
from Utils.Libs import euclidianMetric, computeACC, cosineSimilarity
import numpy as np
import datetime
from Omniglot.Conf import TENSOR_BOARD_PATH
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--adversarial', type=bool, default=False)

    args = parser.parse_args()

    # set up GPUs
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

    # training setting
    eval_interval = 1
    train_shots = 15
    test_shots = 5
    classes = 5
    inner_batch_size = 25
    inner_iters = 4
    n_buffer = 100
    ref_num = 5
    val_loss_th = 1e+3

    # training setting
    epochs = 5000
    lr = 1e-3

    # siamese and discriminator hyperparameter values
    z_dim = 64

    # tensor board
    log_dir = TENSOR_BOARD_PATH + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_path =  TENSOR_BOARD_PATH + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "\\model"
    if args.adversarial == True:
        log_dir = TENSOR_BOARD_PATH + "adv\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_path = TENSOR_BOARD_PATH + "adv\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "\\model"
    train_log_dir = log_dir + "\\train"
    test_log_dir = log_dir + "\\test"
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)



    # loss
    triplet_loss = tfa.losses.TripletSemiHardLoss()
    binary_loss = tf.losses.BinaryCrossentropy(from_logits=True)

    # optimizer
    siamese_optimizer = tf.optimizers.SGD(lr=lr)
    if args.adversarial == True: # using adversarial as well
        discriminator_optimizer = tf.optimizers.SGD(lr=lr / 3)
        generator_optimizer = tf.optimizers.SGD(lr=lr)

    model = FewShotModel(filters=64, z_dim=z_dim)
    disc_model = DiscriminatorModel(n_hidden=z_dim, n_output=1, dropout_rate=0.5)


    #check point
    checkpoint = tf.train.Checkpoint(step=tf.Variable(1), siamese_model=model)
    manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=3)

    for epoch in range(epochs):
        # dataset
        mini_dataset, test_images, test_labels = train_dataset.get_mini_batches(n_buffer,
                                                                                inner_batch_size, inner_iters,
                                                                                train_shots, classes, split=True,
                                                                                test_shots=test_shots
                                                                                )
        train_loss = []
        for images, labels in mini_dataset:
            # sample from gaussian mixture
            samples = Gaussian(len(images), z_dim, n_labels=classes)

            with tf.GradientTape() as siamese_tape, tf.GradientTape() as discriminator_tape, tf.GradientTape() as generator_tape:
                train_logits = model(images, training=True)
                embd_loss = triplet_loss(labels, train_logits)  # triplet loss
                train_loss.append(embd_loss)

                if args.adversarial == True: # using adversarial as well
                    # generative

                    z_fake = disc_model(train_logits, training=True)
                    z_true = disc_model(samples, training=True)

                    # discriminator loss
                    D_loss_fake = binary_loss(z_fake, tf.zeros_like(z_fake))
                    D_loss_real = binary_loss(z_true, tf.ones_like(z_true))
                    D_loss = D_loss_real + D_loss_fake

                    # generator loss
                    G_loss = binary_loss(z_fake, tf.ones_like(z_fake))

                # Use the gradient tape to automatically retrieve
                # the gradients of the trainable variables with respect to the loss.
            siamese_grads = siamese_tape.gradient(embd_loss, model.trainable_weights)
            siamese_optimizer.apply_gradients(zip(siamese_grads, model.trainable_weights))

            if args.adversarial == True: # using adversarial as well
                discriminator_grads = discriminator_tape.gradient(D_loss, disc_model.trainable_weights)
                generator_grads = generator_tape.gradient(G_loss, disc_model.trainable_weights)
                discriminator_optimizer.apply_gradients(zip(discriminator_grads, disc_model.trainable_weights))
                generator_optimizer.apply_gradients(zip(generator_grads, disc_model.trainable_weights))

        with train_summary_writer.as_default():
            tf.summary.scalar('loss', tf.reduce_mean(train_loss), step=epoch)
        if (epoch + 1) % eval_interval == 0:
            val_logits = model(test_images, training=False)
            val_loss = triplet_loss(test_labels, val_logits)
            if (val_loss_th > val_loss):
                val_loss_th = val_loss
                manager.save()

            with test_summary_writer.as_default():
                tf.summary.scalar('loss', val_loss, step=epoch)

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

    print(np.average(acc_avg))
