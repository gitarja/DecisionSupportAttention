import tensorflow as tf

tf.keras.backend.set_floatx('float32')
from Omniglot.DataGenerator import Dataset
from NNModels.FewShotModel import FewShotModel
from NNModels.AdversarialModel import DiscriminatorModel
import tensorflow_addons as tfa
from Utils.PriorFactory import GaussianMixture, Gaussian, GaussianMultivariate
from Utils.Libs import euclidianMetric, computeACC, cosineSimilarity
import numpy as np
import datetime
from Omniglot.Conf import TENSOR_BOARD_PATH
import argparse
import random

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--adversarial', type=bool, default=False)
    parser.add_argument('--double_trip', type=bool, default=False)

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

    cross_tower_ops = tf.distribute.HierarchicalCopyAllReduce(num_packs=3)
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=cross_tower_ops)

    random.seed(1) #set seed
    train_dataset = Dataset(mode="train_val", val_frac=0.1)

    # training setting
    eval_interval = 100
    train_shots = 20
    validation_shots = 20
    classes = 60
    inner_batch_size = 25
    ALL_BATCH_SIZE =  inner_batch_size * strategy.num_replicas_in_sync
    n_buffer = 100
    ref_num = 5
    val_loss_th = 1e+3

    # training setting
    episodes = 5000
    lr = 1e-3
    lr_siamese = 1e-3

    # early stopping
    early_th = 10
    early_idx = 0

    # siamese and discriminator hyperparameter values
    z_dim = 64

    # tensor board
    log_dir = TENSOR_BOARD_PATH + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_path = TENSOR_BOARD_PATH + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "\\model"
    if args.adversarial == True:
        log_dir = TENSOR_BOARD_PATH + "adv\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_path = TENSOR_BOARD_PATH + "adv\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "\\model"
    train_log_dir = log_dir + "\\train"
    test_log_dir = log_dir + "\\test"
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # loss

    triplet_loss = tfa.losses.TripletSemiHardLoss( reduction=tf.keras.losses.Reduction.NONE)
    binary_loss = tf.losses.BinaryCrossentropy(from_logits=True,  reduction=tf.keras.losses.Reduction.NONE)

    with strategy.scope():
        # optimizer
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            lr,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True)
        siamese_optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)

        if args.adversarial == True:  # using adversarial as well
            discriminator_optimizer = tf.optimizers.Adam(lr=lr/3)
            generator_optimizer = tf.optimizers.Adam(lr=lr)

        model = FewShotModel(filters=64, z_dim=z_dim)
        disc_model = DiscriminatorModel(n_hidden=z_dim, n_output=1, dropout_rate=0.3)

        # check point
        checkpoint = tf.train.Checkpoint(step=tf.Variable(1), siamese_model=model)
        manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=3)
        # metrics
        # train
        loss_train = tf.keras.metrics.Mean()
        # test
        loss_test = tf.keras.metrics.Mean()

    with strategy.scope():
        def compute_triplet_loss(labels, predictions, global_batch_size):
            per_example_loss = triplet_loss(labels, predictions)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)

        def compute_binary_loss(labels, predictions, global_batch_size):
            per_example_loss = binary_loss(labels, predictions)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)


    with strategy.scope():
        def train_step(inputs, GLOBAL_BATCH_SIZE=0):
            images, labels = inputs
            # sample from gaussian mixture
            samples = tf.math.l2_normalize(GaussianMultivariate(len(images), z_dim, mean=0, var=1.), -1)

            with tf.GradientTape() as siamese_tape, tf.GradientTape() as discriminator_tape, tf.GradientTape() as generator_tape:
                train_logits = model(images, training=True)
                embd_loss = compute_triplet_loss(labels, train_logits, GLOBAL_BATCH_SIZE)  # triplet loss
                train_loss.append(embd_loss)
                if args.adversarial == True:  # using adversarial as well
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
            siamese_optimizer.apply_gradients(zip(siamese_grads, model.trainable_weights))

            if args.adversarial == True:  # using adversarial as well
                discriminator_grads = discriminator_tape.gradient(D_loss, disc_model.trainable_weights)
                generator_grads = generator_tape.gradient(G_loss, model.trainable_weights)
                discriminator_optimizer.apply_gradients(zip(discriminator_grads, disc_model.trainable_weights))
                generator_optimizer.apply_gradients(zip(generator_grads, model.trainable_weights))
            loss_train(embd_loss)
            return embd_loss


        def val_step(inputs, GLOBAL_BATCH_SIZE=0):
            test_images, test_labels = inputs
            logits = model(test_images, training=False)
            loss = compute_triplet_loss(test_labels, logits, GLOBAL_BATCH_SIZE)
            loss_test(loss)
            return loss
    # validation dataset
    val_dataset = train_dataset.get_mini_batches(n_buffer,
                                                 inner_batch_size,
                                                 validation_shots, classes, validation=True,
                                                 )


    def reset_metric():
        loss_train.reset_states()
        loss_test.reset_states()


    with strategy.scope():
        # `experimental_run_v2` replicates the provided computation and runs it
        # with the distributed input.
        @tf.function
        def distributed_train_step(dataset_inputs, GLOBAL_BATCH_SIZE):
            per_replica_losses = strategy.run(train_step,
                                              args=(dataset_inputs, GLOBAL_BATCH_SIZE))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                                   axis=None)


        @tf.function
        def distributed_test_step(dataset_inputs, GLOBAL_BATCH_SIZE):
            return strategy.run(val_step, args=(dataset_inputs, GLOBAL_BATCH_SIZE))


        # validation dataset
        val_dataset = train_dataset.get_mini_batches(n_buffer,
                                                     inner_batch_size,
                                                     validation_shots, classes, validation=True,
                                                     )

        for epoch in range(episodes):
            # dataset
            train_loss = []

            mini_dataset = train_dataset.get_mini_batches(n_buffer,
                                                          inner_batch_size,
                                                          train_shots, classes, validation=False,
                                                          )

            for images, labels in mini_dataset:
                distributed_train_step([images, labels], ALL_BATCH_SIZE)

            if (ep + 1) % eval_interval == 0:

                manager.save()
                for test_images, test_labels in val_dataset:
                    distributed_test_step([images, labels], ALL_BATCH_SIZE)
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', loss_train.result().numpy(), step=epoch)
                with test_summary_writer.as_default():
                    tf.summary.scalar('loss', loss_test.result().numpy(), step=epoch)
                print("Training loss=%f, validation loss=%f" % (
                    loss_train.result().numpy(), loss_test.result().numpy()))  # print train and val losses
                reset_metric()

                # if (val_loss_th > val_loss):
                #     val_loss_th = val_loss
                #
                #     early_idx = 0
                # else:
                #     early_idx += 1
                # if early_idx == early_th:
                #     break



