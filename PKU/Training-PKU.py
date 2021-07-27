import tensorflow as tf

tf.keras.backend.set_floatx('float32')
from PKU.DataGenerator import Dataset
from NNModels.SketchModel import SketchModel, DomainDiscriminator
from NNModels.AdversarialModel import DiscriminatorModel
import tensorflow_addons as tfa
import numpy as np
import datetime
from PKU.Conf import TENSOR_BOARD_PATH
import argparse
from Utils.CustomLoss import CentroidTriplet

import random
import os

os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"
random.seed(2021)  # set seed
tf.random.set_seed(2021)
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--margin', type=float, default=0.5)
    parser.add_argument('--soft', type=bool, default=False)
    parser.add_argument('--n_class', type=int, default=25)
    parser.add_argument('--z_dim', type=int, default=64)
    parser.add_argument('--mean', type=bool, default=False)


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

    print(args.n_class)

    train_dataset = Dataset(mode="train_val", val_frac=0.05)


    # training setting
    eval_interval = 1
    train_class = args.n_class

    batch_size = 256
    ALL_BATCH_SIZE = batch_size * strategy.num_replicas_in_sync
    val_class = len(train_dataset.val_labels)
    ref_num = 5
    val_loss_th = 1e+3
    shots=4

    # training setting
    epochs = 5000
    lr = 1e-3
    lr_siamese = 1e-3

    # early stopping
    early_th = 100
    early_idx = 0

    # siamese and discriminator hyperparameter values
    z_dim = args.z_dim

    # tensor board
    log_dir = TENSOR_BOARD_PATH + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_double"
    checkpoint_path = TENSOR_BOARD_PATH + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "_double" + "/model"

    train_log_dir = log_dir + "/train"
    test_log_dir = log_dir + "/test"
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # loss
    triplet_loss = CentroidTriplet(margin=args.margin, soft=args.soft,  n_shots=shots + 1, mean=args.mean) # sketch + shots
    binary_loss = tf.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    with strategy.scope():
        model = SketchModel(z_dim=z_dim)
        disc_model = DomainDiscriminator()
        # check point
        checkpoint = tf.train.Checkpoint(step=tf.Variable(1), siamese_model=model, deep_metric_model=disc_model)

        manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=early_th)
        # optimizer
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            lr,
            decay_steps=1000,
            decay_rate=0.8,
            staircase=True)
        siamese_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

        #metrics
        # train
        loss_train = tf.keras.metrics.Mean()
        # test
        loss_test = tf.keras.metrics.Mean()



    with strategy.scope():
        def compute_triplet_loss(embd, n_class, global_batch_size):
            per_example_loss = triplet_loss(embd, n_class)
            return tf.nn.compute_average_loss(per_example_loss)


        def compute_binary_loss(y_true, y_pred):
            per_example_loss = binary_loss(y_true, y_pred)
            return tf.nn.compute_average_loss(per_example_loss)

    with strategy.scope():


        def train_step(inputs, GLOBAL_BATCH_SIZE=0):
            X, y = inputs
            with tf.GradientTape() as siamese_tape, tf.GradientTape():
                embeddings = model(X, training=True)
                logits = disc_model(embeddings, training=True)
                embd_loss = compute_triplet_loss(embeddings, train_class, train_class * shots)  # triplet loss
                domanin_loss = compute_binary_loss(y, logits)
                loss = embd_loss + domanin_loss

            # the gradients of the trainable variables with respect to the loss.
            siamese_grads = siamese_tape.gradient(loss, model.trainable_weights)
            siamese_optimizer.apply_gradients(zip(siamese_grads, model.trainable_weights))
            loss_train(loss)
            return loss


        def val_step(inputs, GLOBAL_BATCH_SIZE=0):
            X, y = inputs
            embeddings = model(X, training=True)
            logits = disc_model(embeddings, training=False)
            embd_loss = compute_triplet_loss(embeddings, train_class, val_class * shots)  # triplet loss
            domanin_loss = compute_binary_loss(y, logits)
            loss = embd_loss + domanin_loss

            loss_test(loss)
            return loss

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
            return strategy.run(val_step, args=(dataset_inputs,GLOBAL_BATCH_SIZE))


        # validation dataset
        val_inputs = train_dataset.get_mini_offline_batches(val_class,
                                                           shots=shots,
                                                            validation=True
                                                           )

        for epoch in range(epochs):
            # dataset
            train_inputs = train_dataset.get_mini_offline_batches(train_class,
                                                                  shots=shots,
                                                                  validation=False

                                                                  )


            distributed_train_step(train_inputs, ALL_BATCH_SIZE)

            if (epoch + 1) % eval_interval == 0:

                distributed_test_step(val_inputs, ALL_BATCH_SIZE)
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', loss_train.result().numpy(), step=epoch)
                with test_summary_writer.as_default():
                    tf.summary.scalar('loss', loss_test.result().numpy(), step=epoch)
                print("Training loss=%f, validation loss=%f" % (
                    loss_train.result().numpy(), loss_test.result().numpy()))  # print train and val losses

                val_loss = loss_test.result().numpy()
                if (val_loss_th > val_loss):
                    val_loss_th = val_loss
                    manager.save()
                    early_idx = 0

                else:
                    early_idx += 1
                reset_metric()
                # if early_idx == early_th:
                #     break


