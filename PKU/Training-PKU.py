import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import tensorflow as tf

tf.keras.backend.set_floatx('float32')
from PKU.DataGenerator import Dataset
from NNModels.RGBModel import RGBModel, DomainDiscriminator
from NNModels.FewShotModel import FewShotModel, FewShotModelSmall
import tensorflow_addons as tfa
import numpy as np
import datetime
from PKU.Conf import TENSOR_BOARD_PATH
import argparse
from Utils.CustomLoss import CentroidTripletSketch

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

    train_dataset = Dataset(mode="training", val_frac=0.1)
    test_dataset =  Dataset(mode="test")


    # training setting
    eval_interval = 5
    train_class = args.n_class

    batch_size = 256
    ALL_BATCH_SIZE = batch_size * strategy.num_replicas_in_sync
    val_class = len(test_dataset.labels)
    ref_num = 5
    val_loss_th = 1e+3
    shots=4

    # training setting
    epochs = 10000
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
    triplet_loss = CentroidTripletSketch(margin=args.margin, soft=args.soft,  n_shots=shots, mean=args.mean) # sketch + shots
    binary_loss = tf.losses.BinaryCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)

    with strategy.scope():
        sketch_model = FewShotModelSmall(z_dim=z_dim)
        rgb_model = FewShotModelSmall(z_dim=z_dim)
        disc_model = DomainDiscriminator()

        # check point
        checkpoint = tf.train.Checkpoint(step=tf.Variable(1), sketch_model=sketch_model, rgb_model=rgb_model, disc_model=disc_model)

        manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=early_th)
        # optimizer
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            lr,
            decay_steps=2000,
            decay_rate=0.8,
            staircase=True)
        siamese_optimizer = tfa.optimizers.RectifiedAdam(learning_rate=1e-4, total_steps=epochs, min_lr=1e-6, warmup_proportion=0.1)
        gen_optimizer =  tfa.optimizers.RectifiedAdam(learning_rate=1e-4, total_steps=epochs, min_lr=1e-6, warmup_proportion=0.1)
        disc_optimizer = tfa.optimizers.RectifiedAdam(learning_rate=1e-4, total_steps=epochs, min_lr=1e-6, warmup_proportion=0.1)


        #metrics
        # train
        loss_train = tf.keras.metrics.Mean()
        # test
        loss_test = tf.keras.metrics.Mean()



    with strategy.scope():
        def compute_triplet_loss(sketch, rgbs, n_class):
            per_example_loss = triplet_loss(sketch, rgbs, n_class)
            return tf.nn.compute_average_loss(per_example_loss)

        def compute_generator_loss(fake_output, sketch=True):
            if sketch:
                y = tf.ones_like(fake_output)
            else:
                y = tf.zeros_like(fake_output)
            per_example_loss = binary_loss(y, fake_output)
            return tf.nn.compute_average_loss(per_example_loss)


        def compute_discriminator_loss(real_output, fake_output):
            real_loss = binary_loss(tf.ones_like(real_output), real_output)
            fake_loss = binary_loss(tf.zeros_like(fake_output), fake_output)
            return tf.nn.compute_average_loss(real_loss) + tf.nn.compute_average_loss(fake_loss)

    with strategy.scope():


        def train_step(inputs, GLOBAL_BATCH_SIZE=0):
            sketch_inputs, rgb_inputs = inputs
            with tf.GradientTape() as siamese_tape, tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                sketch_embed = sketch_model(sketch_inputs, training=True)
                rgb_embed = rgb_model(rgb_inputs, training=True)
                sketch_logits = disc_model(sketch_embed, training=True)
                rgb_logits = disc_model(rgb_embed, training=True)
                embd_loss = compute_triplet_loss(sketch_embed, rgb_embed, train_class)  # triplet loss
                generate_loss = compute_generator_loss(sketch_logits) + compute_generator_loss(rgb_logits, False)
                disc_loss = compute_discriminator_loss(rgb_logits, sketch_logits)
                # loss  = embd_loss + generate_loss

            disc_grads = disc_tape.gradient(disc_loss, disc_model.trainable_weights)
            disc_optimizer.apply_gradients(zip(disc_grads, disc_model.trainable_weights))

            weights = sketch_model.trainable_weights + rgb_model.trainable_weights

            gen_grads = gen_tape.gradient(generate_loss, weights)
            gen_optimizer.apply_gradients(zip(gen_grads, weights))

            # the gradients of the trainable variables with respect to the loss.

            siamese_grads = siamese_tape.gradient(embd_loss, weights)
            siamese_optimizer.apply_gradients(zip(siamese_grads, weights))

            loss_train(embd_loss)
            return embd_loss


        def val_step(inputs, GLOBAL_BATCH_SIZE=0):
            sketch_inputs, rgb_inputs = inputs
            sketch_embed = sketch_model(sketch_inputs, training=False)
            rgb_embed = rgb_model(rgb_inputs, training=False)
            embd_loss = compute_triplet_loss(sketch_embed, rgb_embed, val_class)  # triplet loss
            loss = embd_loss

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
        val_inputs = test_dataset.get_mini_sketch_batches(val_class,
                                                           shots=shots,
                                                            validation=False
                                                           )

        for epoch in range(epochs):
            # dataset
            train_inputs = train_dataset.get_mini_sketch_batches(train_class,
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


