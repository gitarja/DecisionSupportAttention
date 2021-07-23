import tensorflow as tf

tf.keras.backend.set_floatx('float32')
from Omniglot.DataGenerator import Dataset
from NNModels.FewShotModel import FewShotModel, DeepMetric
from NNModels.AdversarialModel import DiscriminatorModel
import tensorflow_addons as tfa
import numpy as np
import datetime
from Omniglot.Conf import TENSOR_BOARD_PATH
import argparse
from Utils.CustomLoss import DoubleTriplet, TripletBarlow, DoubleTripletSoft

import random
import os

os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"
random.seed(2021)  # set seed
tf.random.set_seed(2021)
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--margin', type=float, default=1.)
    parser.add_argument('--soft', type=bool, default=False)
    parser.add_argument('--project', type=bool, default=False)
    parser.add_argument('--squared', type=bool, default=False)
    parser.add_argument('--z_dim', type=int, default=64)


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



    train_dataset = Dataset(mode="train")
    test_dataset = Dataset(mode="test")

    # training setting
    eval_interval = 1
    train_buffer = 512
    batch_size = 256
    ALL_BATCH_SIZE = batch_size * strategy.num_replicas_in_sync
    val_buffer = 500
    ref_num = 5
    val_loss_th = 1e+3
    shots=20

    # training setting
    epochs = 500000
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
    triplet_loss = DoubleTriplet(margin=args.margin, soft=args.soft, squared=args.squared)
    triplet_soft_loss = DoubleTripletSoft(squared=args.squared)
    triplet_barlow_loss = TripletBarlow()

    with strategy.scope():
        model = FewShotModel(filters=64, z_dim=z_dim)
        disc_model = DeepMetric()
        # check point
        checkpoint = tf.train.Checkpoint(step=tf.Variable(1), siamese_model=model, deep_metric_model=disc_model)

        manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=early_th)
        # optimizer
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            lr,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True)
        siamese_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

        #metrics
        # train
        loss_train = tf.keras.metrics.Mean()
        # test
        loss_test = tf.keras.metrics.Mean()



    with strategy.scope():

        def compute_triplet_loss(ap, pp, an, pn, global_batch_size):

            per_example_loss = triplet_loss(ap, pp, an, pn)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)

        def compute_triplet_soft_loss(pos, neg, pos_neg,  global_batch_size):
            per_example_loss = triplet_soft_loss(pos, neg, pos_neg)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)

        def compute_triplet_barlow_loss(ap, pp, an, pn):

            per_example_loss = triplet_barlow_loss(ap, pp, an, pn)
            return per_example_loss
    with strategy.scope():


        def train_step(inputs, GLOBAL_BATCH_SIZE=0):
            ap, pp,  an, pn = inputs

            with tf.GradientTape() as siamese_tape, tf.GradientTape():
                if args.project:
                    ap_logits = model.forward_pos(ap, training=True)
                    an_logits = model.forward_pos(an, training=True)
                else:
                    ap_logits = model(ap, training=True)
                    an_logits = model(an, training=True)

                pp_logits = model(pp, training=True)
                pn_logits = model(pn, training=True)

                # positive_dist = disc_model([ap_logits, pp_logits])
                # negative_dist = disc_model([an_logits, pn_logits])
                # pos_neg_dist = disc_model([(ap_logits+pp_logits) / 2., (an_logits + pn_logits) / 2.])
                #
                # embd_loss = compute_triplet_soft_loss(positive_dist, negative_dist, pos_neg_dist, GLOBAL_BATCH_SIZE)

                embd_loss = compute_triplet_loss(ap_logits, pp_logits, an_logits, pn_logits, GLOBAL_BATCH_SIZE)  # triplet loss
                # embd_loss = compute_triplet_barlow_loss(ap_logits, pp_logits, an_logits, pn_logits)
                # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            siamese_grads = siamese_tape.gradient(embd_loss, model.trainable_weights)
            siamese_optimizer.apply_gradients(zip(siamese_grads, model.trainable_weights))
            loss_train(embd_loss)
            return embd_loss


        def val_step(inputs, GLOBAL_BATCH_SIZE=0):
            ap, pp, an, pn = inputs
            ap_logits = model(ap, training=False)
            pp_logits = model(pp, training=False)
            an_logits = model(an, training=False)
            pn_logits = model(pn, training=False)

            # positive_dist = disc_model([ap_logits, pp_logits])
            # negative_dist = disc_model([an_logits, pn_logits])
            # pos_neg_dist = disc_model([(ap_logits + pp_logits) / 2., (an_logits + pn_logits) / 2.])
            #
            # loss = compute_triplet_soft_loss(positive_dist, negative_dist, pos_neg_dist, GLOBAL_BATCH_SIZE)

            loss = compute_triplet_loss(ap_logits, pp_logits, an_logits, pn_logits,
                                                 GLOBAL_BATCH_SIZE)  # triplet loss
            # loss = compute_triplet_barlow_loss(ap_logits, pp_logits, an_logits, pn_logits)

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
        val_dataset = train_dataset.get_mini_pairoffline_batches(val_buffer,
                                                             batch_size, shots=shots,
                                                             validation=False,
                                                             )

        for epoch in range(epochs):
            # dataset
            mini_dataset = train_dataset.get_mini_pairoffline_batches(train_buffer,
                                                                  batch_size,shots=shots,
                                                                  validation=False,
                                                                  )

            for ap, pp,  an, pn in mini_dataset:
                distributed_train_step([ap, pp,  an, pn], ALL_BATCH_SIZE)

            if (epoch + 1) % eval_interval == 0:

                for ap, pp,  an, pn  in val_dataset:
                    distributed_test_step([ap, pp,  an, pn], ALL_BATCH_SIZE)
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
                if early_idx == early_th:
                    break
