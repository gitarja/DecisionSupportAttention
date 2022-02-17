import os.path
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import tensorflow_addons as tfa
import tensorflow as tf
tf.keras.backend.set_floatx('float32')
from MiniImageNet.DataGenerator import Dataset
from NNModels.FewShotModel import FewShotModel
from NNModels.RGBModel import RGBModel
from Utils.CustomLoss import NucleusTriplet, CentroidTriplet, BarlowTwins
import datetime
from MiniImageNet.Conf import TENSOR_BOARD_PATH
import argparse
from Utils.Libs import classify

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
    parser.add_argument('--beta', type=float, default=0.5)
    parser.add_argument('--soft', type=bool, default=False)
    parser.add_argument('--n_class', type=int, default=25)
    parser.add_argument('--z_dim', type=int, default=64)
    parser.add_argument('--mean', type=bool, default=False)
    parser.add_argument('--shots', type=int, default=5)
    parser.add_argument('--query_train', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=50000)


    args = parser.parse_args()

    # set up GPUs
    gpus = tf.config.list_physical_devices('GPU')
    n_gpus = len(gpus)
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

    cross_tower_ops = tf.distribute.HierarchicalCopyAllReduce(num_packs=n_gpus)
    strategy = tf.distribute.MirroredStrategy(cross_device_ops=cross_tower_ops)

    print(args.n_class)

    train_dataset = Dataset(mode="training")
    val_dataset = Dataset(mode="validation")


    # training setting
    eval_interval = 15
    train_query = args.query_train
    train_class = args.n_class
    train_shots = args.shots


    val_class = 5

    val_shots = 5
    val_loss_th = 1e+3
    val_acc_th = 0.


    # training setting
    epochs = args.epochs
    lr = 1e-3


    # early stopping
    early_th = 25
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
    triplet_loss = CentroidTriplet( margin=args.margin, soft=args.soft,  mean=args.mean)
    # triplet_loss = NucleusTriplet(beta=args.beta, margin=args.margin, soft=args.soft, mean=args.mean)

    with strategy.scope():
        model = FewShotModel(z_dim=z_dim)
        # check point
        checkpoint = tf.train.Checkpoint(step=tf.Variable(1), siamese_model=model)

        manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=early_th)
        # optimizer
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            lr,
            decay_steps=2000,
            decay_rate=0.5)

        siamese_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        # siamese_optimizer = tfa.optimizers.RectifiedAdam(learning_rate=lr, min_lr=1e-6, total_steps=epochs, warmup_proportion=0.1)

        #metrics
        # train
        loss_train = tf.keras.metrics.Mean()
        acc_train = tf.keras.metrics.Mean()

        # test
        loss_test = tf.keras.metrics.Mean()
        acc_test = tf.keras.metrics.Mean()



    with strategy.scope():
        # def compute_triplet_loss(embd, n_class, global_batch_size):
        #     per_example_loss = triplet_loss(embd, n_class)
        #     return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)

        # def compute_triplet_loss(r_logits, n_class, n_shots, global_batch_size):
        #     per_example_loss = triplet_loss(r_logits, n_class, n_shots)
        #     return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)

        def compute_triplet_loss(q_logits, q_labels, r_logits, n_class, n_shots, n_query, global_batch_size):
            # per_example_loss = triplet_loss(q_logits, q_labels, r_logits, n_class, n_shots, n_query)
            per_example_loss = triplet_loss(r_logits, n_class, n_shots)
            return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)

    with strategy.scope():


        def train_step(inputs, GLOBAL_BATCH_SIZE=0):
            query, labels, references, ref_labels = inputs
            with tf.GradientTape() as siamese_tape, tf.GradientTape():
                q_logits = model(query, training=True)
                ref_logits = model(references, training=True)
                embd_loss = compute_triplet_loss(q_logits, labels, ref_logits, train_class, train_shots, train_query, train_class*train_query*n_gpus)  # triplet loss
                           # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            acc = classify(q_logits, labels, ref_logits, train_class, train_shots, mean=args.mean)
            siamese_grads = siamese_tape.gradient(embd_loss, model.trainable_weights)
            siamese_optimizer.apply_gradients(zip(siamese_grads, model.trainable_weights))
            loss_train(embd_loss)
            acc_train(acc)
            return embd_loss


        def val_step(inputs, GLOBAL_BATCH_SIZE=0):
            query, labels, references, ref_labels = inputs
            q_logits = model(query, training=False)
            ref_logits = model(references, training=False)
            loss = compute_triplet_loss(q_logits, labels, ref_logits, val_class, val_shots, 1, val_class * val_shots * n_gpus)  # triplet loss

            acc = classify(q_logits, labels, ref_logits, val_class,  val_shots, mean=args.mean)

            loss_test(loss)
            acc_test(acc)
            return loss

        def reset_metric():
            loss_train.reset_states()
            acc_train.reset_states()
            loss_test.reset_states()
            acc_test.reset_states()

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




        for epoch in range(epochs):

            if epoch == 2000:
                siamese_optimizer.beta_1 = 0.5
            # dataset
            train_inputs = train_dataset.get_batches(train_shots, train_class, num_query=train_query)


            distributed_train_step(train_inputs, train_class)

            if (epoch + 1) % eval_interval == 0:
                # validation dataset
                for val_epoch in range(300):
                    val_inputs = val_dataset.get_batches(val_shots, val_class)
                    distributed_test_step(val_inputs, val_class)
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', loss_train.result().numpy(), step=epoch)
                    tf.summary.scalar('acc', acc_train.result().numpy(), step=epoch)
                with test_summary_writer.as_default():
                    tf.summary.scalar('loss', loss_test.result().numpy(), step=epoch)
                    tf.summary.scalar('acc', acc_test.result().numpy(), step=epoch)
                print("Epoch: %f, Training loss=%f, validation loss=%f, validation acc=%f" % (epoch,
                                                                           loss_train.result().numpy(),
                                                                           loss_test.result().numpy(), acc_test.result().numpy()))  # print train and val losses

                val_loss = loss_test.result().numpy()
                val_acc = acc_test.result().numpy()

                if val_acc_th <= val_acc:
                    val_acc_th = val_acc
                    manager.save()


                if val_loss <= val_loss_th:
                    val_loss_th = val_loss
                    early_idx = 0
                else:
                    early_idx += 1
                reset_metric()
                # if early_idx == early_th:
                #     break

