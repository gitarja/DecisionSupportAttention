import tensorflow as tf

tf.keras.backend.set_floatx('float32')
from Omniglot.DataGeneratorSelf import Dataset
from NNModels.FewShotModel import FewShotModel, DeepMetric
from NNModels.AdversarialModel import DiscriminatorModel
import tensorflow_addons as tfa
import datetime
from Omniglot.Conf import TENSOR_BOARD_PATH
import argparse
from Utils.CustomLoss import BarlowTwins, DoubleBarlow, TripletOffline
import random
import os

os.environ["TF_XLA_FLAGS"] = "--tf_xla_enable_xla_devices"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2"

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--negative', type=bool, default=False)
    parser.add_argument('--alpha', type=float, default=1.)
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


    # random.seed(1)  # set seed
    train_dataset = Dataset(mode="train_val", val_frac=0.05)

    # training setting
    eval_interval = 5
    batch_size = 200
    ALL_BATCH_SIZE = batch_size * strategy.num_replicas_in_sync
    ref_num = 5
    val_loss_th = 1e+3
    shots= 20

    # training setting
    episodes = 5000
    lr = 1e-3

    # early stopping
    early_th = 10
    early_idx = 0

    # siamese and discriminator hyperparameter values
    z_dim = args.z_dim

    # tensor board
    log_dir = TENSOR_BOARD_PATH + "barlow\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    checkpoint_path = TENSOR_BOARD_PATH + "barlow\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "\\model"
    if args.negative == True:
        log_dir = TENSOR_BOARD_PATH + "barlow_negative\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        checkpoint_path = TENSOR_BOARD_PATH + "barlow_negative\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "\\model"
    train_log_dir = log_dir + "\\train"
    test_log_dir = log_dir + "\\test"
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
    test_summary_writer = tf.summary.create_file_writer(test_log_dir)

    # loss
    barlow_loss_pos = BarlowTwins(positive=True, reduction=tf.keras.losses.Reduction.SUM, alpha=args.alpha)
    # double_barlow = DoubleBarlow(alpha=args.alpha)
    double_barlow = TripletOffline()



    with strategy.scope():
        model = FewShotModel(filters=64, z_dim=z_dim)
        # check point
        checkpoint = tf.train.Checkpoint(step=tf.Variable(1), siamese_model=model)

        manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=10)

        # optimizer
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            lr,
            decay_steps=1000,
            decay_rate=0.96,
            staircase=True)

        siamese_optimizer = tf.optimizers.Adam(learning_rate=lr_schedule)


        #metrics
        # train
        loss_train = tf.keras.metrics.Mean()
        # test
        loss_test = tf.keras.metrics.Mean()



    with strategy.scope():

        def compute_baron_loss_pos(a, p):
            per_example_loss = barlow_loss_pos(a, p)
            return per_example_loss


        def compute_double_barlow_loss(a, p, n):
            per_example_loss = double_barlow(a, p, n)
            return per_example_loss



    with strategy.scope():


        def train_step(inputs, GLOBAL_BATCH_SIZE=0):

            with tf.GradientTape() as siamese_tape, tf.GradientTape():
                ap_logits = model.forward_pos(inputs, training=True)
                pp_logits = model.forward_pos(inputs, training=True)
                if args.negative == True:
                    nn_logits = model.forward_neg(inputs, training=True)
                    embd_loss = compute_double_barlow_loss(ap_logits, pp_logits, nn_logits)
                else:
                    embd_loss = compute_baron_loss_pos(ap_logits, pp_logits)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            siamese_grads = siamese_tape.gradient(embd_loss, model.trainable_weights)
            siamese_optimizer.apply_gradients(zip(siamese_grads, model.trainable_weights))
            loss_train(embd_loss)
            return embd_loss


        def val_step(inputs, GLOBAL_BATCH_SIZE=0):

            ap_logits = model.forward_pos(inputs, training=False)
            pp_logits = model.forward_pos(inputs, training=False)
            if args.negative == True:
                nn_logits = model.forward_neg(inputs, training=False)
                loss = compute_double_barlow_loss(ap_logits, pp_logits, nn_logits)  # triplet loss
            else:
               loss =  compute_baron_loss_pos(ap_logits, pp_logits)




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
        val_dataset = train_dataset.get_mini_self_batches(batch_size, shots=shots,
                                                             validation=True,
                                                             )
        mini_dataset = train_dataset.get_mini_self_batches(batch_size, shots=shots,
                                                           validation=False,
                                                           )
        for epoch in range(episodes):
            # dataset


            for a in mini_dataset:
                 distributed_train_step(a, ALL_BATCH_SIZE)

            if (epoch + 1) % eval_interval == 0:

                for a in val_dataset:
                    distributed_test_step(a, ALL_BATCH_SIZE)
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', loss_train.result().numpy(), step=epoch)
                with test_summary_writer.as_default():
                    tf.summary.scalar('loss', loss_test.result().numpy(), step=epoch)
                print("Epoch:%f,Training loss=%f, validation loss=%f" % (epoch,
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
