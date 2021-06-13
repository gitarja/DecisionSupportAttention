from DNNModels.NNModels.GazeModel import GazeModel, GazeEnsembleModel
from DNNModels.DataFetch import DataFetch
import tensorflow as tf

# folder
checkpoint_prefix = "D:\\usr\\pras\\result\\AttentionTest-Analysis\\Simulation\\"
turn_file = "D:\\usr\\pras\\data\\AttentionTestData\\Collaboration\\Turns_children.csv"

# model param
N = 7
output = 2
batch_size = 256
learning_rate = 1e-3
EPOCH_NUM = 1500
# setting generator
data_fetch = DataFetch("D:\\usr\\pras\\data\\AttentionTestData\\Collaboration\\High-risk\\", "_gazeHeadPose.csv",
                       "_gameResults.csv", turn_file)
generator = data_fetch.loadData

# setting model
model = GazeEnsembleModel(output=output)

# generator

train_data = tf.data.Dataset.from_generator(
    lambda: generator(),
    output_types=(tf.int32, tf.float32, tf.float32),
    output_shapes=((), tf.TensorShape([N, ]), tf.TensorShape([output])))

train_data = train_data.shuffle(batch_size * 100).padded_batch(batch_size, padded_shapes=(
    (), tf.TensorShape([N, ]), tf.TensorShape([output])))

# loss
mean_loss = tf.losses.MeanSquaredError()

# optimizer
optimizer = tf.keras.optimizers.Adamax(learning_rate=learning_rate)

# metrics
loss_metric = tf.keras.metrics.Mean()

# manager
checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=10)
checkpoint.read(manager.latest_checkpoint)

loss_th = 1000

for epoch in range(EPOCH_NUM):
    for step, inputs in enumerate(train_data):
        _, X, y = inputs
        with tf.GradientTape() as tape:
            z = model(X)
            loss = mean_loss(y, z)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        loss_metric(loss)

    template = ("Epoch {}, Loss: {}")
    print(template.format(epoch + 1, loss_metric.result().numpy()))

    if (loss_th > loss_metric.result().numpy()):
        manager.save()
        loss_th = loss_metric.result().numpy()

    loss_metric.reset_states()


# # set generator
# generator = data_fetch.loadData()
# train_data = tf.data.Dataset.from_generator(
#     lambda: generator,
#     output_types=(tf.int32, tf.float32, tf.float32),
#     output_shapes=((), tf.TensorShape([1, N]), tf.TensorShape([1, output])))

# reset generator
# data_fetch.reset()
