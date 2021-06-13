from DNNModels.NNModels.GazeModel import GazeModel, GazeEnsembleModel
import pandas as pd
import tensorflow as tf
import numpy as np


#folder
checkpoint_prefix = "D:\\usr\\pras\\result\\AttentionTest-Analysis\\Simulation\\"

#model param
output = 2
batch_size = 1
learning_rate = 1e-3

#setting generator
test_file = "D:\\usr\\pras\\data\\AttentionTestData\\Collaboration\\High-risk\\HT_000005_gazeHeadPose.csv"
test_file_game = "D:\\usr\\pras\\data\\AttentionTestData\\Collaboration\\High-risk\\HT_000005_gameResults.csv"
turn_file = "D:\\usr\\pras\\data\\AttentionTestData\\Collaboration\\Turns_children.csv"
test_data = pd.read_csv(test_file)
test_data = test_data.iloc[test_data.index % 4 == 1]
test_data = test_data[test_data["Time"] > 0][["Time", "GazeX", "GazeY", "ObjectX", "ObjectY"]].values
file_game = pd.read_csv(test_file_game)
data_game = file_game[["SpawnTime", "ResponseTime"]].values
turn_file = pd.read_csv(turn_file)[["Character"]].values

#setting model
model = GazeEnsembleModel(output=output)


#loss
mean_loss = tf.losses.MeanSquaredError()

#optimizer
optimizer = tf.keras.optimizers.Adamax(learning_rate=learning_rate)

#metrics
loss_metric = tf.keras.metrics.Mean()


#manager
checkpoint = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, model=model)
manager = tf.train.CheckpointManager(checkpoint, checkpoint_prefix, max_to_keep=3)
checkpoint.read(manager.latest_checkpoint)

gaze_x = test_data[0,1]
gaze_y = test_data[0,2]
for i in range(1, 100):
    current = test_data[i]
    past = test_data[i-1]
    type_stimulus = [-1]
    if current[-1] != -1:
        type_stimulus = turn_file[np.argmin(np.abs(data_game[:, 0] - current[0]))]
    X = tf.expand_dims(tf.Variable(np.concatenate([[gaze_x], [gaze_y], past[3:], current[3:], type_stimulus])), 0)
    # X = np.where(X == -1, 0, X)
    next_gaze = model(X)
    next_gaze = next_gaze.numpy().flatten()
    gaze_x = next_gaze[0]
    gaze_y = next_gaze[1]

    print("Gaze prediction: (%f, %f), Gaze label: (%f, %f) " % (gaze_x, gaze_y, test_data[i,1], test_data[i, 2]))



