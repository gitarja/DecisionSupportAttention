from AttentionTest.Conf import TENSOR_BOARD_PATH
from AttentionTest.DataGenerator import Dataset
from NNModels.SiameseModel import AttentionModel
import tensorflow as tf
from Utils.Libs import kNN
import numpy as np
from sklearn.metrics import matthews_corrcoef
import random
import matplotlib.pyplot as plt

import shap
random.seed(2021)

generator_train = Dataset(mode="train_val", val_frac=0.2)
generator_test = Dataset(mode="test")
#prepare dataset
x, y, val_x, val_y = generator_train.fetch_all(validation=True)

test_x, test_y = generator_test.fetch_all()


model = AttentionModel(filters=32, z_dim=32)

checkpoint_path = TENSOR_BOARD_PATH + "centroid" + "\\model\\"
checkpoint = tf.train.Checkpoint(step=tf.Variable(1), siamese_model=model)

manager = tf.train.CheckpointManager(checkpoint, checkpoint_path, max_to_keep=10)

checkpoint.restore(manager.latest_checkpoint)
print(manager.latest_checkpoint)
ref_logits = model(x, training=False)

# add new class
outlier_data = Dataset("outlier")
x_out = outlier_data.data[1:]
out_logits =  model(x_out, training=False)
y_out = np.ones(len(x_out)) * outlier_data.labels

support_outliers = out_logits[3:]
support_y_outliers = y_out[3:]

query = x_out[:3]
query_logits = out_logits[:3]
ref_logits = tf.concat([ref_logits, support_outliers], 0)


x = np.concatenate([x, x_out[3:]], 0)
y = np.concatenate([y, support_y_outliers])


# set support
model.setSupports(ref_logits[y==0], ref_logits[y==1], ref_logits[y==2])

def f(X):
    X = X.astype("f")
    predictions = [model.predictClass(np.expand_dims(X[i,:], 0)) for i in range(X.shape[0])]
    return np.array(predictions)


explainer = shap.KernelExplainer(f, x)
feature_names = ["gaze-adj1", "gaze-adj2", "gaze-adj3", "gaze-adj4", "gaze-adj5",
                 "velocity-sen",
    "acceleration-avg",
    "fixation-std",
    "distance-sen",
    "angle-sen",
    "gaze-obj-en",
    "gaze-obj-sen",
    "gaze-obj-spe",
    "go-positive",
    "go-negative",
        "RT-var",
    "acceleration-std"
                 ]


for i in [1, 2]:
    x_batch_test = query[i]
    shap_values = explainer.shap_values(x_batch_test, nsamples=300)

    shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0],
                                           shap_values[0], show=True, feature_names=feature_names)

    shap.plots._waterfall.waterfall_legacy(explainer.expected_value[1],
                                           shap_values[1], show=True, feature_names=feature_names)

    shap.plots._waterfall.waterfall_legacy(explainer.expected_value[2],
                                           shap_values[2], show=True, feature_names=feature_names)
    plt.close()
    # np.savetxt(TENSOR_BOARD_PATH + "shapley_" + str(i) + "_typical.csv",
    #            np.concatenate([np.expand_dims(explainer.expected_value[0], 0), shap_values[0]]), delimiter=",")
    # np.savetxt(TENSOR_BOARD_PATH + "shapley_" + str(i) + "_asd.csv",
    #            np.concatenate([np.expand_dims(explainer.expected_value[1], 0), shap_values[1]]), delimiter=",")

# for i in [0, 15, 21]:
#     x_batch_test = test_x[i]
#     shap_values = explainer.shap_values(x_batch_test, nsamples=300)
#
#     shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0],
#                                            shap_values[0], show=True, feature_names=feature_names)
#
#     shap.plots._waterfall.waterfall_legacy(explainer.expected_value[1],
#                                            shap_values[1], show=True, feature_names=feature_names)
#     plt.close()
#     np.savetxt(TENSOR_BOARD_PATH + "shapley_" + str(i) + "_typical.csv",
#                np.concatenate([np.expand_dims(explainer.expected_value[0], 0), shap_values[0]]), delimiter=",")
#     np.savetxt(TENSOR_BOARD_PATH + "shapley_" + str(i) + "_asd.csv",
#                np.concatenate([np.expand_dims(explainer.expected_value[1], 0), shap_values[1]]), delimiter=",")