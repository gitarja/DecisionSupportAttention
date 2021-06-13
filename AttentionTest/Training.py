from Conf.Settings import FEATURES_PATH
from DNNModels.Siamese.DataGazeFeaturesFetch import DataGenerator
from DNNModels.NNModels.SiameseModel import SiameseModel
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import math
import shap

#make TF reproducible
seed = 0
tf.random.set_seed(seed)
np.random.seed(seed)

fold = 0
N = 16
k = 5



training_features_file = FEATURES_PATH + "gaze_significant_performance\\" + "data_train" + str(fold) + ".npy"
training_labels_file = FEATURES_PATH + "gaze_significant_performance\\" + "label_train" + str(fold) + ".npy"
# #validation
val_features_file = FEATURES_PATH + "gaze_significant_performance\\" + "data_val" + str(fold) + ".npy"
val_labels_file = FEATURES_PATH + "gaze_significant_performance\\" + "label_val" + str(fold) + ".npy"
#testing
testing_features_file = FEATURES_PATH + "gaze_significant_performance\\" + "data_test" + str(fold) + ".npy"
testing_labels_file = FEATURES_PATH + "gaze_significant_performance\\" + "label_test" + str(fold) + ".npy"

# #validation
# val_features_file = FEATURES_PATH + "gaze_significant_performance\\" + "data_test" + str(fold) + ".npy"
# val_labels_file = FEATURES_PATH + "gaze_significant_performance\\" + "label_test" + str(fold) + ".npy"
# #testing
# testing_features_file = FEATURES_PATH + "gaze_significant_performance\\" + "data_val" + str(fold) + ".npy"
# testing_labels_file = FEATURES_PATH + "gaze_significant_performance\\" + "label_val" + str(fold) + ".npy"


#training generator
data_generator = DataGenerator(training_features_file, training_labels_file, offline=True, double=False, training=False)
generator = data_generator.fetch_offline
train_generator = tf.data.Dataset.from_generator(
    lambda: generator(),
    output_types=( tf.float32, tf.float32, tf.float32),
    output_shapes=(tf.TensorShape([N, ]), tf.TensorShape([N, ]), tf.TensorShape([N, ])))

#training generator
data_generator_val = DataGenerator(val_features_file, val_labels_file, offline=True, double=False)
generator_val = data_generator_val.fetch_offline
val_generator = tf.data.Dataset.from_generator(
    lambda: generator_val(),
    output_types=( tf.float32, tf.float32, tf.float32),
    output_shapes=(tf.TensorShape([N, ]), tf.TensorShape([N, ]), tf.TensorShape([N, ])))

batch_size = 1
train_data = train_generator.shuffle(data_generator.data_n).batch(batch_size)
val_data = val_generator.shuffle(data_generator_val.data_n).batch(1)


model = SiameseModel()
optimizer = tf.optimizers.Adamax(learning_rate=0.001)
epochs = 20
it = 0
loss_val_th = 1e+3
early_stop = 0
early_stop_th = 3
margin = 0.5
for epoch in range(epochs):
    loss_avg = []
    for step, (x_batch_train,  x_batch_positive_train,  x_batch_negative_train) in enumerate(train_data):

        with tf.GradientTape() as tape:
            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            anchor = model(x_batch_train, training=True)  # Logits for this minibatch
            positive = model(x_batch_positive_train, training=True)  # Logits for this minibatch
            negative = model(x_batch_negative_train, training=True)  # Logits for this minibatch

            loss = model.tripletOffline(anchor, positive, negative, margin=margin)
            loss_avg.append(loss)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))




    loss_val = []
    for step, (x_batch_val, x_batch_positive_val, x_batch_negative_val) in enumerate(val_data):
        anchor = model(x_batch_val, training=False)  # Logits for this minibatch
        positive = model(x_batch_positive_val, training=False)  # Logits for this minibatch
        negative = model(x_batch_negative_val, training=False)  # Logits for this minibatch
        loss = model.tripletOffline(anchor, positive, negative, margin=margin)
        loss_val.append(loss)
    # Log every 16 batches.
    avg_loss_val = np.average(loss_val)
    print(
        "Validation loss at iteration %d: %.4f"
        % (epoch, float(avg_loss_val))
    )

    if avg_loss_val <= loss_val_th:
        loss_val_th = avg_loss_val
        early_stop = 0
    else:
        early_stop += 1
    if (early_stop >= early_stop_th):
        break






data_generator = DataGenerator(training_features_file, training_labels_file, offline=False)
generator = data_generator.fetch
train_generator = tf.data.Dataset.from_generator(
    lambda: generator(),
    output_types=( tf.float32,  tf.int32),
    output_shapes=(tf.TensorShape([N, ]), ()))


embed = []
label = []
for step, (x_batch_train, y_batch_train) in enumerate(train_generator.batch(1)):
    embed.append(model(x_batch_train, training=False))
    label.append(y_batch_train.numpy())




data_generator = DataGenerator(testing_features_file, testing_labels_file)
generator = data_generator.fetch


test_generator = tf.data.Dataset.from_generator(
    lambda: generator(),
    output_types=( tf.float32, tf.int32),
    output_shapes=(tf.TensorShape([N, ]), ()))

test_data = test_generator.shuffle(data_generator.data_n).batch(1)

embed = tf.concat(embed, 0)
label = tf.concat(label, 0)
typical_anchor = tf.random.shuffle(embed[label== 0])
asd_anchor = tf.random.shuffle(embed[label== 1])

model.setAnchor(typical_anchor, asd_anchor)
preds = []
labels = []
acc = []
for step, (x_batch_test, y_batch_test) in enumerate(test_data):


    dist_typical, dist_asd =  model.predictClass(x_batch_test)



    pred = np.argmax(np.array([dist_typical, dist_asd]).transpose())
    acc.append(y_batch_test.numpy() == pred)
    preds.append(pred)
    labels.append(y_batch_test.numpy())

    print(
        "label = %f, typical = %f, asd = %f"
        % (y_batch_test.numpy(), np.median(dist_typical), np.median(dist_asd))
    )


#save predictions
np.savetxt("labels_"+str(fold)+".csv", np.concatenate(labels), delimiter=",")
np.savetxt("preds_"+str(fold)+".csv", np.array(preds), delimiter=",")

labels = tf.keras.utils.to_categorical(np.concatenate(labels))
preds = tf.keras.utils.to_categorical(np.array(preds))
mcc_metric = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=2)
mcc_metric.update_state(labels, preds)
mcc_score = mcc_metric.result().numpy()
#print results
print(mcc_score)
print(np.average(acc))



def f(X):
    X = X.astype("f")
    predictions = [model.predictClass(np.expand_dims(X[i,:], 0)) for i in range(X.shape[0])]
    return np.array(predictions)

# X_train = np.load(training_features_file)
# explainer = shap.KernelExplainer(f, X_train)
# feature_names = ["gaze-adj1", "gaze-adj2", "gaze-adj3", "gaze-adj4", "gaze-adj5",  "velocity-sen",
#     "acceleration-avg",
#     "fixation-std",
#     "distance-sen",
#     "angle-sen",
#     "gaze-obj-en",
#     "gaze-obj-sen",
#     "gaze-obj-spe",
#     "go-positive",
#     "go-negative","RT-var"]
# for step, (x_batch_test, y_batch_test) in enumerate(test_data):
#
#     shap_values = explainer.shap_values(x_batch_test.numpy(), nsamples=300)
#     print(shap_values)
    # shap.plots._waterfall.waterfall_legacy(explainer.expected_value[0],
    #                                    shap_values[0][0], show=True, feature_names=feature_names)

