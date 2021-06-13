from DNNModels.NNModels.SiameseModel import BaseLine
from kerastuner.tuners import Hyperband
from Conf.Settings import FEATURES_PATH
import tensorflow as tf
import numpy as np
import tensorflow_addons as tfa

fold = 0


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

#prepare dataset
x = tf.convert_to_tensor(np.load(training_features_file), dtype=tf.float32)
y = tf.convert_to_tensor(np.load(training_labels_file), dtype=tf.float32)

val_x = tf.convert_to_tensor(np.load(val_features_file), dtype=tf.float32)
val_y = tf.convert_to_tensor(np.load(val_labels_file), dtype=tf.float32)

test_x = tf.convert_to_tensor(np.load(testing_features_file), dtype=tf.float32)
test_y = tf.convert_to_tensor(np.load(testing_labels_file), dtype=tf.float32)
baseline = BaseLine()

tuner = Hyperband(
    baseline.build_model,
    objective='val_accuracy',
    max_epochs=10,
    directory='.',
    project_name='fold0')

tuner.search(x, y,
             epochs=10,
             validation_data=(val_x, val_y))


tuner.results_summary()

model = tuner.get_best_models(num_models=1)[0]

#save predictions
np.savetxt("labels_"+str(fold)+".csv", test_y.numpy(), delimiter=",")
np.savetxt("preds_"+str(fold)+".csv", model.predict(test_x) >= 0.5, delimiter=",")

labels = tf.keras.utils.to_categorical(test_y.numpy())
preds = tf.keras.utils.to_categorical(model.predict(test_x) >= 0.5)

mcc_metric = tfa.metrics.MatthewsCorrelationCoefficient(num_classes=2)
mcc_metric.update_state(labels, preds)
mcc_score = mcc_metric.result().numpy()
#print results
print(mcc_score)
print(model.evaluate(test_x, test_y))
