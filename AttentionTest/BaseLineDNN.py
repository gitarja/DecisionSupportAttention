from NNModels.SiameseModel import BaseLine
from kerastuner.tuners import Hyperband
from AttentionTest.DataGenerator import Dataset
from AttentionTest.Conf import TENSOR_BOARD_PATH
fold = 0
from sklearn.metrics import matthews_corrcoef
import tensorflow as tf
import numpy as np
generator_train = Dataset(mode="train_val", val_frac=0.2)
generator_test = Dataset(mode="test")
#prepare dataset
x, y, val_x, val_y = generator_train.fetch_all(validation=True)

test_x, test_y = generator_train.fetch_all(validation=False)


baseline = BaseLine()

tuner = Hyperband(
    baseline.build_model,
    objective='val_accuracy',
    max_epochs=15,
    directory='.',
    project_name='fold0', seed=2021)

tuner.search(x, y,
             epochs=15,
             validation_data=(val_x, val_y))


tuner.results_summary()

model = tuner.get_best_models(num_models=1)[0]
model_path = TENSOR_BOARD_PATH + "baseline\\"
# model.save(model_path)

#save predictions
preds = model.predict(test_x)
np.savetxt(model_path + "labels_"+str(fold)+".csv", test_y, delimiter=",")
np.savetxt(model_path + "preds_"+str(fold)+".csv", preds , delimiter=",")
#

mcc_score = matthews_corrcoef(test_y, preds >= 0.5)
#print results
print(mcc_score)
print(model.evaluate(test_x, test_y))

