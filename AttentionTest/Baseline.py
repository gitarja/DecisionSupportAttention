import numpy as np
from AttentionTest.DataGenerator import Dataset
from AttentionTest.Conf import TENSOR_BOARD_PATH
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics.pairwise import cosine_similarity
import math
import random
random.seed(2021)
fold = 0

generator_train = Dataset(mode="train_val", val_frac=0.2)
generator_test = Dataset(mode="test")
#prepare dataset
x, y, val_x, val_y = generator_train.fetch_all(validation=True)

test_x, test_y = generator_test.fetch_all()


base_model =KNeighborsClassifier( metric="euclidean", algorithm="kd_tree")
parameters = {'n_neighbors': [1, 2, 3]}
clf = GridSearchCV(base_model, parameters)
clf.fit(x, y)
print(clf.best_params_)
model = clf.best_estimator_

#save predictions
model_path = TENSOR_BOARD_PATH + "baseline-knn\\"
preds = model.predict(test_x)
np.savetxt(model_path + "labels_"+str(fold)+".csv", test_y, delimiter=",")
np.savetxt(model_path + "preds_"+str(fold)+".csv", preds , delimiter=",")
#

mcc_score = matthews_corrcoef(test_y, preds)
#print results
print(np.average(model.predict(test_x) == test_y))
print(mcc_score)






# test outlier
outlier_data = Dataset("outlier")
x_out = outlier_data.data[1:]
y_out = np.ones(len(x_out)) * outlier_data.labels

support_outliers = x_out[3:]
support_y_outliers = y_out[3:]

q_outliers = x_out[:3]
q_y_outliers = y_out[:3]


# add to ref
x = np.concatenate([x, support_outliers], 0)
y = np.concatenate([y, support_y_outliers])

# add to query
test_x = np.concatenate([test_x, q_outliers], 0)
test_y = np.concatenate([test_y, q_y_outliers])

model.fit(x, y)
# print acc of new classes only
print(model.score(q_outliers, q_y_outliers))

# print acc and mcc of all claases
preds = model.predict(test_x)
print(preds)
print(np.average(preds == test_y))
mcc_score = matthews_corrcoef(test_y, preds)
print(mcc_score)
