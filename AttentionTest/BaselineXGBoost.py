from AttentionTest.DataGenerator import Dataset
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import matthews_corrcoef
import random
import xgboost as xgb
import numpy as np
random.seed(2021)
fold = 0

generator_train = Dataset(mode="train_val", val_frac=0.2)
generator_test = Dataset(mode="test")
#prepare dataset
x, y, val_x, val_y = generator_train.fetch_all(validation=True)

test_x, test_y = generator_test.fetch_all()


parameters = {'objective':['binary:logistic'],
              'eta': [0.5], #so called `eta` value
              'max_depth': [3, 5, 7],
              'n_estimators': [3, 5, 7], #number of trees, change it to 1000 for better results
              'seed': [1337]}

xgb_model = xgb.XGBClassifier(verbosity=0, use_label_encoder=False)
clf = GridSearchCV(xgb_model, parameters)

clf.fit(x, y)

model = clf.best_estimator_

#save predictions
preds = model.predict(test_x)

mcc_score = matthews_corrcoef(test_y, preds)
#print results
print(mcc_score)
print(np.average(model.predict(test_x) == test_y))

