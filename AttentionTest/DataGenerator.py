import glob
import pandas as pd
import numpy as np
from AttentionTest.Conf import DATASET_PATH, N_FEATURES
import random
import pickle
from sklearn.model_selection import  train_test_split

class Dataset:

    def __init__(self, mode="training", val_frac=0.1):
        self.i = 0

        if mode == "train" or mode == "train_val":
            with open(DATASET_PATH + "data_train.pickle", 'rb') as f:
                ds = pickle.load(f)
        elif mode == "test":
            with open(DATASET_PATH + "data_test.pickle", 'rb') as f:
                ds = pickle.load(f)
        elif mode == "outlier":

            ds = np.load(DATASET_PATH + "data_adults.npy")

        self.data = ds
        self.val_frac = val_frac

        if mode == "outlier":
            self.labels = [2]
        else:
            self.labels = list(self.data.keys())

    def fetch_all(self, validation=False):
        inputs_data = []
        inputs_label = []
        for i in range(len(self.labels)):
            for j in range(len(self.data[self.labels[i]])):
                inputs_data.append(self.data[self.labels[i]][j])
                inputs_label.append(i)

        inputs_data = np.array(inputs_data)
        inputs_label = np.array(inputs_label)

        if validation:

            X_train, X_test, y_train, y_test = train_test_split(inputs_data, inputs_label, test_size=self.val_frac, random_state=2021, stratify=inputs_label)
            train_reindex = np.argsort(y_train)
            test_reindex = np.argsort(y_test)
            return X_train[train_reindex], y_train[train_reindex], X_test[test_reindex], y_test[test_reindex]
        else:
            return inputs_data.astype(np.float32), inputs_label.astype(np.float32)

    def get_mini_offline_batches(self, X_train, y_train, n_class=2, shots=2):


        anchors = np.zeros(shape=(n_class * shots, N_FEATURES))
        labels = np.zeros(shape=(n_class * shots, 1))
        for i in range(n_class):
            if shots > len(X_train[y_train==i].tolist()):
                positive_to_split = random.choices(
                    X_train[y_train == i].tolist(), k=shots)
            else:
                positive_to_split = random.sample(
                    X_train[y_train==i].tolist(), k=shots)
                 # set anchor and pair positives
            anchors[i * shots:(i + 1) * shots] = positive_to_split
            labels[i * shots:(i + 1) * shots] = i


        return anchors.astype(np.float32), labels.astype(np.float32)


    def get_batches(self, X_train, y_train, shots, n_class, n_query=1): #get data for training
        anchor_data = np.zeros(shape=(n_class * n_query, N_FEATURES))
        anchor_labels = np.zeros(shape=(n_class * n_query))

        support_data = np.zeros(shape=(n_class * shots, N_FEATURES))
        support_labels = np.zeros(shape=(n_class * shots))

        for i in range(n_class):
            if shots > len(X_train[y_train == i].tolist()):
                positive_to_split = random.choices(
                    X_train[y_train == i].tolist(), k=shots + n_query)
            else:
                positive_to_split = random.sample(
                    X_train[y_train == i].tolist(), k=shots + n_query)
                # set anchor and pair positives
            anchor_data[i * n_query:(i + 1) * n_query] = positive_to_split[:n_query]
            anchor_labels[i * n_query:(i + 1) * n_query] = i
            support_data[i * shots:(i + 1) * shots] = positive_to_split[n_query:]
            support_labels[i * shots:(i + 1) * shots] = i

        return (anchor_data, anchor_labels), (support_data, support_labels)

if __name__ == '__main__':
    test_dataset = Dataset(mode="train_val")
    X_train, y_train, X_test, y_test = test_dataset.fetch_all(validation=True)

    anchors, supports = test_dataset.get_batches(X_train, y_train, n_class=2, shots=3, n_query=1)

    print(anchors[0].shape)
    print(supports[0].shape)




