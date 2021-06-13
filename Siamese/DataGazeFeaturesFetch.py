import glob
import pandas as pd
import numpy as np
import random

class DataGenerator:

    def __init__(self, features_file, labels_file, offline=False, double=False, training=False):
        self.i = 0

        X = np.load(features_file)
        y = np.load(labels_file)


        if training:
            idx = np.arange(0, len(X))
            random.shuffle(idx)
            idx = idx[:len(X)//4]
            X = X[idx]
            y = y[idx]

        if offline:
            if double:
                self.data = self.readTripletOffline(X, y)
            else:
                self.data = self.readOffline(X, y)
        else:
            self.data = self.readData(X, y)

        self.data_n = len( self.data)



    def fetch(self):
        i = 0
        while i < len(self.data):
            data_i = self.data[i]
            yield data_i[0], data_i[1] #return data and the label
            i+=1

    def fetch_offline(self):
        i = 0

        while i < len(self.data):
            data_i = self.data[i]
            yield data_i[0], data_i[1], data_i[2]  # return anchor, positif, negative
            i += 1


    def fetch_double_triplet(self):
        i = 0

        while i < len(self.data):
            data_i = self.data[i]
            yield data_i[0], data_i[1], data_i[2], data_i[3]  # return anchor, positif, negative achor and the negative
            i += 1

    def readData(self, X, y):
        data = []

        for i in range(len(X)):
            data.append([X[i], y[i]])
        return data


    def readOffline(self, X, y):
        data = []
        X_positif = X[y==1]
        X_negatif = X[y==0]

        for i in range(len(X_positif) - 1):
            for k in range(i + 1, len(X_positif)):
                for j in range(len(X_negatif)):
                    data.append([X_positif[i], X_positif[k], X_negatif[j]])
        for i in range(len(X_negatif) - 1):
            for k in range(i + 1, len(X_negatif)):
                for j in range(len(X_positif)):
                    data.append([X_negatif[i], X_negatif[k], X_positif[j]])
        return data

    def readTripletOffline(self, X, y):
        data = []
        X_positif = X[y==1]
        X_negatif = X[y==0]
        for i in range(len(X_positif) - 1):
            for k in range(i+1, len(X_positif)):
                for j in range(len(X_negatif)):
                    for l in range(j+1, len(X_negatif)):
                        data.append([X_positif[i], X_positif[k], X_negatif[j], X_negatif[l]])

        return data





