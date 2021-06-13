import glob
import pandas as pd
import numpy as np


class DataFetch:

    def __init__(self, training_path=None, prefix=None, prefix_game = None, turn_file=None):
        self.i = 0

        self.turn_file = pd.read_csv(turn_file)[["Character"]].values

        self.files = glob.glob(training_path + "*"+prefix)
        self.prefix_game = prefix_game
        self.prefix = prefix
        self.len_files = len(self.files)-1




    def loadData(self):

        for i in range(self.len_files):
            data, data_game = self.readData(i)
            for j in range(1, len(data)):
                past = data[j - 1]
                current = data[j]
                #when stimulus appears
                type_stimulus = [-1]
                if current[-1] != -1:
                    type_stimulus = self.turn_file[np.argmin(np.abs(data_game[:, 0] - current[0]))]
                noise = np.random.normal(0, 0.1, 2)
                X = np.concatenate([past[1:3] + noise, past[3:], current[3:], type_stimulus])
                # X = np.where(X==-1, 0, X)
                y = current[1:3]
                yield j, X, y

            #return True


    def readData(self, i=-1):
        if i == -1:
            i = self.i
        file = pd.read_csv(self.files[i])
        file = file.iloc[file.index % 4 == 1]
        file_game = pd.read_csv(self.files[i].replace(self.prefix, self.prefix_game))
        data = file[file["Time"] > 0][["Time", "GazeX", "GazeY", "ObjectX", "ObjectY"]].values
        data_game = file_game[["SpawnTime", "ResponseTime"]].values
        return data, data_game

    def next(self):
        self.i +=1
        self.readData()



    def reset(self):
        self.i = 0
        self.readData()

