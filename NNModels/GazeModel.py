from tensorflow import keras as K


class GazeModel(K.models.Model):


    def __init__(self, output=2):
        super(GazeModel, self).__init__()

        # self.rnn1 = K.layers.LSTM(units=64, return_sequences=True, stateful=True, name="rnn1", recurrent_dropout=0.15, dropout=0.35)
        # self.rnn2 = K.layers.LSTM(units=128, return_sequences=False, stateful=True, name="rnn2", recurrent_dropout=0.15, dropout=0.35)
        self.dense1 = K.layers.Dense(units=256, activation="elu")
        self.dense2 = K.layers.Dense(units=512, activation="elu")
        self.logit = K.layers.Dense(units=output, activation=None)

        self.gaze_activation = K.layers.ReLU(max_value=1.)

        #dropout
        self.dropout = K.layers.Dropout(0.1)



    def call(self, inputs, training=None, mask=None):

        x = self.dense1(inputs)
        x = self.dense2(x)
        # x = self.rnn1(inputs)
        # x = self.rnn2(x)
        z = K.activations.sigmoid(self.logit(x))

        return z



class GazeEnsembleModel(K.models.Model):


    def __init__(self, output=2):
        super(GazeEnsembleModel, self).__init__()

        #first model
        self.dense1_1 = K.layers.Dense(units=32, activation="elu", name="dense1_1")
        self.dense1_2 = K.layers.Dense(units=64, activation="elu", name="dense1_2")
        self.dense1_3 = K.layers.Dense(units=128, activation="elu", name="dense1_3")
        self.dense1_4 = K.layers.Dense(units=256, activation="elu", name="dense1_4")
        #second model
        self.dense2_1 = K.layers.Dense(units=256, activation="elu", name="dense1_1")
        self.dense2_2 = K.layers.Dense(units=512, activation="elu", name="dense1_2")
        self.dense2_3 = K.layers.Dense(units=1024, activation="elu", name="dense2_3")
        #third model
        self.dense3_1 = K.layers.Dense(units=1024, activation="elu", name="dense1_1")
        self.dense3_2 = K.layers.Dense(units=2048, activation="elu", name="dense3_2")

        self.output1 = K.layers.Dense(units=output, activation="sigmoid", name="output1")
        self.output2 = K.layers.Dense(units=output, activation="sigmoid", name="output2")
        self.output3 = K.layers.Dense(units=output, activation="sigmoid", name="output3")

        self.average = K.layers.Average()


    def call(self, inputs, training=None, mask=None):

        #first model
        x1 = self.dense1_1(inputs)
        x1 = self.dense1_2(x1)
        x1 = self.dense1_3(x1)
        x1 = self.dense1_4(x1)
        z1 = self.output1(x1)

        #second model
        x2 = self.dense2_1(inputs)
        x2 = self.dense2_2(x2)
        x2 = self.dense2_3(x2)
        z2 = self.output2(x2)

        #third model
        x3 = self.dense3_1(inputs)
        x3 = self.dense3_2(x3)
        z3 = self.output3(x3)

        return self.average([z1, z2, z3])