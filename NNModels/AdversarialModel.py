import tensorflow.keras as K

class DiscriminatorModel(K.models.Model):

    def __init__(self, n_hidden, n_output, dropout_rate):
        super(DiscriminatorModel, self).__init__()

        #first hidden layer
        self.dense_1 = K.layers.Dense(units=n_hidden, activation="elu", name="dense_1")
        self.dense_2 = K.layers.Dense(units=n_hidden, activation="elu", name="dense_2")

        #dropout
        self.dropout_1 = K.layers.Dropout(rate=dropout_rate)

        #logit
        self.dense_logit = K.layers.Dense(units=n_output, activation="sigmoid", name="dense_logit")



    def call(self, inputs, training=None, mask=None):

        x = self.dropout_1(self.dense_1(inputs))
        x = self.dropout_1(self.dense_2(x))

        logit = self.dense_logit(x)

        return logit

