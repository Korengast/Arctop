__author__ = "Koren Gast"
from models.model import GenModel
from keras.layers import Dense, Input, LSTM
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler

class LSTM_model(GenModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.args = kwargs
        self.model = self.build()
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.name = "LSTM"

    def build(self):
        inputs = Input(shape=(1,))
        # X = LSTM(4)(inputs)
        # outputs = Dense(3, activation='softmax')(X)
        model = Model(inputs=inputs, outputs=inputs)
        print(model.summary())
        return model

    def fit(self, X, y, epochs=1, batch_size=15):
        X = self.scaler.fit_transform(X)
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        X = self.scaler.transform(X)
        print(self.model.predict(X))
