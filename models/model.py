__author__ = "Koren Gast"
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd


class GenModel(object):
    def __init__(self):
        self.model = None
        self.name = "Generic"

    def fit(self, X, y):
        self.model.fit(X, y)

    def evaluate(self, X, y):
        preds = self.predict(X)[0]
        print(confusion_matrix(y, preds))
        return confusion_matrix(y, preds)

    def evaluate_on_signal(self, X, origin_df, w_size):
        preds, probs = self.predict(X)
        preds_df = pd.DataFrame({'Y_preds': origin_df.shape[0] * [-1],
                                 'Y_probs': origin_df.shape[0] * [0]})
        for i in range(0, origin_df.shape[0] - w_size + 1):
            pred = preds[i]
            prob = max(probs[i])
            for j in range(i, i + w_size):
                if preds_df['Y_probs'].iloc[j] < prob:
                    preds_df.set_value(j, ['Y_preds', 'Y_probs'], [pred, prob])
        print(confusion_matrix(np.array(origin_df['Y']), np.array(
            preds_df['Y_preds'])))
        return confusion_matrix(np.array(origin_df['Y']), np.array(
            preds_df['Y_preds']))

    def predict(self, X):
        return self.model.predict(X), self.model.predict_proba(X)
