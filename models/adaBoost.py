__author__ = "Koren Gast"
from models.model import GenModel
from sklearn.ensemble import AdaBoostClassifier


class AdaBoost(GenModel):
    def __init__(self, n_estimators=100):
        super().__init__()
        self.model = AdaBoostClassifier(n_estimators=n_estimators)
        self.name = "Random forest"
