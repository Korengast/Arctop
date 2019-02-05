__author__ = "Koren Gast"
from models.model import GenModel
from sklearn.ensemble import RandomForestClassifier


class RandomForest(GenModel):
    def __init__(self, n_estimators=100):
        super().__init__()
        self.model = RandomForestClassifier(n_estimators=n_estimators, n_jobs=-1)
        self.name = "Random forest"
