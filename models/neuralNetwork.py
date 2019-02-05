__author__ = "Koren Gast"
from models.model import GenModel
from sklearn.neural_network import MLPClassifier


class MLP(GenModel):
    def __init__(self, layers_sizes = [64, 16, ]):
        super().__init__()
        self.model = MLPClassifier(hidden_layer_sizes=layers_sizes, max_iter=1000)
        self.name = "MLP"