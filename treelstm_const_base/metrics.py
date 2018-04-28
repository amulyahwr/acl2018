from copy import deepcopy

import torch
import scipy.stats

class Metrics():
    def __init__(self, num_classes):
        self.num_classes = num_classes

    def pearson(self, predictions, labels):
        x = deepcopy(predictions)
        y = deepcopy(labels)
        x = (x - x.mean()) / x.std()
        y = (y - y.mean()) / y.std()
        return torch.mean(torch.mul(x,y))

    def spearmann(self,predictions,labels):
        x = deepcopy(predictions.numpy())
        y = deepcopy(labels.numpy())
        return scipy.stats.stats.spearmanr(x, y)[0]


    def mse(self, predictions, labels):
        x = deepcopy(predictions)
        y = deepcopy(labels)
        return torch.mean((x-y)**2)