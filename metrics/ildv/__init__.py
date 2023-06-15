import torch
from metrics import Metric

from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score
from torchmetrics import Accuracy, Recall, Specificity


class MulticlassBalancedAccuracy(Metric):
    def __init__(self, num_classes):
        super().__init__()
        self.sensitivity = Recall(task='multiclass', average='micro', num_classes=num_classes)
        self.specificity = Specificity(task='multiclass', average='micro', num_classes=num_classes)

    def update(self, outputs, targets):
        self.sensitivity.update(outputs, targets)
        self.specificity.update(outputs, targets)

    def compute(self):
        ba = (self.sensitivity.compute() + self.specificity.compute()) / 2
        return ba

    def to(self, device):
        self.sensitivity.to(device)
        self.specificity.to(device)
        return self


class MulticlassOptimizedPrecision(Metric):
    def __init__(self, num_classes):
        super().__init__()
        self.accuracy = Accuracy(task='multiclass', average='micro', num_classes=num_classes)
        self.sensitivity = Recall(task='multiclass', average='micro', num_classes=num_classes)
        self.specificity = Specificity(task='multiclass', average='micro', num_classes=num_classes)

    def update(self, outputs, targets):
        self.accuracy.update(outputs, targets)
        self.sensitivity.update(outputs, targets)
        self.specificity.update(outputs, targets)

    def compute(self):
        sensitivity = self.sensitivity.compute()
        specificity = self.specificity.compute()
        op = self.accuracy.compute() - torch.abs(sensitivity - specificity) / (sensitivity + specificity)
        return op

    def to(self, device):
        self.accuracy.to(device)
        self.sensitivity.to(device)
        self.specificity.to(device)
        return self
