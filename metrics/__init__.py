# from metrics import pldv
# from metrics import plcv
# from metrics import ildv
import torch


class Metric:
    def __init__(self):
        super().__init__()
        self.device = torch.device('cpu')

    def __call__(self, *args, **kwargs):
        return self

    def update(self, outputs, labels):
        pass

    def compute(self):
        pass

    def to(self, device):
        return self
