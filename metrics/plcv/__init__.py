import torch
from torch import Tensor
from metrics import Metric
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.fid import FrechetInceptionDistance
from . import swd, ndb

from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm


def _get_transform(data_name):
    if data_name == 'CIFAR10':
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])


def _get_data_loader(data_name, data_dir):
    data = datasets.ImageFolder(root=data_dir,
                                transform=_get_transform(data_name))

    data_loader = DataLoader(dataset=data,
                             batch_size=512,
                             num_workers=4,
                             shuffle=False)
    return data_loader


class FID(FrechetInceptionDistance):
    def __init__(self, data_name, data_dir, feature=2048, reset_real_features=True, normalize=True):
        super(FID, self).__init__(feature, reset_real_features, normalize)

        data_loader = _get_data_loader(data_name, data_dir)
        for inputs, _ in tqdm(data_loader, desc='load the real img'):
            inputs.to(self.device)
            self.update(inputs, real=True)

    def update(self, imgs: Tensor, real=False) -> None:
        super(FID, self).update(imgs, real)


class SWD(Metric):
    """
    copy from https://github.com/koshian2/swd-pytorch
    """

    def __init__(self, data_name, data_dir):
        super().__init__()
        self.inputs = None
        self.outputs = None
        data_loader = _get_data_loader(data_name, data_dir)
        for inputs, _ in tqdm(data_loader, desc='load the real img'):
            if self.inputs is None:
                self.inputs = inputs
            else:
                self.inputs = torch.concatenate((self.inputs, inputs), dim=0)
        print(self.inputs.shape)

    def update(self, outputs, **kwargs):
        if self.outputs is None:
            self.outputs = outputs
        else:
            self.outputs = torch.concatenate((self.outputs, outputs), dim=0)

    def compute(self):
        print('SWD', self.outputs.shape)
        out = swd.swd(self.inputs, self.outputs, device=self.device)
        return out

    def to(self, device):
        self.device = device


class NDB(Metric):
    """
    https://github.com/eitanrich/gans-n-gmms/blob/master/ndb_mnist_demo.py
    """

    def __init__(self, data_name, data_dir):
        super().__init__()
        self.inputs = None
        self.outputs = None
        data_loader = _get_data_loader(data_name, data_dir)
        for inputs, _ in tqdm(data_loader, desc='load the real img'):
            if self.inputs is None:
                self.inputs = inputs
            else:
                self.inputs = torch.concatenate((self.inputs, inputs), dim=0)
        self.inputs = self.inputs.reshape((self.inputs.shape[0], -1)).numpy()
        print(self.inputs.shape)

        self.ndb = ndb.NDB(training_data=self.inputs, number_of_bins=100, whitening=True)

    def update(self, outputs, **kwargs):
        if self.outputs is None:
            self.outputs = outputs
        else:
            self.outputs = torch.concatenate((self.outputs, outputs), dim=0)

    def compute(self):
        self.outputs = self.outputs.reshape((self.outputs.shape[0], -1)).cpu().numpy()
        out = self.ndb.evaluate(self.outputs)
        return out['NDB']

    def to(self, device):
        self.device = device
        return self


if __name__ == '__main__':
    fid = FID('CIFAR10', '/nfs3-p1/hjc/datasets/cifar10/train', feature=64)
    data_loader = _get_data_loader('CIFAR10', '/nfs3-p1/hjc/datasets/cifar10/test')
    for inputs, _ in tqdm(data_loader, desc='load the fake img'):
        inputs.to(torch.device('cuda'))
        fid.update(inputs, real=False)
    print('===>', fid.compute())
