import cv2
import numpy as np
import torchvision.transforms

from metrics import Metric
import torch

"""
confusionMetric  # row: predicted value，clown: actual value
P\L     P    N
P      TP    FP
N      FN    TN
"""


class ConfusionMatrix(Metric):
    def __init__(self, num_classes):
        super(ConfusionMatrix, self).__init__()
        self.num_classes = num_classes
        self.cm = torch.zeros((self.num_classes, self.num_classes))

    def __call__(self, *args, **kwargs):
        return self.update(*args, **kwargs)

    def update(self, outputs, labels):
        outputs = torch.argmax(outputs, dim=1)  # [B, C, H, W] -> [B, H, W]
        outputs = outputs.flatten()  # [B, H, W] -> [B*H*W]
        labels = labels.flatten()  # [B, H, W] -> [B*H*W]

        mask = (labels >= 0) & (labels < self.num_classes)
        inds = self.num_classes * labels[mask] + outputs[mask]
        self.cm += torch.bincount(inds, minlength=self.num_classes ** 2).reshape(self.num_classes, self.num_classes)

    def compute(self):
        return self.cm

    def to(self, device):
        self.cm = self.cm.to(device)
        return self


def iou(cm):
    # IoU = TP / (TP + FP + FN)
    intersection = torch.diag(cm)
    union = torch.sum(cm, dim=1) + torch.sum(cm, dim=0) - torch.diag(cm)
    ious = intersection / union
    return ious


def dice(cm):
    # IoU =  2 TP / (2 TP + FP + FN)
    intersection = 2 * torch.diag(cm)
    union = torch.sum(cm, dim=1) + torch.sum(cm, dim=0)
    ious = intersection / union
    return ious


def m_iou(cm):
    miou = iou(cm).mean()
    return miou

def m_dice(cm):
    mdice = dice(cm).mean()
    return mdice


def fwiou(cm):
    pos = torch.sum(cm, dim=1)
    total = torch.sum(cm)
    freqs = pos / total
    ious = iou(cm)

    print(freqs.shape, ious.shape)

    res = torch.sum(freqs * ious)
    return res


def pixel_accuracy(cm):
    # PA = (TP + TN) / (TP + TN + FP + TN)
    pa = torch.sum(torch.diag(cm)) / torch.sum(cm)
    return pa


"""
copy from: https://github.com/srajanpaliwal/oce_python/blob/master/oce.py
"""


class OCE(Metric):
    def __init__(self, num_classes):
        super(OCE, self).__init__()
        self.num_classes = num_classes
        self.oce_score = torch.tensor(0.0)
        self.num_samples = torch.tensor(0)

    def partial_error(self, g, s):
        """Patial Error function."""

        clust_g = torch.unique(g)
        clust_s = torch.unique(s)
        if clust_g[0] == 0:
            clust_g = clust_g[1:]
        if clust_s[0] == 0:
            clust_s = clust_s[1:]
        err = torch.tensor(0.0).to(g.device)
        if clust_s.size == 0 or clust_g.size == 0:
            return torch.tensor(1)

        for j in clust_g:
            # initializing inner summation is the formaula.
            inner = 0.0

            Aj = (g == j)
            Wj = 0.0
            Wj += torch.sum(Aj, dtype=torch.float) / (torch.sum(g > 0))

            # calculating denominator for Wji
            Wji_den = torch.sum(torch.asarray([(torch.sum(torch.logical_and(Aj, s == x)) != 0)
                                               * torch.sum(s == x) for x in clust_s]), dtype=torch.float)

            # calculating inner summation
            for i in clust_s:
                # Wji
                Bi = (s == i)
                Wji = ((torch.sum(torch.logical_and(Aj, Bi), dtype=torch.float) != 0)
                       * torch.sum(Bi)) / Wji_den

                inner += (torch.sum(torch.logical_and(Aj, Bi), dtype=torch.float) /
                          torch.sum(torch.logical_or(Aj, Bi), dtype=torch.float)) * Wji

            # calculating outer summation
            err += (1 - inner) * Wj
        if torch.isnan(err):
            return torch.tensor(1)
        return err

    def oce(self, gtImage, sImage):
        """Object-level Consistency Error Martin Index."""

        score = torch.min(self.partial_error(gtImage, sImage),
                          self.partial_error(sImage, gtImage))
        return score

    def update(self, outputs, labels):
        # outputs = outputs_.detach().cpu()
        # labels = labels_.detach().cpu()
        outputs = torch.argmax(outputs, dim=1)  # [B, C, H, W] -> [B, H, W]
        mask = (labels >= 0) & (labels < self.num_classes)

        for output, label in zip(outputs * mask, labels * mask):
            self.oce_score += self.oce(label, output)
            self.num_samples += torch.tensor(1)

    def compute(self):
        return self.oce_score / self.num_samples

    def to(self, device):
        self.oce_score = self.oce_score.to(device)
        self.num_samples = self.num_samples.to(device)
        return self


"""
most copy from ChatGPT
"""


class BoundaryIoU(Metric):
    def __init__(self, num_classes):
        super(BoundaryIoU, self).__init__()
        self.num_classes = num_classes
        self.iou = []

    def get_boundary_mask(self, label):
        """
        返回二值边界 mask

        :param label: 真实标签，shape 为 [H, W]
        :return: 二值边界 mask，shape 为 [H, W]
        """
        label = label.astype('uint8')
        kernel = np.ones((3, 3), np.uint8)
        eroded = cv2.erode(label, kernel, iterations=1)
        dilated = cv2.dilate(label, kernel, iterations=1)
        boundary = dilated - eroded
        return np.where(boundary > 0, 1, 0)

    def get_intersection_and_union(self, pred, label, class_id):
        """
        计算每个类别的边界和真实标签的交集和并集

        :param pred: 预测结果，shape 为 [batch_size, H, W]
        :param label: 真实标签，shape 为 [batch_size, H, W]
        :param class_id: 类别标识，从 0 到 num_classes - 1
        :return: 边界和真实标签的交集和并集
        """
        pred_mask = np.where(pred == class_id, 1, 0)
        label_mask = np.where(label == class_id, 1, 0)
        boundary_pred_mask = self.get_boundary_mask(pred_mask)
        boundary_label_mask = self.get_boundary_mask(label_mask)
        intersection = np.logical_and(boundary_pred_mask, boundary_label_mask).sum()
        union = np.logical_or(boundary_pred_mask, boundary_label_mask).sum()
        return intersection, union

    def boundary_iou(self, pred, label):
        """
        计算多类别分割的 Boundary IoU 值

        :param pred: 预测结果，shape 为 [batch_size, H, W]
        :param label: 真实标签，shape 为 [batch_size, H, W]
        :param num_classes: 类别数
        :return: 所有类别边界 IoU 的平均值
        """
        ious = []
        for class_id in range(1, self.num_classes):
            intersection, union = self.get_intersection_and_union(pred, label, class_id)
            iou = intersection / (union + 1e-7)
            ious.append(iou)
        return sum(ious) / self.num_classes

    def update(self, outputs, labels):
        outputs = torch.argmax(outputs, dim=1)  # [B, C, H, W] -> [B, H, W]
        outputs = torchvision.transforms.Resize((500, 500))(outputs)
        labels = torchvision.transforms.Resize((500, 500))(labels)
        outputs = outputs.detach().cpu()
        labels = labels.detach().cpu()
        idx = (labels >= 0) & (labels < self.num_classes)
        # print(outputs.shape)
        self.iou.append(self.boundary_iou(outputs * idx, labels * idx))

    def compute(self):
        return np.asarray(self.iou).mean()


# ====================================


if __name__ == '__main__':
    # gt = torch.asarray([[[0, 0, 0, 0],
    #                      [0, 1, 1, 0],
    #                      [0, 2, 2, 0],
    #                      [0, 2, 2, 0],
    #                      [0, 0, 0, 0]],
    #                     [[0, 0, 0, 0],
    #                      [0, 1, 1, 0],
    #                      [0, 1, 1, 0],
    #                      [0, 2, 2, 0],
    #                      [0, 0, 0, 0]],
    #                     [[0, 0, 0, 0],
    #                      [0, 1, 1, 0],
    #                      [0, 2, 2, 0],
    #                      [0, 1, 1, 0],
    #                      [0, 0, 0, 0]],
    #                     [[0, 0, 0, 0],
    #                      [0, 1, 1, 0],
    #                      [0, 1, 1, 0],
    #                      [0, 2, 5, 0],
    #                      [0, 0, 0, 0]]])
    # pred = torch.asarray([[[0, 0, 0, 0],
    #                        [0, 1, 1, 0],
    #                        [0, 1, 1, 0],
    #                        [0, 1, 9, 0],
    #                        [0, 0, 0, 0]],
    #                       [[0, 0, 0, 0],
    #                        [0, 1, 1, 0],
    #                        [0, 2, 2, 0],
    #                        [0, 1, 1, 0],
    #                        [0, 0, 0, 0]],
    #                       [[0, 0, 0, 0],
    #                        [0, 1, 1, 0],
    #                        [0, 1, 1, 0],
    #                        [0, 2, 2, 0],
    #                        [0, 0, 0, 0]],
    #                       [[0, 0, 0, 0],
    #                        [0, 1, 1, 0],
    #                        [0, 1, 1, 0],
    #                        [0, 2, 3, 0],
    #                        [0, 0, 0, 0]]])
    pred = np.random.randint(0, 21, size=(16, 500, 500), dtype=np.uint8)
    label = np.random.randint(0, 21, size=(16, 500, 500), dtype=np.uint8)
    # print(boundary_iou(pred, pred, 5))
    biou = BoundaryIoU(21)
    print(biou.boundary_iou(pred, label))
