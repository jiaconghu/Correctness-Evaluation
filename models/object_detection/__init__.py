from .detr import build
import argparse


class Args:
    def __init__(self):
        self.lr = 0.0001
        self.lr_backbone = 1e-05
        self.batch_size = 2
        self.weight_decay = 0.0001
        self.epochs = 300
        self.lr_drop = 200
        self.clip_max_norm = 0.1
        self.frozen_weights = None
        self.backbone = 'resnet50'
        self.dilation = False
        self.position_embedding = 'sine'
        self.enc_layers = 6
        self.dec_layers = 6
        self.dim_feedforward = 2048
        self.hidden_dim = 256
        self.dropout = 0.1
        self.nheads = 8
        self.num_queries = 100
        self.pre_norm = False
        self.masks = False
        self.aux_loss = False
        self.set_cost_class = 1
        self.set_cost_bbox = 5
        self.set_cost_giou = 2
        self.mask_loss_coef = 1
        self.dice_loss_coef = 1
        self.bbox_loss_coef = 5
        self.giou_loss_coef = 2
        self.eos_coef = 0.1
        self.dataset_file = 'coco'
        self.coco_path = '/path/to/coco'
        self.coco_panoptic_path = None
        self.remove_difficult = False
        self.output_dir = ''
        self.device = 'cuda'
        self.seed = 42
        # self.resume = 'https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth'
        self.start_epoch = 0
        self.eval = True
        self.num_workers = 2
        self.world_size = 1
        self.dist_url = 'env://'


def load_detr():
    args = Args()
    return build(args)
