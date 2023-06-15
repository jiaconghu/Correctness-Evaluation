from . import pose_hrnet
from .default import _C as cfg, update_config


class Args:
    def __init__(self):
        self.cfg = 'models/human_pose_estimation/hrnet/w32_256x192_adam_lr1e-3.yaml'
        self.opts = ['TEST.MODEL_FILE', 'models/pytorch/pose_coco/pose_hrnet_w32_256x192.pth', 'TEST.USE_GT_BBOX',
                     'False']
        self.modelDir = ''
        self.logDir = ''
        self.dataDir = ''
        self.prevModelDir = ''


def load_hrnet():
    args = Args()
    update_config(cfg, args)
    model = pose_hrnet.get_pose_net(cfg, is_train=False)
    return model
