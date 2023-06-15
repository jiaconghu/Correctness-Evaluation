import torch

from models.image_generation.nvae import model
from models.image_generation.nvae import utils
import argparse


def load_nvae():
    # args = argparse.ArgumentParser()
    # args.
    model_path = '/nfs3-p1/hjc/pretraind_models/checkpoints/nvae_cifar10_checkpoint.pt'
    checkpoint = torch.load(model_path)
    args = checkpoint['args']

    if not hasattr(args, 'ada_groups'):
        args.ada_groups = False

    if not hasattr(args, 'min_groups_per_scale'):
        args.min_groups_per_scale = 1

    if not hasattr(args, 'num_mixture_dec'):
        args.num_mixture_dec = 10

    arch_instance = utils.get_arch_cells(args.arch_instance)
    ae = model.AutoEncoder(args, None, arch_instance)
    ae.load_state_dict(checkpoint['state_dict'], strict=False)

    return ae
