import argparse
import os
import time

import numpy as np
from tqdm import tqdm

import torch

import loaders
import models
from metrics import pldv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='', type=str, help='model name')
    parser.add_argument('--num_classes', default='', type=int, help='num classes')
    parser.add_argument('--model_path', default='', type=str, help='model path')
    parser.add_argument('--data_name', default='', type=str, help='data name')
    parser.add_argument('--data_dir', default='', type=str, help='data dir')
    parser.add_argument('--data_ann_path', default='', type=str, help='ann path')
    parser.add_argument('--save_dir', default='', type=str, help='save directory')
    args = parser.parse_args()

    # ----------------------------------------
    # basic configuration
    # ----------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    print('-' * 50)
    print('DEVICE:', device)
    print('MODEL PATH:', args.model_path)
    print('DATA PATH:', args.data_dir)
    print('-' * 50)

    # ----------------------------------------
    # test configuration
    # ----------------------------------------
    model = models.load_model(model_name=args.model_name, num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    test_loader = loaders.load_segmentation_data(data_name=args.data_name,
                                                 data_dir=args.data_dir,
                                                 data_ann_path=args.data_ann_path,
                                                 data_type='val')

    evaluates = [
        pldv.ConfusionMatrix(args.num_classes).to(device),
        pldv.OCE(args.num_classes).to(device),
        pldv.BoundaryIoU(args.num_classes)
    ]

    # ----------------------------------------
    # each epoch
    # ----------------------------------------
    since = time.time()

    scores = test(test_loader, model, evaluates, device)
    save_path = os.path.join(args.save_dir, '{}_{}.npy'.format(args.model_name, args.data_name))
    np.save(save_path, scores)

    print('-' * 50)
    print('TIME CONSUMED', time.time() - since)


# torch.set_printoptions(profile="full")


def test(test_loader, model, evaluates, device):
    model.eval()

    for i, samples in enumerate(tqdm(test_loader)):
        inputs, labels = samples
        inputs = inputs.to(device)  # [B, D, H, W]
        labels = labels.to(device)  # [B, H, W]

        with torch.no_grad():
            outputs = model(inputs)  # [B, C, H, W]

        for evaluate in evaluates:
            evaluate.update(outputs['out'], )

    scores = {}
    for i, evaluate in enumerate(evaluates):
        if i == 0:
            cm = evaluate.compute()
            scores['A'] = [pldv.m_iou(cm), pldv.m_dice(cm)]
            scores['C'] = [pldv.fwiou(cm)]
        elif i == 1:
            oce = evaluate.compute()
            scores['B'] = [oce]
        elif i == 2:
            biou = evaluate.compute()
            scores['I'] = [biou]

    print(scores)
    return scores


if __name__ == '__main__':
    main()
