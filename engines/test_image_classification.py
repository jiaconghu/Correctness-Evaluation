import argparse
import os.path
import re
import time
from collections import OrderedDict

import numpy as np
from tqdm import tqdm

import torch
from torch import nn

import loaders
import models
from metrics import ildv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='', type=str, help='model name')
    parser.add_argument('--num_classes', default='', type=int, help='num classes')
    parser.add_argument('--model_path', default='', type=str, help='model path')
    parser.add_argument('--data_name', default='', type=str, help='data name')
    parser.add_argument('--data_dir', default='', type=str, help='data directory')
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
    # model.load_state_dict(torch.load(args.model_path), strict=False)

    # ----------DenseNet------------
    pattern = re.compile(
        r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
    )

    state_dict = torch.load(args.model_path)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)
    # ----------DenseNet------------

    model.to(device)

    test_loader = loaders.load_data(data_name=args.data_name, data_dir=args.data_dir, data_type='test')

    evaluates = {
        'A': [ildv.MulticlassAccuracy(average='macro', num_classes=args.num_classes).to(device),
              ildv.MulticlassF1Score(average='macro', num_classes=args.num_classes).to(device)],
        'B': [],
        'C': [ildv.MulticlassBalancedAccuracy(args.num_classes).to(device),
              ildv.MulticlassOptimizedPrecision(args.num_classes).to(device)]
    }

    # ----------------------------------------
    # each epoch
    # ----------------------------------------
    since = time.time()

    test_type = os.path.split(args.data_dir)[1]

    scores = test(test_loader, model, evaluates, device)
    # save_path = os.path.join(args.save_dir, '{}_{}_{}.npy'.format(args.model_name, args.data_name, test_type))
    save_path = os.path.join(args.save_dir, '{}_{}.npy'.format(args.model_name, args.data_name))
    np.save(save_path, scores)

    print('-' * 50)
    print('TIME CONSUMED', time.time() - since)


def test(test_loader, model, evaluates, device):
    model.eval()

    for i, samples in enumerate(tqdm(test_loader)):
        inputs, labels = samples
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)

        for value in evaluates.values():
            for evaluate in value:
                evaluate.update(outputs, labels)

    # calculate result
    scores = {}
    for key, value in zip(evaluates.keys(), evaluates.values()):
        scores[key] = []
        for evaluate in value:
            score = evaluate.compute().item()
            scores[key].append(score)

    print(scores)

    return scores


if __name__ == '__main__':
    main()
