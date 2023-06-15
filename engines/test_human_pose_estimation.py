import argparse
import time
from tqdm import tqdm

import torch
from torch import nn

import loaders
import models
# import metrics
from torcheval import metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='', type=str, help='model name')
    parser.add_argument('--num_classes', default='', type=int, help='num classes')
    parser.add_argument('--model_path', default='', type=str, help='model path')
    parser.add_argument('--data_name', default='', type=str, help='data name')
    parser.add_argument('--data_dir', default='', type=str, help='data dir')
    parser.add_argument('--data_ann_path', default='', type=str, help='ann path')
    args = parser.parse_args()

    # ----------------------------------------
    # basic configuration
    # ----------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print('-' * 50)
    print('DEVICE:', device)
    print('MODEL PATH:', args.model_path)
    print('DATA DIR:', args.data_dir)
    print('DATA ANN PATH:', args.data_ann_path)
    print('-' * 50)

    # ----------------------------------------
    # test configuration
    # ----------------------------------------
    model = models.load_model(model_name=args.model_name, num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.model_path))
    model.to(device)

    test_loader = loaders.load_estimation_data(data_name=args.data_name, data_dir=args.data_dir, data_type='test')
    print(len(test_loader))

    evaluates = [
        metrics.MulticlassAccuracy().to(device),
        metrics.MulticlassF1Score().to(device),
        metrics.MulticlassAUROC(num_classes=args.num_classes).to(device)
    ]

    # ----------------------------------------
    # each epoch
    # ----------------------------------------
    since = time.time()

    # test(test_loader, model, evaluates, device)

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

        for evaluate in evaluates:
            evaluate.update(outputs, )

    for evaluate in evaluates:
        score = evaluate.compute()
        print(score)


if __name__ == '__main__':
    main()
