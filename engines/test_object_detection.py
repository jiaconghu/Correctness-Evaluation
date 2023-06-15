import argparse
import os
import time

import numpy as np
from tqdm import tqdm

import torch

import loaders
import models
from metrics import ilpl


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
    print('DATA DIR:', args.data_dir)
    print('DATA ANN PATH:', args.data_ann_path)
    print('-' * 50)

    # ----------------------------------------
    # test configuration
    # ----------------------------------------

    test_loader = loaders.load_detection_data(data_name=args.data_name,
                                              data_dir=args.data_dir,
                                              data_ann_path=args.data_ann_path,
                                              data_type='val')

    evaluates = [
        ilpl.MeanAveragePrecision()
    ]
    cocoeval = ilpl.COCOMetrics(args.data_ann_path)

    # ----------------------------------------
    # each epoch
    # ----------------------------------------
    since = time.time()
    model = models.load_model(model_name=args.model_name, num_classes=args.num_classes)
    model.to(device)

    if args.model_name == 'DETR_DC5_ResNet50':
        model.load_state_dict(torch.load(args.model_path)['model'])
        scores = test_detr(test_loader, model, evaluates, device)
    else:
        model.load_state_dict(torch.load(args.model_path))
        scores = test(test_loader, model, evaluates, cocoeval, device)

    save_path = os.path.join(args.save_dir, '{}_{}.npy'.format(args.model_name, args.data_name))
    np.save(save_path, scores)  # 注意带上后缀名

    print('-' * 50)
    print('TIME CONSUMED', time.time() - since)


def test_detr(test_loader, model, evaluates, device):
    model.eval()

    for samples in tqdm(test_loader):
        inputs, labels = samples

        inputs = list(img.to(device) for img in inputs)
        labels = [{k: v.to(device) for k, v in t.items()} for t in labels]
        # print(torch.tensor(inputs).shape)
        # print(len(labels))

        with torch.no_grad():
            outputs = model(inputs)
        # print(outputs.keys())
        # print(outputs)
        # "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
        # "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

        # -------- boxes --------
        pred_boxes = outputs['pred_boxes']
        x_c, y_c, w, h = torch.unbind(pred_boxes, dim=2)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        b = torch.stack(b, dim=2)
        r = torch.tensor([[img.shape[2], img.shape[1], img.shape[2], img.shape[1]] for img in inputs],
                         dtype=torch.float32).unsqueeze(1).to(device)
        # print(r)
        pred_boxes = b * r
        # print(pred_boxes)
        # print(pred_boxes.shape)

        # -------- logits --------
        pred_logits = outputs['pred_logits']
        # print(pred_logits.shape)
        pred_logits = torch.softmax(pred_logits, dim=2)[:, :, :-1]
        pred_labels = torch.argmax(pred_logits, dim=2)
        pred_scores, _ = torch.max(pred_logits, dim=2)
        # print(pred_labels.shape)
        # print(pred_scores.shape)
        keep = pred_scores > 0.7
        # print(keep.shape)
        # print(keep.sum(-1))
        preds = []
        for i in range(len(pred_boxes)):
            pred = {
                'boxes': pred_boxes[i][keep[i]],
                'labels': pred_labels[i][keep[i]],
                'scores': pred_scores[i][keep[i]]
            }
            preds.append(pred)

        # print(preds[0])
        # print(labels[0])
        # print('-' * 20)

        for evaluate in evaluates:
            evaluate.update(preds, )

    scores = []
    for evaluate in evaluates:
        score = evaluate.compute()
        scores.append(score)

    return scores


def test(test_loader, model, evaluates, cocoeval, device):
    model.eval()

    res_list = []
    for i, samples in enumerate(tqdm(test_loader)):
        inputs, labels = samples

        inputs = list(img.to(device) for img in inputs)
        labels = [{k: v.to(device) for k, v in t.items()} for t in labels]
        # print(torch.tensor(inputs).shape)
        # print(labels[0])

        with torch.no_grad():
            outputs = model(inputs)

        outputs = [{k: v for k, v in t.items()} for t in outputs]

        # ================== prepare for LRP Error in cocoeval.py ==================

        def prepare_for_coco_detection(predictions):
            coco_results = []
            for original_id, prediction in predictions.items():
                if len(prediction) == 0:
                    continue

                boxes = prediction["boxes"]
                boxes = convert_to_xywh(boxes).tolist()
                scores = prediction["scores"].tolist()
                labels = prediction["labels"].tolist()

                coco_results.extend(
                    [
                        {
                            "image_id": original_id,
                            "category_id": labels[k],
                            "bbox": box,
                            "score": scores[k],
                        }
                        for k, box in enumerate(boxes)
                    ]
                )
            return coco_results

        def convert_to_xywh(boxes):
            xmin, ymin, xmax, ymax = boxes.unbind(1)
            return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

        res = {target["image_id"].item(): output for target, output in zip(labels, outputs)}
        results = prepare_for_coco_detection(res)
        res_list.extend(results)

        # ================== prepare for LRP Error in cocoeval.py ==================

        # print(results)
        # print('-' * 20)

        for evaluate in evaluates:
            evaluate.update(outputs, )

    cocoeval.compute(res_list)

    scores = []
    for evaluate in evaluates:
        score = evaluate.compute()
        scores.append(score)

    return scores


if __name__ == '__main__':
    main()
