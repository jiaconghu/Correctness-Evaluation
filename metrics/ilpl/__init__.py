import copy
import json

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torchmetrics.detection.mean_ap import MeanAveragePrecision


# just for COCO
class COCOMetrics:
    def __init__(self, data_ann_path):
        self.data_ann_path = data_ann_path

    def compute(self, res_list):
        cocoGt = COCO(self.data_ann_path)

        cocoDt = COCO()
        dataset = json.load(open(self.data_ann_path, 'r'))
        cocoDt.dataset['images'] = [img for img in dataset['images']]
        # {"image_id": 397133, "bbox": [375.0799560546875, 70.51233673095703, 125.82656860351562, 276.19359588623047],
         # "score": 0.999484658241272, "category_id": 1},

        cocoDt.dataset['categories'] = copy.deepcopy(dataset['categories'])
        for id, ann in enumerate(res_list):
            bb = ann['bbox']
            x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
            if 'segmentation' not in ann:
                ann['segmentation'] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
            ann['area'] = bb[2] * bb[3]
            ann['id'] = id + 1
            ann['iscrowd'] = 0

        cocoDt.dataset['annotations'] = res_list
        cocoDt.createIndex()

        imgIds = sorted(cocoGt.getImgIds())
        annType = 'bbox'
        print_lrp_components_over_size = False

        # running evaluation
        cocoEval = COCOeval(cocoGt, cocoDt, annType, print_lrp_components_over_size)
        cocoEval.params.imgIds = imgIds
        cocoEval.evaluate()
        cocoEval.accumulate()
        return cocoEval.summarize()
