from loaders.classification import imagenet
from loaders.segmentation import coco as s_coco
from loaders.detection import coco as d_coco


# print('-' * 50)
# print('LOAD DATA:', data_name)
# print('DATA DIR:', data_dir)
# print('DATA TYPE:', data_type)
# print('-' * 50)


def load_data(data_name, data_dir, data_type):
    if data_name == 'ImageNet':
        return imagenet.load(data_dir, data_type)


def load_segmentation_data(data_name, data_dir, data_ann_path, data_type):
    if data_name == 'COCO':
        return s_coco.load(data_dir, data_ann_path, data_type)


def load_detection_data(data_name, data_dir, data_ann_path, data_type):
    if data_name == 'COCO':
        return d_coco.load(data_dir, data_ann_path, data_type)


# def load_estimation_data(data_name, data_dir, data_ann_path, data_type):
#     if data_name == 'COCO':
#         return e_coco.load(data_dir, data_ann_path, data_type)
