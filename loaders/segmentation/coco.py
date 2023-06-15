import copy
import torch
import torch.utils.data
import torchvision
from PIL import Image
from pycocotools import mask as coco_mask
from torch.utils.data import DataLoader
from loaders.segmentation import transforms as T


class FilterAndRemapCocoCategories(object):
    def __init__(self, categories, remap=True):
        self.categories = categories
        self.remap = remap

    def __call__(self, image, anno):
        anno = [obj for obj in anno if obj["category_id"] in self.categories]
        if not self.remap:
            return image, anno
        anno = copy.deepcopy(anno)
        for obj in anno:
            obj["category_id"] = self.categories.index(obj["category_id"])
        return image, anno


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    def __call__(self, image, anno):
        w, h = image.size
        segmentations = [obj["segmentation"] for obj in anno]
        cats = [obj["category_id"] for obj in anno]
        if segmentations:
            masks = convert_coco_poly_to_mask(segmentations, h, w)
            cats = torch.as_tensor(cats, dtype=masks.dtype)
            # merge all instance masks into a single segmentation map
            # with its corresponding categories
            target, _ = (masks * cats[:, None, None]).max(dim=0)
            # discard overlapping instances
            target[masks.sum(0) > 1] = 255
        else:
            target = torch.zeros((h, w), dtype=torch.uint8)
        target = Image.fromarray(target.numpy())
        return image, target


def _coco_remove_images_without_annotations(dataset, cat_list=None):
    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        # if more than 1k pixels occupied in the image
        return sum(obj["area"] for obj in anno) > 1000

    assert isinstance(dataset, torchvision.datasets.CocoDetection)
    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        if cat_list:
            anno = [obj for obj in anno if obj["category_id"] in cat_list]
        if _has_valid_annotation(anno):
            ids.append(ds_idx)

    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs


def collate_fn(batch):
    images, targets = list(zip(*batch))
    batched_imgs = cat_list(images, fill_value=0)
    batched_targets = cat_list(targets, fill_value=255)
    return batched_imgs, batched_targets


def get_transform(data_type):
    base_size = 520
    crop_size = 480

    # min_size = int((0.5 if train else 1.0) * base_size)
    # max_size = int((2.0 if train else 1.0) * base_size)
    transforms = []
    # transforms.append(T.RandomResize(min_size, max_size))
    # if data_type == 'train':
    #     transforms.append(T.RandomHorizontalFlip(0.5))
    #     transforms.append(T.RandomCrop(crop_size))
    transforms.append(T.RandomResize(base_size, base_size))
    transforms.append(T.ToTensor())
    transforms.append(T.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225]))

    return T.Compose(transforms)


def get_coco(data_dir, data_ann_path, data_type, transforms):
    # PATHS = {
    #     "train": ("train2017", os.path.join("annotations", "instances_train2017.json")),
    #     "val": ("val2017", os.path.join("annotations", "instances_val2017.json")),
    #     # "train": ("val2017", os.path.join("annotations", "instances_val2017.json"))
    # }

    CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4,
                1, 64, 20, 63, 7, 72]

    transforms = T.Compose([
        FilterAndRemapCocoCategories(CAT_LIST, remap=True),
        ConvertCocoPolysToMask(),
        transforms
    ])

    # img_folder, ann_file = PATHS[image_set]
    # img_folder = os.path.join(root, img_folder)
    # ann_file = os.path.join(root, ann_file)

    dataset = torchvision.datasets.CocoDetection(root=data_dir,
                                                 annFile=data_ann_path,
                                                 transforms=transforms)

    if data_type == "train":
        dataset = _coco_remove_images_without_annotations(dataset, CAT_LIST)

    return dataset


def load(data_dir, data_ann_path, data_type):
    data = get_coco(data_dir,
                    data_ann_path,
                    data_type,
                    get_transform(data_type))

    data_loader = DataLoader(dataset=data,
                             collate_fn=collate_fn,
                             batch_size=16,
                             num_workers=4,
                             shuffle=False)

    return data_loader
