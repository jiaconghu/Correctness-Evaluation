#!/bin/bash
#export PYTHONPATH=/nfs3-p1/hjc/projects/CE/codes
#export CUDA_VISIBLE_DEVICES=0
#export result_path='/nfs3-p1/hjc/projects/CE/outputs'
#export exp_name='dec'
#export model_name='FasterRCNN_ResNet50_FPN'
##export model_name='FCOS_ResNet50_FPN'
##export model_name='RetinaNet_ResNet50_FPN'
##export model_name='DETR_DC5_ResNet50'
#export data_name='COCO'
#export num_classes=91
#export model_path='/nfs3-p1/hjc/pretraind_models/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'
##export model_path='/nfs3-p1/hjc/pretraind_models/checkpoints/fcos_resnet50_fpn_coco-99b0c9b7.pth'
##export model_path='/nfs3-p1/hjc/pretraind_models/checkpoints/retinanet_resnet50_fpn_coco-eeacb38b.pth'
##export model_path='/nfs3-p1/hjc/pretraind_models/checkpoints/detr-r50-dc5-f0fb7ef5.pth'
#export data_dir='/datasets/COCO2017/images/val'
#export data_ann_path='/datasets/COCO2017/annotations/instances_val2017.json'
#python engines/test_object_detection.py \
#  --model_name ${model_name} \
#  --data_name ${data_name} \
#  --num_classes ${num_classes} \
#  --model_path ${model_path} \
#  --data_dir ${data_dir} \
#  --data_ann_path ${data_ann_path}

export PYTHONPATH=/nfs3-p1/hjc/projects/CE/codes
export CUDA_VISIBLE_DEVICES=0
result_dir='/nfs3-p1/hjc/projects/CE/outputs'
exp_name='dec'
data_name='COCO'
num_classes=91
data_dir='/datasets/COCO2017/images/val'
data_ann_path='/datasets/COCO2017/annotations/instances_val2017.json'
save_dir=${result_dir}'/'${exp_name}

model_names=(
  'FasterRCNN_ResNet50_FPN'
  'FCOS_ResNet50_FPN'
  'RetinaNet_ResNet50_FPN'
  'SSD300_VGG16'
  'SSDLite320_MobileNetV3'
  'DETR_DC5_ResNet50'
)
model_paths=(
  '/nfs3-p1/hjc/pretraind_models/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'
  '/nfs3-p1/hjc/pretraind_models/checkpoints/fcos_resnet50_fpn_coco-99b0c9b7.pth'
  '/nfs3-p1/hjc/pretraind_models/checkpoints/retinanet_resnet50_fpn_coco-eeacb38b.pth'
  '/nfs3-p1/hjc/pretraind_models/checkpoints/ssd300_vgg16_coco-b556d3b4.pth'
  '/nfs3-p1/hjc/pretraind_models/checkpoints/ssdlite320_mobilenet_v3_large_coco-a79551df.pth'
  '/nfs3-p1/hjc/pretraind_models/checkpoints/detr-r50-dc5-f0fb7ef5.pth'
)

for i in {0..5}; do
  echo ${model_names[$i]}
  python engines/test_object_detection.py \
    --model_name ${model_names[$i]} \
    --data_name ${data_name} \
    --num_classes ${num_classes} \
    --model_path ${model_paths[$i]} \
    --data_dir ${data_dir} \
    --data_ann_path ${data_ann_path} \
    --save_dir ${save_dir}
done
