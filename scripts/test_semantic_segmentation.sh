#!/bin/bash
#export PYTHONPATH=/nfs3-p1/hjc/projects/CE/codes
#export CUDA_VISIBLE_DEVICES=0
#export result_dir='/nfs3-p1/hjc/projects/CE/outputs'
#export exp_name='seg'
##export model_name='FCN_ResNet50'
##export model_name='DeepLabV3_ResNet50'
#export model_name='LRASPP_MobileNetV3'
#export data_name='COCO'
#export num_classes=21
##export model_path='/nfs3-p1/hjc/pretraind_models/checkpoints/fcn_resnet50_coco-1167a1af.pth'
##export model_path='/nfs3-p1/hjc/pretraind_models/checkpoints/deeplabv3_resnet50_coco-cd0a2569.pth'
#export model_path='/nfs3-p1/hjc/pretraind_models/checkpoints/lraspp_mobilenet_v3_large-d234d4ea.pth'
#export data_dir='/datasets/COCO2017/images/val'
#export data_ann_path='/datasets/COCO2017/annotations/instances_val2017.json'
#export save_dir=${result_dir}'/'${exp_name}
#python engines/test_semantic_segmentation.py \
#  --model_name ${model_name} \
#  --data_name ${data_name} \
#  --num_classes ${num_classes} \
#  --model_path ${model_path} \
#  --data_dir ${data_dir} \
#  --data_ann_path ${data_ann_path} \
#  --save_dir ${save_dir}

export PYTHONPATH=/nfs3-p1/hjc/projects/CE/codes
export CUDA_VISIBLE_DEVICES=0
result_dir='/nfs3-p1/hjc/projects/CE/outputs'
exp_name='seg'
data_name='COCO'
num_classes=21
data_dir='/datasets/COCO2017/images/val'
data_ann_path='/datasets/COCO2017/annotations/instances_val2017.json'
save_dir=${result_dir}'/'${exp_name}

model_names=(
  'FCN_ResNet50'
  'DeepLabV3_ResNet50'
  'LRASPP_MobileNetV3'
)
model_paths=(
  '/nfs3-p1/hjc/pretraind_models/checkpoints/fcn_resnet50_coco-1167a1af.pth'
  '/nfs3-p1/hjc/pretraind_models/checkpoints/deeplabv3_resnet50_coco-cd0a2569.pth'
  '/nfs3-p1/hjc/pretraind_models/checkpoints/lraspp_mobilenet_v3_large-d234d4ea.pth'
)

for i in {0..2}; do
  echo ${model_names[$i]}
  python engines/test_semantic_segmentation.py \
    --model_name ${model_names[$i]} \
    --data_name ${data_name} \
    --num_classes ${num_classes} \
    --model_path ${model_paths[$i]} \
    --data_dir ${data_dir} \
    --data_ann_path ${data_ann_path} \
    --save_dir ${save_dir}
done
