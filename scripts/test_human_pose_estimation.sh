#!/bin/bash
export PYTHONPATH=/nfs3-p1/hjc/projects/CE/codes
export CUDA_VISIBLE_DEVICES=0
export result_path='/nfs3-p1/hjc/projects/CE/outputs'
export exp_name='dec'
#export model_name='KeypointRCNN'
export model_name='HRNet'
export data_name='COCO'
export num_classes=17
#export model_path='/nfs3-p1/hjc/pretraind_models/checkpoints/keypointrcnn_resnet50_fpn_coco-fc266e95.pth'
export model_path='/nfs3-p1/hjc/pretraind_models/checkpoints/pose_hrnet_w32_256x192.pth'
export data_dir='/datasets/COCO2017/images/val'
export data_ann_path='/datasets/COCO2017/annotations/person_keypoints_val2017.json'
python engines/test_human_pose_estimation.py \
  --model_name ${model_name} \
  --data_name ${data_name} \
  --num_classes ${num_classes} \
  --model_path ${model_path} \
  --data_dir ${data_dir} \
  --data_ann_path ${data_ann_path}
