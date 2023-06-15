#!/bin/bash
#export PYTHONPATH=/nfs3-p1/hjc/projects/CE/codes
#export CUDA_VISIBLE_DEVICES=1
#export result_dir='/nfs3-p1/hjc/projects/CE/outputs'
#export exp_name='cls'
##export model_name='VGG16BN'
##export model_name='ResNet50'
#export model_name='DenseNet121'
##export model_name='DenseNet161'
##export model_name='MobileNetV3'
##export model_name='ViTB16'
#export data_name='ImageNet'
#export num_classes=1000
##export model_path='/nfs3-p1/hjc/pretraind_models/checkpoints/vgg16_bn-6c64b313.pth'
##export model_path='/nfs3-p1/hjc/pretraind_models/checkpoints/resnet50-11ad3fa6.pth'
#export model_path='/nfs3-p1/hjc/pretraind_models/checkpoints/densenet121-a639ec97.pth'
##export model_path='/nfs3-p1/hjc/pretraind_models/checkpoints/densenet161-8d451a50.pth'
##export model_path='/nfs3-p1/hjc/pretraind_models/checkpoints/mobilenet_v3_large-8738ca79.pth'
##export model_path='/nfs3-p1/hjc/pretraind_models/checkpoints/vit_b_16-c867db91.pth'
#export data_dir='/nfs3-p1/hjc/datasets/imagenet1k/val'
#export save_dir=${result_dir}'/'${exp_name}
#python engines/test_image_classification.py \
#  --model_name ${model_name} \
#  --data_name ${data_name} \
#  --num_classes ${num_classes} \
#  --model_path ${model_path} \
#  --data_dir ${data_dir} \
#  --save_dir ${save_dir}

# ===========================================================

export PYTHONPATH=/nfs3-p1/hjc/projects/CE/codes
export CUDA_VISIBLE_DEVICES=0
result_dir='/nfs3-p1/hjc/projects/CE/outputs'
exp_name='cls'
data_name='ImageNet'
num_classes=1000
data_dir='/nfs3-p1/hjc/datasets/imagenet1k/val'
save_dir=${result_dir}'/'${exp_name}
#model_names=(
#  'VGG16BN'
#  'ResNet50'
#  'InceptionV3'
#  'MobileNetV3'
#  'ViTB16'
#  'SwinB'
#)
#model_paths=(
#  '/nfs3-p1/hjc/pretraind_models/checkpoints/vgg16_bn-6c64b313.pth'
#  '/nfs3-p1/hjc/pretraind_models/checkpoints/resnet50-11ad3fa6.pth'
#  '/nfs3-p1/hjc/pretraind_models/checkpoints/inception_v3_google-0cc3c7bd.pth'
#  '/nfs3-p1/hjc/pretraind_models/checkpoints/mobilenet_v3_large-8738ca79.pth'
#  '/nfs3-p1/hjc/pretraind_models/checkpoints/vit_b_16-c867db91.pth'
#  '/nfs3-p1/hjc/pretraind_models/checkpoints/swin_b-68c6b09e.pth'
#)
model_names=(
#  'ResNet18'
#  'ResNet34'
#  'ResNet50'
#  'ResNet101'
#  'ResNet152'
  'DenseNet121'
  'DenseNet161'
  'DenseNet169'
  'DenseNet201'
#  'ViTB16'
#  'ViTB32'
#  'ViTL16'
#  'ViTL32'
#  'ViTH14'
#  'SwinB'
#  'SwinT'
#  'SwinS'
)
model_paths=(
#  '/nfs3-p1/hjc/pretraind_models/checkpoints/resnet18-f37072fd.pth'
#  '/nfs3-p1/hjc/pretraind_models/checkpoints/resnet34-b627a593.pth'
#  '/nfs3-p1/hjc/pretraind_models/checkpoints/resnet50-0676ba61.pth'
#  '/nfs3-p1/hjc/pretraind_models/checkpoints/resnet101-63fe2227.pth'
#  '/nfs3-p1/hjc/pretraind_models/checkpoints/resnet152-394f9c45.pth'
  '/nfs3-p1/hjc/pretraind_models/checkpoints/densenet121-a639ec97.pth'
  '/nfs3-p1/hjc/pretraind_models/checkpoints/densenet161-8d451a50.pth'
  '/nfs3-p1/hjc/pretraind_models/checkpoints/densenet169-b2777c0a.pth'
  '/nfs3-p1/hjc/pretraind_models/checkpoints/densenet201-c1103571.pth'
#  '/nfs3-p1/hjc/pretraind_models/checkpoints/vit_b_16-c867db91.pth'
#  '/nfs3-p1/hjc/pretraind_models/checkpoints/vit_b_32-d86f8d99.pth'
#  '/nfs3-p1/hjc/pretraind_models/checkpoints/vit_l_16-852ce7e3.pth'
#  '/nfs3-p1/hjc/pretraind_models/checkpoints/vit_l_32-c7638314.pth'
#  '/nfs3-p1/hjc/pretraind_models/checkpoints/vit_h_14_lc_swag-c1eb923e.pth'
#  '/nfs3-p1/hjc/pretraind_models/checkpoints/swin_b-68c6b09e.pth'
#  '/nfs3-p1/hjc/pretraind_models/checkpoints/swin_t-704ceda3.pth'
#  '/nfs3-p1/hjc/pretraind_models/checkpoints/swin_s-5e29d889.pth'
)

for i in {0..3}; do
  echo ${model_names[$i]}
  python engines/test_image_classification.py \
    --model_name ${model_names[$i]} \
    --data_name ${data_name} \
    --num_classes ${num_classes} \
    --model_path ${model_paths[$i]} \
    --data_dir ${data_dir} \
    --save_dir ${save_dir}
done

# ===========================================================

#export PYTHONPATH=/nfs3-p1/hjc/projects/CE/codes
#export CUDA_VISIBLE_DEVICES=0
#result_dir='/nfs3-p1/hjc/projects/CE/outputs'
#exp_name='cls'
#data_name='ImageNet'
#num_classes=1000
#save_dir=${result_dir}'/'${exp_name}
#model_names=(
#  'VGG16BN'
#  'ResNet50'
#  'InceptionV3'
#  'MobileNetV3'
#  'ViTB16'
#  'SwinB'
#)
#model_paths=(
#  '/nfs3-p1/hjc/pretraind_models/checkpoints/vgg16_bn-6c64b313.pth'
#  '/nfs3-p1/hjc/pretraind_models/checkpoints/resnet50-11ad3fa6.pth'
#  '/nfs3-p1/hjc/pretraind_models/checkpoints/inception_v3_google-0cc3c7bd.pth'
#  '/nfs3-p1/hjc/pretraind_models/checkpoints/mobilenet_v3_large-8738ca79.pth'
#  '/nfs3-p1/hjc/pretraind_models/checkpoints/vit_b_16-c867db91.pth'
#  '/nfs3-p1/hjc/pretraind_models/checkpoints/swin_b-68c6b09e.pth'
#)
#data_dirs=(
#  '/nfs3-p1/hjc/datasets/imagenet1k/val@0.1'
#  '/nfs3-p1/hjc/datasets/imagenet1k/val@0.2'
#  '/nfs3-p1/hjc/datasets/imagenet1k/val@0.3'
#  '/nfs3-p1/hjc/datasets/imagenet1k/val@0.4'
#  '/nfs3-p1/hjc/datasets/imagenet1k/val@0.5'
#  '/nfs3-p1/hjc/datasets/imagenet1k/val@0.6'
#  '/nfs3-p1/hjc/datasets/imagenet1k/val@0.7'
#  '/nfs3-p1/hjc/datasets/imagenet1k/val@0.8'
#  '/nfs3-p1/hjc/datasets/imagenet1k/val@0.9'
#  '/nfs3-p1/hjc/datasets/imagenet1k/val'
#)
#
#for i in {0..5}; do
#  for j in {0..9}; do
#    echo ${model_names[$i]}
#    python engines/test_image_classification.py \
#      --model_name ${model_names[$i]} \
#      --data_name ${data_name} \
#      --num_classes ${num_classes} \
#      --model_path ${model_paths[$i]} \
#      --data_dir ${data_dirs[j]} \
#      --save_dir ${save_dir}
#  done
#done
