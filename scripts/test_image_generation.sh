#!/bin/bash
export PYTHONPATH=/nfs3-p1/hjc/projects/CE/codes
export CUDA_VISIBLE_DEVICES=1
export result_dir='/nfs3-p1/hjc/projects/CE/outputs'
export exp_name='gen'
#export model_name='DDPM'
#export model_name='StyleGAN3'
export model_name='NVAE'
#export model_name='TransGAN'
#export model_name='ImprovedDDPM'
#export model_name='DCGAN'
#export model_path='/nfs3-p1/hjc/pretraind_models/checkpoints/ddpm_cifar10.pt'
#export model_path='/nfs3-p1/hjc/pretraind_models/checkpoints/stylegan2-cifar10-32x32.pkl'
export model_path='/nfs3-p1/hjc/pretraind_models/checkpoints/nvae_cifar10_checkpoint.pt'
#export model_path='/nfs3-p1/hjc/pretraind_models/checkpoints/transgan_cifar10.pth'
#export model_path='/nfs3-p1/hjc/pretraind_models/checkpoints/iddpm_cifar10_uncond_50M_500K.pt'
#export model_path='/nfs3-p1/hjc/pretraind_models/checkpoints/dcgan_cifar10_netG_epoch_199.pth'
export data_name='CIFAR10'
export data_dir='/nfs3-p1/hjc/datasets/cifar10/train'
export save_dir=${result_dir}'/'${exp_name}
python engines/test_image_generation.py \
  --model_name ${model_name} \
  --data_name ${data_name} \
  --model_path ${model_path} \
  --data_dir ${data_dir} \
  --save_dir ${save_dir}

#export PYTHONPATH=/nfs3-p1/hjc/projects/CE/codes
#export CUDA_VISIBLE_DEVICES=0
#export result_dir='/nfs3-p1/hjc/projects/CE/outputs'
#export exp_name='gen'
#export data_name='CIFAR10'
#export data_dir='/nfs3-p1/hjc/datasets/cifar10/train'
#export save_dir=${result_dir}'/'${exp_name}
#
#model_names=(
#  #  'DCGAN'
#  #  'StyleGAN3'
##  'NVAE'
##  'TransGAN'
#  'DDPM'
#  'ImprovedDDPM'
#)
#model_paths=(
##  '/nfs3-p1/hjc/pretraind_models/checkpoints/dcgan_cifar10_netG_epoch_199.pth'
##  '/nfs3-p1/hjc/pretraind_models/checkpoints/stylegan2-cifar10-32x32.pkl'
##  '/nfs3-p1/hjc/pretraind_models/checkpoints/nvae_cifar10_checkpoint.pt'
##  '/nfs3-p1/hjc/pretraind_models/checkpoints/transgan_cifar10.pth'
#  '/nfs3-p1/hjc/pretraind_models/checkpoints/ddpm_cifar10.pt'
#  '/nfs3-p1/hjc/pretraind_models/checkpoints/iddpm_cifar10_uncond_50M_500K.pt'
#)
#for i in {0..4}; do
#  python engines/test_image_generation.py \
#    --model_name ${model_names[$i]} \
#    --data_name ${data_name} \
#    --model_path ${model_paths[$i]} \
#    --data_dir ${data_dir} \
#    --save_dir ${save_dir}
#done
