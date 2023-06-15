from torchvision import models
from models import image_generation, object_detection, human_pose_estimation


def load_model(model_name, num_classes=None):
    # assert name in []
    print('-' * 50)
    print('LOAD MODEL:', model_name)
    print('-' * 50)

    ##################################################
    # IMAGE CLASSIFICATION
    ##################################################
    if model_name == 'VGG16BN':
        return models.vgg16_bn(num_classes=num_classes)
    if model_name == 'ResNet18':
        return models.resnet18(num_classes=num_classes)
    if model_name == 'ResNet34':
        return models.resnet34(num_classes=num_classes)
    if model_name == 'ResNet50':
        return models.resnet50(num_classes=num_classes)
    if model_name == 'ResNet101':
        return models.resnet101(num_classes=num_classes)
    if model_name == 'ResNet152':
        return models.resnet152(num_classes=num_classes)
    if model_name == 'InceptionV3':
        return models.inception_v3(num_classes=num_classes)
    if model_name == 'EfficientNetB7':
        return models.efficientnet_b7(num_classes=num_classes)
    if model_name == 'MobileNetV3':
        return models.mobilenet_v3_large(num_classes=num_classes)
    if model_name == 'DenseNet121':
        return models.densenet121(num_classes=num_classes)
    if model_name == 'DenseNet161':
        return models.densenet161(num_classes=num_classes)
    if model_name == 'DenseNet169':
        return models.densenet169(num_classes=num_classes)
    if model_name == 'DenseNet201':
        return models.densenet201(num_classes=num_classes)
    if model_name == 'ViTB16':
        return models.vit_b_16(num_classes=num_classes)
    if model_name == 'ViTB32':
        return models.vit_b_32(num_classes=num_classes)
    if model_name == 'ViTL16':
        return models.vit_l_16(num_classes=num_classes)
    if model_name == 'ViTL32':
        return models.vit_l_32(num_classes=num_classes)
    if model_name == 'ViTH14':
        return models.vit_h_14(num_classes=num_classes)
    if model_name == 'SwinB':
        return models.swin_b(num_classes=num_classes)
    if model_name == 'SwinT':
        return models.swin_t(num_classes=num_classes)
    if model_name == 'SwinS':
        return models.swin_s(num_classes=num_classes)
    if model_name == 'SwinV2B':
        return models.swin_v2_b(num_classes=num_classes)

    ##################################################
    # SEMANTIC SEGMENTATION
    ##################################################
    if model_name == 'FCN_ResNet50':
        return models.segmentation.fcn_resnet50(num_classes=num_classes, aux_loss=True, weights_backbone=None)
    if model_name == 'DeepLabV3_ResNet50':
        return models.segmentation.deeplabv3_resnet50(num_classes=num_classes, aux_loss=True, weights_backbone=None)
    if model_name == 'LRASPP_MobileNetV3':
        return models.segmentation.lraspp_mobilenet_v3_large(num_classes=num_classes, weights_backbone=None)

    ##################################################
    # HUMAN POSE ESTIMATION
    ##################################################
    # KeypointRCNN: https://pytorch.org/vision/stable/_modules/torchvision/models/detection/keypoint_rcnn.html#keypointrcnn_resnet50_fpn
    if model_name == 'KeypointRCNN':
        return models.detection.keypointrcnn_resnet50_fpn(num_classes=num_classes, weights_backbone=None)
    # HRNet: https://github.com/HRNet/HRNet-Human-Pose-Estimation
    if model_name == 'HRNet':
        return human_pose_estimation.load_hrnet()
    # HigherHRNet: https://github.com/HRNet/HigherHRNet-Human-Pose-Estimation
    # LitePose: https://github.com/mit-han-lab/litepose
    # InsPose: https://github.com/hikvision-research/opera/tree/main/configs/inspose
    # ViTPose: https://github.com/ViTAE-Transformer/ViTPose
    # PETR: https://github.com/hikvision-research/opera/tree/main/configs/petr

    ##################################################
    # IMAGE GENERATION
    ##################################################
    # DDPM: https://github.com/w86763777/pytorch-ddpm
    if model_name == 'DDPM':
        return image_generation.load_ddpm()
    # NVAE: https://github.com/NVlabs/NVAE
    if model_name == 'NVAE':
        return image_generation.load_nvae()
    # TransGAN: https://github.com/asarigun/TransGAN
    if model_name == 'TransGAN':
        return image_generation.load_transgan()
    # VDVAE: https://github.com/openai/vdvae
    # LSGM: https://github.com/NVlabs/LSGM (?)
    # ImprovedDDPM： https://github.com/openai/improved-diffusion
    if model_name == 'ImprovedDDPM':
        return image_generation.load_iddpm()
    # StyleGAN: https://github.com/search?q=styleGAN&type=repositories
    # DCGAN: https://github.com/csinva/gan-vae-pretrained-pytorch/tree/master
    if model_name == 'DCGAN':
        return image_generation.load_dcgcn()

    ##################################################
    # OBJECT DETECTION
    ##################################################
    # https://pytorch.org/vision/stable/models.html
    if model_name == 'FasterRCNN_ResNet50_FPN':
        return models.detection.fasterrcnn_resnet50_fpn(num_classes=num_classes, weights_backbone=None)
    if model_name == 'FCOS_ResNet50_FPN':
        return models.detection.fcos_resnet50_fpn(num_classes=num_classes, weights_backbone=None)
    if model_name == 'RetinaNet_ResNet50_FPN':
        return models.detection.retinanet_resnet50_fpn(num_classes=num_classes, weights_backbone=None)
    if model_name == 'SSD300_VGG16':
        return models.detection.ssd300_vgg16(num_classes=num_classes, weights_backbone=None)
    if model_name == 'SSDLite320_MobileNetV3':
        return models.detection.ssdlite320_mobilenet_v3_large(num_classes=num_classes, weights_backbone=None)
    # DETR: https://github.com/facebookresearch/detr
    if model_name == 'DETR_DC5_ResNet50':
        # return torch.hub.load('facebookresearch/detr:main', 'detr_dc5_resnet50', pretrained=False)
        return object_detection.load_detr()


if __name__ == '__main__':
    # print(models.list_models())
    # print(models.vgg16_bn(num_classes=100))
    # print('#' * 50)
    # weights = models.ResNet50_Weights.DEFAULT
    # print(models.ResNet50_Weights.DEFAULT.transforms())
    # print(models.VGG16_BN_Weights.DEFAULT.transforms())
    print(load_model('HRNet'))
    # print(load_model('ResNet50'))
