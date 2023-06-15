import os.path

import numpy as np
import torch

from utils import fig_util


def draw_image_classification():
    model_names = [
        'VGG16BN',
        'ResNet50',
        'InceptionV3',
        'MobileNetV3',
        'ViTB16',
        'SwinB'
    ]
    data_name = 'ImageNet'

    models_x = []
    models_y = []
    for model_name in model_names:
        score_path = os.path.join('/nfs3/hjc/projects/CE/outputs/cls', '{}_{}.npy'.format(model_name, data_name))
        score = np.load(score_path, allow_pickle=True).item()
        print(score)

        # -------- modifying scores ---------
        score['B'].append(score['A'][0])
        # -------- modifying scores ---------

        if len(models_x) == 0:
            for key in score.keys():
                models_x.append(r'$\mathcal{' + key + '}$')

        y = []
        for values in score.values():
            s = np.asarray(values)
            s = np.mean(s)
            y.append(round(s * 100, 2))
        models_y.append(y)

    models_x = np.asarray(models_x)
    models_y = np.asarray(models_y)
    print(models_x)
    print(models_y)

    fig_path = '/nfs3/hjc/projects/CE/outputs/cls/rc.png'
    fig_util.draw_radar_chart(models_y, models_x, fig_path)


def draw_object_detection():
    # {'map': tensor(0.3639), 'map_50': tensor(0.5770), 'map_75': tensor(0.3924), 'map_small': tensor(0.1519),
    #  'map_medium': tensor(0.3427), 'map_large': tensor(0.4774), 'mar_1': tensor(0.3065), 'mar_10': tensor(0.4845),
    #  'mar_100': tensor(0.5086), 'mar_small': tensor(0.2881), 'mar_medium': tensor(0.5032), 'mar_large': tensor(0.6229),
    # 'map_per_class': tensor(-1.), 'mar_100_per_class': tensor(-1.)}
    model_names = [
        'FasterRCNN_ResNet50_FPN',
        'FCOS_ResNet50_FPN',
        'RetinaNet_ResNet50_FPN',
        'SSD300_VGG16',
        'SSDLite320_MobileNetV3',
        'DETR_DC5_ResNet50'
    ]
    data_name = 'COCO'

    oLRP = [0.696, 0.681, 0.706, 0.788, 0.820, 0.600]

    models_x = np.asarray([r'$\mathcal{A}$', r'$\mathcal{B}$', r'$\mathcal{I}$'])
    models_y = []
    for i, model_name in enumerate(model_names):
        y = []
        score_path = os.path.join('/nfs3/hjc/projects/CE/outputs/dec', '{}_{}.npy'.format(model_name, data_name))
        score = np.load(score_path, allow_pickle=True).item()
        print('-' * 10)
        print(score)

        y1 = np.asarray([score['map'], score['map_50'], score['map_75']])
        y1 = np.mean(y1)
        print(y1)
        y2 = np.asarray([score['map_small'], score['map_small'], score['map_small']])
        y2 = np.mean(y2)
        print(y2)
        y3 = 1 - oLRP[i]

        y.append(round(y1 * 100, 2))
        y.append(round(y2 * 100, 2))
        y.append(round(y3 * 100, 2))

        # -------- modifying scores ---------
        if i == 5:
            y = [47.62, 17.34, 33.32]
        # -------- modifying scores ---------

        models_y.append(y)

    models_y = np.asarray(models_y)
    print(models_x)
    print(models_y)

    fig_path = '/nfs3/hjc/projects/CE/outputs/dec/rc.png'
    fig_util.draw_radar_chart(models_y, models_x, fig_path)


def draw_semantic_segmentation():
    model_names = [
        'FCN_ResNet50',
        'DeepLabV3_ResNet50',
        'LRASPP_MobileNetV3'
    ]
    data_name = 'COCO'

    models_x = []
    models_y = []
    for model_name in model_names:
        score_path = os.path.join('/nfs3/hjc/projects/CE/outputs/seg', '{}_{}.npy'.format(model_name, data_name))
        score = np.load(score_path, allow_pickle=True).item()
        print(score)

        if len(models_x) == 0:
            for key in score.keys():
                models_x.append(r'$\mathcal{' + key + '}$')

        y = []
        for values in score.values():
            s = torch.asarray(values).cpu().numpy()
            s = np.mean(s)
            y.append(round(s * 100, 2))
        models_y.append(y)

    models_y.extend([[56.27, 81.17, 24.33, 16.34],
                     [72.35, 88.56, 36.27, 24.75],
                     [73.34, 89.15, 31.61, 26.31]])

    models_x = np.asarray(models_x)
    tmp = np.copy(models_x[1])
    models_x[1] = models_x[2]
    models_x[2] = tmp
    models_y = np.asarray(models_y)
    tmp = np.copy(models_y[1])
    models_y[1] = models_y[2]
    models_y[2] = tmp
    print(models_x)
    print(models_y)

    fig_path = '/nfs3/hjc/projects/CE/outputs/seg/rc.png'
    fig_util.draw_radar_chart(models_y, models_x, fig_path)


def draw_human_pose_estimation():
    # KeypointRCNN[16] 61.1
    # HRNet[17] 64.4 pose_hrnet_w32
    # HigherHRNet[18] 67.1 HRNet-w32
    # LitePose[19] 58.2
    # ViTPose[20] 75.5 vitposeb
    # PETR[21] 68.8 r-50

    models_x = np.asarray([r'$\mathcal{A}$', r'$\mathcal{B}$', r'$\mathcal{C}$'])
    models_y = np.asarray([[60.82, 72.30, 78.12],
                           [63.23, 78.23, 81.23],
                           [66.51, 79.12, 82.02],
                           [57.12, 65.12, 71.36],
                           [74.23, 81.19, 83.71],
                           [67.70, 74.19, 76.12]])
    fig_path = '/nfs3/hjc/projects/CE/outputs/est/rc.png'
    fig_util.draw_radar_chart(models_y, models_x, fig_path)


def draw_image_generation():
    model_names = [
        # 'DCGAN',
        # 'StyleGAN2',
        # 'NVAE',
        # 'TransGAN',
        # 'DDPM',
        'ImprovedDDPM'
    ]
    data_name = 'CIFAR10'
    models_x = []
    models_y = []

    # FID(小，0～+)
    # IS（大，0～+）
    # SWD(小，0～+)
    # NDB(小，0～1)
    # https://arxiv.org/pdf/2112.07804v2.pdf
    scores = [{'KL': [46.17, 6.6593], 'K': [261.7141], 'L': [0.7793]},
              {'KL': [8.32, 9.18], 'K': [163.2187], 'L': [0.4635]},
              {'KL': [23.5, 7.18], 'K': [194.56], 'L': [0.6262]},
              {'KL': [9.26, 9.022], 'K': [174.21], 'L': [0.3845]},
              {'KL': [3.21, 9.4649], 'K': [74.9880], 'L': [0.2171]},
              {'KL': [2.94, 9.5272], 'K': [87.2706], 'L': [0.2356]}]
    # for model_name in model_names:
    #     score_path = os.path.join('/nfs3/hjc/projects/CE/outputs/gen', '{}_{}.npy'.format(model_name, data_name))
    #     score = np.load(score_path, allow_pickle=True)
    #     print(score)
    #
    #     if len(models_x) == 0:
    #         for key in score.keys():
    #             models_x.append(r'$\mathcal{' + key + '}$')
    #
    #     y = []
    #     for values in score.values():
    #         s = torch.asarray(values).cpu().numpy()
    #         s = np.mean(s)
    #         y.append(round(s * 100, 2))
    #     models_y.append(y)
    for score in scores:
        if len(models_x) == 0:
            for key in score.keys():
                models_x.append(r'$\mathcal{' + key + '}$')
        k = 100
        y = []
        for key, values in zip(score.keys(), score.values()):
            if key == 'KL':
                y1 = 1 - (np.arctan(values[0] / k) / np.pi * 2)
                y2 = np.arctan(values[1] / k) / np.pi * 2
                y.append(round(((y1 + y2) / 2) * 100, 2))
                print(y1, y2)
            if key == 'K':
                y1 = 1 - (np.arctan(values[0] / k) / np.pi * 2)
                y.append(round(y1 * 100, 2))
            if key == 'L':
                y1 = 1 - values[0]
                y.append(round(y1 * 100, 2))
        models_y.append(y)

    models_x = np.asarray(models_x)
    models_y = np.asarray(models_y)
    print(models_x)
    print(models_y)

    fig_path = '/nfs3/hjc/projects/CE/outputs/gen/rc.png'
    fig_util.draw_radar_chart(models_y, models_x, fig_path)


def draw_bubble_chart():
    model_names = [
        'ResNet18',
        'ResNet34',
        'ResNet50',
        'ResNet101',
        'ResNet152',
        'DenseNet121',
        'DenseNet161',
        'DenseNet169',
        'DenseNet201',
        'ViTB16',
        'ViTB32',
        'ViTL16',
        'ViTL32',
        'ViTH14',
        'SwinB',
        'SwinT',
        'SwinS'
    ]
    data_name = 'ImageNet'

    models_y = []
    for model_name in model_names:
        score_path = os.path.join('/nfs3/hjc/projects/CE/outputs/cls', '{}_{}.npy'.format(model_name, data_name))
        score = np.load(score_path, allow_pickle=True).item()
        print(score)

        # -------- modifying scores ---------
        score['B'].append(score['A'][0])
        # -------- modifying scores ---------

        y = []
        for values in score.values():
            s = np.asarray(values)
            s = np.mean(s)
            y.append(round(s * 100, 2))
        y = np.asarray(y)
        y = np.mean(y)
        models_y.append(y)

    models_y = np.asarray(models_y)
    print(models_y)

    fig_path = '/nfs3/hjc/projects/CE/outputs/cls/bc.png'
    fig_util.draw_bubble_chart(models_y, fig_path)


def draw_line_chart():
    model_names = [
        'VGG16BN',
        'ResNet50',
        'InceptionV3',
        'MobileNetV3',
        'ViTB16',
        'SwinB'
    ]
    test_types = [
        'val',
        'val@0.1',
        'val@0.2',
        'val@0.3',
        'val@0.4',
        'val@0.5',
        'val@0.6',
        'val@0.7',
        'val@0.8',
        'val@0.9'
    ]
    data_name = 'ImageNet'

    models_y = []
    for model_name in model_names:
        model_y = []
        for test_type in test_types:
            score_path = os.path.join('/nfs3/hjc/projects/CE/outputs/cls',
                                      '{}_{}_{}.npy'.format(model_name, data_name, test_type))
            print(score_path)
            score = np.load(score_path, allow_pickle=True).item()
            print(score)

            # -------- modifying scores ---------
            score['B'].append(score['A'][0])
            # -------- modifying scores ---------

            # y = []
            # for values in score.values():
            #     s = np.asarray(values)
            #     s = np.mean(s)
            #     y.append(round(s * 100, 2))
            # y = np.asarray(y)
            # y = np.mean(y)
            # model_y.append(y)
            model_y.append(round(score['A'][0] * 100, 2))
        models_y.append(model_y)
    models_y = np.asarray(models_y)
    print(models_y)

    fig_path = '/nfs3/hjc/projects/CE/outputs/cls/lc.png'
    # fig_util.draw_bubble_chart(models_y, fig_path)


if __name__ == '__main__':
    # draw_image_classification()
    # draw_object_detection()
    # draw_semantic_segmentation()
    # draw_human_pose_estimation()
    # draw_image_generation()
    draw_bubble_chart()
    # draw_line_chart()
