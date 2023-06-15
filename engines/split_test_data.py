import os
import shutil

data_dir = '/nfs3/hjc/datasets/imagenet1k/val'

total_num = 50
split_ratios = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for split_ratio in split_ratios:
    # result_path = '{}@{}'.format(data_path, split_ratio)
    # print(result_path)
    print('===>', split_ratio)

    for root, dirs, files in os.walk(data_dir):
        if len(files) != 0:
            result_dir = root.replace('val', 'val@{}'.format(split_ratio))

            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            for i, file in enumerate(files):
                if i == total_num * split_ratio:
                    break

                src_path = os.path.join(root, file)
                dst_path = os.path.join(result_dir, file)
                # print(i, dst_path)
                shutil.copyfile(src_path, dst_path)
