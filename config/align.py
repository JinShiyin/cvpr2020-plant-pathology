from argparse import ArgumentParser
import yaml
import numpy as np
import shutil
import os


def merge(source_path, target_path):
    """align the configuration from source to targe

    Args:
        source_path (_type_): _description_
        target_path (_type_): _description_
    """
    with open(source_path, 'r') as fp1:
        with open(target_path, 'r') as fp2:
            src = yaml.load(fp1, Loader=yaml.FullLoader)
            target = yaml.load(fp2, Loader=yaml.FullLoader)
            new = src.copy()
            for k, v in target.items():
                if k not in src:
                    new[k] = v

    with open(source_path, 'w') as fp:
        yaml.dump(new, fp, sort_keys=False, default_flow_style=None)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--tf',
                        type=str,
                        default='train_[roi]_small_batch_v2.yaml',
                        help='config path')
    args = parser.parse_args()
    config_filenames = [
        filename for filename in os.listdir('.')
        if filename.endswith('.yaml') and not os.path.isdir(filename)
    ]
    backup_path = 'backup'
    os.makedirs(backup_path, exist_ok=True)
    # target_filename = 'train_[original]_small_batch_v2.yaml'
    target_filename = args.tf
    shutil.copy(target_filename, backup_path)
    for filename in config_filenames:
        if filename == target_filename:
            continue
        print(f'Align {filename} to {target_filename}')
        shutil.copy(filename, backup_path)
        merge(filename, target_filename)