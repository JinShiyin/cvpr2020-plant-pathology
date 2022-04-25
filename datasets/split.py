import json
import pdb
import pandas as pd
import os
import numpy as np
import csv
import random
import cv2
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm


def seed_reproducer(seed=2020):
    """Reproducer for pytorch experiment.

    Parameters
    ----------
    seed: int, optional (default = 2019)
        Radnom seed.

    Example
    -------
    seed_reproducer(seed=2019).
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)


label_map = {
    'OK': 0,
    'AirRoomShake': 1,
    'Dead': 2,
    'Empty': 3,
    'NoAirRoom': 4,
    'Split': 5,
    'Weak': 6,
    'Flower': 7
}
header_names = [
    'filename', 'OK', 'AirRoomShake', 'Dead', 'Empty', 'NoAirRoom', 'Split',
    'Weak', 'Flower'
]
one_hot_vector = [0 for _ in range(len(label_map.keys()))]


def compose_rows(class_dir, image_ids):
    rows = []
    """the row format in csv: filename, OK, AirRoomShake, Dead, Empty, NoAirRoom, Split, Weak, Flower
                              'Weak/XXXX.jpg', 0, 0, 0, 0, 0, 0, 1, 0

    Returns:
        _type_: _description_
    """
    for image_id in image_ids:
        one_hot = one_hot_vector.copy()
        one_hot[label_map[class_dir]] = 1
        row = [image_id] + one_hot
        rows.append(row)
    return rows


def collect_datas(input_dir):
    """collect training and test datas with 3:1

    Args:
        input_dir (_type_): _description_

    Returns:
        _type_: _description_
    """
    seed_reproducer(2022)
    class_dirs = os.listdir(input_dir)
    train_datas = []
    test_datas = []
    for class_dir in class_dirs:
        class_path = os.path.join(input_dir, class_dir)
        if os.path.isdir(class_path):
            image_ids = [
                os.path.join(class_dir, image_id)
                for image_id in os.listdir(class_path)
                if image_id.endswith('.jpg')
            ]
            random.shuffle(image_ids)
            train_image_ids = image_ids[:len(image_ids) // 4 * 3]
            test_image_ids = image_ids[len(image_ids) // 4 * 3:]
            train_datas.extend(compose_rows(class_dir, train_image_ids))
            test_datas.extend(compose_rows(class_dir, test_image_ids))
    train_df: DataFrame = DataFrame(train_datas, columns=header_names)
    test_df: DataFrame = DataFrame(test_datas, columns=header_names)
    return train_df, test_df


def init_dataset_spliting(input_dir, output_dir):
    train_df, test_df = collect_datas(input_dir)
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)


def init_incremental_dataset_spliting(old_dir, output_dir):
    seed_reproducer(2022)
    if old_dir == output_dir:
        print('The datset does not exist any changes.')
        return
    old_train_df: DataFrame = pd.read_csv(os.path.join(old_dir, 'train.csv'))
    old_test_df: DataFrame = pd.read_csv(os.path.join(old_dir, 'test.csv'))
    old_df = pd.concat([old_train_df, old_test_df])
    train_df, test_df = collect_datas(output_dir)
    # get the set difference
    incremental_df = pd.concat([old_train_df, old_test_df, train_df,
                                test_df]).drop_duplicates(keep=False)
    incremental_train_df = []
    # random incremental training samples from incremental set.
    for class_name in header_names[1:]:
        class_samples = incremental_df.loc[
            incremental_df['filename'].str.startswith(class_name)]
        incremental_train_df.append(
            class_samples.sample(n=len(class_samples) // 4 * 3))
    incremental_train_df = pd.concat(incremental_train_df)
    # the remained samples are for incremental testing.
    incremental_test_df = pd.concat([incremental_train_df, incremental_df
                                     ]).drop_duplicates(keep=False)
    print(
        f'Old samples {len(old_df)}, for training {len(old_train_df)}, for testing {len(old_test_df)}'
    )
    print(
        f'Incremental samples {len(incremental_df)}, for training {len(incremental_train_df)}, for testing {len(incremental_test_df)}'
    )
    train_df = pd.concat([old_train_df, incremental_train_df])
    test_df = pd.concat([old_test_df, incremental_test_df])
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
    total_df = pd.concat([train_df, test_df])
    print(
        f'New samples {len(total_df)}, for traning {len(train_df)}, for testing {len(test_df)}'
    )
    for class_name in header_names[1:]:
        class_samples = total_df.loc[total_df['filename'].str.startswith(
            class_name)]
        print(f'{class_name}: {len(class_samples)}')


def random_positive_negative(input_dir, output_dir):
    pos_ratio = 0.92
    neg_ratio = 0.08
    os.makedirs(output_dir, exist_ok=True)
    # test_data, _ = load_test_data_with_header(None, input_dir, header_names=header_names)
    test_data = pd.read_csv(os.path.join(input_dir, 'test.csv'))
    pos: pd.DataFrame = test_data.loc[test_data['filename'].str.startswith(
        'OK')]
    # neg :pd.DataFrame = test_data.loc[test_data['filename'].str.startswith('OK') == False]

    print(pos.shape)
    # print(neg.shape)

    pos_num = pos.shape[0]
    neg_num = int(pos_num / pos_ratio * neg_ratio)
    print(f'Sampling positive number {pos_num}, negative number {neg_num}')
    # sample_pos = pos.sample(n=pos_num)
    sample_pos = pos  # we take all the positive samples which are occupying a dominant number in test set.
    # sample_neg = neg.sample(n=neg_num)

    num_avg_per_neg_class = neg_num // len(header_names[2:])
    # generate 10 groups of test sets.
    seeds = np.arange(start=2013, stop=2023)
    for idx, seed in enumerate(seeds):
        sample_neg = []
        print(f'Using seed {seed}')
        seed_reproducer(seed)
        for neg_class_name in header_names[2:]:
            sample_neg_per = test_data.loc[
                test_data['filename'].str.startswith(neg_class_name)].sample(
                    n=num_avg_per_neg_class)
            sample_neg.append(sample_neg_per)
        # print(sample_neg[0])
        sample_neg = pd.concat(sample_neg)
        df = pd.concat([sample_pos, sample_neg])
        output_path = os.path.join(output_dir, f'seed_by_{seed}.csv')
        df.to_csv(output_path, index=False)
        print(f'Processed {seed}, {output_path}')

        # print(sample_neg.shape)


def add_intermediate_csv(df: pd.DataFrame,
                         output_dir,
                         output,
                         intermediate='egg_roi'):
    new_df = []
    for index, row in df.iterrows():
        filename = row['filename']
        split_filename = filename.split('/')
        split_filename.insert(1, intermediate)
        filename = ''
        for f in split_filename:
            filename = os.path.join(filename, f)
        if not os.path.exists(os.path.join(output_dir, filename)):
            raise Exception(f'Not found {filename}')
        print(filename)
        df.at[index, 'filename'] = filename
    # pdb.set_trace()
    df.to_csv(output, index=False)


def generate_from_recent_csv(input_dir, output_dir):
    train_csv = os.path.join(input_dir, 'train.csv')
    test_csv = os.path.join(input_dir, 'test.csv')
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)
    add_intermediate_csv(train_df, output_dir,
                         os.path.join(output_dir, 'train.csv'))
    add_intermediate_csv(test_df, output_dir,
                         os.path.join(output_dir, 'test.csv'))


if __name__ == '__main__':

    input_dir = '/data/lxd/datasets/2022-03-02-Eggs'
    output_dir = '/data/lxd/datasets/2022-03-02-Eggs'
    # output_dir = '/data/lxd/datasets/2022-03-15-EggCandingTest/2022-03-15-PN0.0'
    os.makedirs(output_dir, exist_ok=True)
    # random_positive_negative(input_dir='/data/lxd/datasets/2022-03-02-Eggs',
    #  output_dir='/data/lxd/datasets/2022-03-15-EggCandingTest/2022-03-15-P_[0.92]_N_[0.08]')
    # cal_overall_mean_and_std()
    collect_datas(input_dir, output_dir)