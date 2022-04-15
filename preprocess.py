import json
import pandas as pd
import os
import numpy as np
import csv
import random
import cv2
import pandas as pd
from dataset import generate_tensor_dataloaders, generate_transforms
from tqdm import tqdm
from utils import *
input_dir = '/data/lxd/datasets/2022-04-15-Eggs'
output_dir = '/data/lxd/datasets/2022-04-15-Eggs'
# output_dir = '/data/lxd/datasets/2022-03-15-EggCandingTest/2022-03-15-PN0.0'
os.makedirs(output_dir, exist_ok=True)
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

class_dirs = os.listdir(input_dir)
one_hot_vector = [0 for _ in range(len(label_map.keys()))]


def write_csv(csv_instance, class_dir, image_ids):
    for image_id in image_ids:
        one_hot = one_hot_vector.copy()
        one_hot[label_map[class_dir]] = 1
        row_item = [image_id] + one_hot
        csv_instance.writerow(row_item)
        # print(f'Processing {image_id}')


def split():
    seed_reproducer(2022)
    with open(os.path.join(output_dir, 'train_4_3.csv'), 'w',
            newline='') as train_file:
        with open(os.path.join(output_dir, 'test_4_1.csv'), 'w',
                newline='') as test_file:
            train_writer = csv.writer(train_file)
            test_writer = csv.writer(test_file)
            for class_dir in class_dirs:
                class_path = os.path.join(input_dir, class_dir)
                if os.path.isdir(class_path):
                    image_ids = [
                        os.path.join(class_dir, image_id)
                        for image_id in os.listdir(class_path) if image_id.endswith('.jpg')
                    ]
                    random.shuffle(image_ids)
                    train_image_ids = image_ids[:len(image_ids) // 4 * 3]
                    test_image_ids = image_ids[len(image_ids) // 4 * 3:]
                    write_csv(train_writer, class_dir, train_image_ids)
                    write_csv(test_writer, class_dir, test_image_ids)
                    print(f'{class_dir} - {len(test_image_ids)}')

def validate():
    train_data, _ = load_training_data(None, output_dir)
    test_data, _ = load_test_data(None, output_dir)

    train_filenames = train_data.iloc[:,0]
    test_filenames = test_data.iloc[:,0]

    error_train_filenames = []
    error_test_filenames = []
    # image = cv2.imread(os.path.join(input_dir, 'OK/000513851_Egg1_(ok)_R_0_cam2 - 副本.bmp'))
    # if image is None:
        # raise Exception()
    for filename in train_filenames:
        try:
            image_path = os.path.join(input_dir, filename)
            print(image_path)
            print(f'Processing {image_path}')
            image = cv2.imread(image_path)
            if image is None:
                raise Exception(f'Error {image_path}')
        except: 
            error_train_filenames.append(image_path)

    for filename in test_filenames:
        try:
            image_path = os.path.join(input_dir, filename)
            print(image_path)
            print(f'Processing {image_path}')
            image = cv2.imread(image_path)
            if image is None:
                raise Exception(f'Error {image_path}')
        except:
            error_test_filenames.append(image_path)
    print(error_train_filenames)
    print(error_test_filenames)
            
def random_positive_negative(input_dir, output_dir):
    pos_ratio = 0.92
    neg_ratio = 0.08
    os.makedirs(output_dir, exist_ok=True)
    # test_data, _ = load_test_data_with_header(None, input_dir, header_names=header_names)
    test_data = pd.read_csv(os.path.join(input_dir, 'test_4_1.csv'))
    pos :pd.DataFrame = test_data.loc[test_data['filename'].str.startswith('OK')]
    # neg :pd.DataFrame = test_data.loc[test_data['filename'].str.startswith('OK') == False]

    print(pos.shape)
    # print(neg.shape)

    pos_num = pos.shape[0]
    neg_num = int(pos_num / pos_ratio * neg_ratio)
    print(f'Sampling positive number {pos_num}, negative number {neg_num}')
    # sample_pos = pos.sample(n=pos_num)
    sample_pos = pos # we take all the positive samples which are occupying a dominant number in test set.
    # sample_neg = neg.sample(n=neg_num)

    num_avg_per_neg_class = neg_num // len(header_names[2:])
    # generate 10 groups of test sets.
    seeds = np.arange(start=2013, stop=2023)
    for idx, seed in enumerate(seeds):
        sample_neg = []
        print(f'Using seed {seed}')
        seed_reproducer(seed)
        for neg_class_name in header_names[2:]:
            sample_neg_per = test_data.loc[test_data['filename'].str.startswith(neg_class_name)].sample(n=num_avg_per_neg_class)
            sample_neg.append(sample_neg_per)
        # print(sample_neg[0])
        sample_neg = pd.concat(sample_neg)
        df = pd.concat([sample_pos, sample_neg])
        output_path = os.path.join(output_dir, f'seed_by_{seed}.csv')
        df.to_csv(output_path, index=False)
        print(f'Processed {seed}, {output_path}')
        
        # print(sample_neg.shape)
        

def cal_overall_mean_and_std():
    seed_reproducer(2022)
    hparams = init_hparams()
    train_data_path = '/data/lxd/datasets/2022-03-02-Eggs/train_4_3.csv'
    test_data_path = '/data/lxd/datasets/2022-03-02-Eggs/test_4_1.csv'
    train_data = pd.read_csv(train_data_path)
    test_data = pd.read_csv(test_data_path)
    data = pd.concat([train_data, test_data])
    # data = data.head(8)
    transforms = generate_transforms(hparams)
    dataloader = generate_tensor_dataloaders(hparams, data, transforms=transforms)
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0
    for batch_id, (images, label, times, filenames) in enumerate(tqdm(dataloader)):
        # Mean over batch, height and width, but not over the channels
        images = images.cuda()
        channels_sum += torch.mean(images, dim=[0,2,3])
        channels_squared_sum += torch.mean(images**2, dim=[0,2,3])
        num_batches += 1
    
    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5
    mean = mean.detach().cpu().numpy().tolist()
    std = std.detach().cpu().numpy().tolist()
    print(mean)
    print(std)
    norm = {'mean': mean, 'std': std}
    with open(os.path.join(hparams.data_folder, 'normalization.json'), 'w') as f:
        json.dump(norm, f, indent=2)
    
if __name__ == '__main__':

    # random_positive_negative(input_dir='/data/lxd/datasets/2022-03-02-Eggs', 
                            #  output_dir='/data/lxd/datasets/2022-03-15-EggCandingTest/2022-03-15-P_[0.92]_N_[0.08]')
    # cal_overall_mean_and_std()
    split()