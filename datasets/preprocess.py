from merge import *
from segmentation import *
from incremental import *
from transfer import *
from split import *


def generate_roi_dataset(input_dir, output_dir):
    pass


def process_dataset(input_dir, mask_output_dir, new_dir):
    if new_dir != input_dir:
        # merge old data with new data
        merge(input_dir, new_dir)

    # transfer the images format from bmp to jpg for compressing the size of dataset
    transfer_bmp2jpg(new_dir)

    # update training set and test set using incremental strategy
    init_incremental_dataset_spliting(input_dir, new_dir)

    # preprocessing roi datasets
    init_seg(new_dir, mask_output_dir)

    # generate training/test csv for refactored roi datasets
    generate_from_recent_csv(new_dir, mask_output_dir)

    # random_positive_negative(new_dir, new_dir)
    random_positive_negative(mask_output_dir, mask_output_dir)


if __name__ == '__main__':
    # input_dir = '/Users/shandalau/Documents/Datasets/EggCanding/2022-04-15-Eggs'
    # new_dir = '/Users/shandalau/Documents/Datasets/EggCanding/2022-04-18-Eggs'
    # mask_output_dir = '/Users/shandalau/Documents/Datasets/EggCanding/2022-04-18-Egg-Masks'
    input_dir = '/data/lxd/datasets/2022-04-15-Eggs'
    new_dir = '/data/lxd/datasets/2022-04-18-Eggs'
    mask_output_dir = '/data/lxd/datasets/2022-04-18-Egg-Masks'
    process_dataset(input_dir, mask_output_dir, new_dir)
