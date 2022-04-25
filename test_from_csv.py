# @Author: yican, yelanlan
# @Date: 2020-07-07 14:48:03
# @Last Modified by:   yican
# @Last Modified time: 2020-07-07 14:48:03
# Standard libraries
import shutil
import numpy as np
import numpy as np
import time
import os
from sklearn import metrics
# User defined libraries
import pandas as pd
from utils import *

if __name__ == "__main__":
    # Init Hyperparameters
    # test_result_files= ['submission_distill_five.csv', 'submission_distill_five_no_aug.csv','submission_distill_one.csv','submission_distill_one_no_aug.csv']
    # test_dir = 'test_results'
    test_dir = '/data/lxd/project/cvpr2020-plant-pathology/test_results/20220316-1719-group-testing/avg'
    test_result_files = [
        filename for filename in os.listdir(test_dir)
        if filename.endswith('.csv')
    ]
    # hparams = init_hparams()

    # Make experiment reproducible
    seed_reproducer(hparams.seed)

    timestamp = time.strftime("%Y%m%d-%H%M", time.localtime())

    output_dir = f'outputs/{timestamp}'

    fn_output_dir = os.path.join(output_dir, 'fn')

    # init logger
    # logger = init_logger("kun_out", log_dir=hparams.log_dir)

    os.makedirs(output_dir, exist_ok=True)

    # class_names = ['OK', 'AirRoomShake', 'Dead', 'Empty', 'NoAirRoom', 'Split', 'Weak', 'Flower']

    # bn_class_names = ['OK', 'NoOK']

    # header_names = ['filename'] + class_names
    # gt_data, data = load_test_data_with_header(logger, hparams.data_folder, header_names)
    gt_data = pd.read_csv(os.path.join(hparams.data_folder, f'test_4_1.csv'))
    print(gt_data.head(10))
    for pred_file in test_result_files:
        test_pred_file_path = os.path.join(test_dir, pred_file)
        print(test_pred_file_path)
        pred_data = pd.read_csv(test_pred_file_path)
        print(pred_data.head(10))
        generate_report(pred_data, gt_data, pred_file, output_dir)