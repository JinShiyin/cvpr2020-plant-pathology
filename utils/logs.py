# @Author: yican
# @Date: 2020-06-14 16:19:48
# @Last Modified by:   yican
# @Last Modified time: 2020-06-30 10:11:22
# Standard libraries
import logging
import os
import pdb
import random
from argparse import ArgumentParser
from logging import Logger
from logging.handlers import TimedRotatingFileHandler
import shutil
from tkinter.font import names
import time as ts
# Third party libraries
import cv2
import numpy as np
import pandas as pd
import torch
from dotmap import DotMap
from os.path import dirname
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers.base import LightningLoggerBase
from pytorch_lightning.utilities.distributed import rank_zero_only
IMG_SHAPE = (700, 600, 3)
IMAGE_FOLDER = "data/images"
NPY_FOLDER = "/home/public_data_center/kaggle/plant_pathology_2020/npys"
LOG_FOLDER = "logs"
import yaml


def mkdir(path: str):
    """Create directory.

     Create directory if it is not exist, else do nothing.

     Parameters
     ----------
     path: str
        Path of your directory.

     Examples
     --------
     mkdir("data/raw/train/")
     """
    try:
        if path is None:
            pass
        else:
            os.stat(path)
    except Exception:
        os.makedirs(path)


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
    seed_everything(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--config',
                        type=str,
                        default='config/train_[original]_small_batch.yaml',
                        help='config path')
    return parser


def get_args(config):
    with open(config, encoding='utf-8') as cfg:
        cfg = yaml.load(cfg, Loader=yaml.FullLoader)
        cfg = DotMap(cfg, _dynamic=False)
    return cfg


def retrival_yaml(path):
    return [
        os.path.join(path, filename) for filename in os.listdir(path)
        if filename.endswith('.yaml')
    ][0]


def get_fold_i(checkpoint_path):
    checkpoint_name = os.path.basename(checkpoint_path)
    names = checkpoint_name.split('-')
    fold_i = 0
    for name in names:
        if 'fold' in name:
            fold_i = int(name.split('=')[-1])
            break
    return fold_i


def parse_checkpoint_meta_info(checkpoint_name):
    names = os.path.splitext(checkpoint_name)[0].split('-')
    meta = {}
    for name in names:
        if 'fold' in name:
            fold_i = int(name.split('=')[-1])
            meta['fold'] = fold_i
        if 'val_loss' in name:
            val_loss = float(name.split('=')[-1])
            meta['val_loss'] = val_loss
        if 'val_roc_auc' in name:
            val_roc_auc = float(name.split('=')[-1])
            meta['val_roc_auc'] = val_roc_auc
        if 'other_loss' in name:
            other_loss = float(name.split('=')[-1])
            meta['other_loss'] = other_loss
        if 'other_roc_auc' in name:
            other_roc_auc = float(name.split('=')[-1])
            meta['other_roc_auc'] = other_roc_auc
    return DotMap(meta)


def parse_checkpoint_dir(checkpoint_dir):
    checkpoint_names = os.listdir(checkpoint_dir)
    best_checkpoint_paths = {}
    best_val_roc_auc = {}
    for checkpoint_name in checkpoint_names:
        if not checkpoint_name.startswith(
                'fold') or 'test-real-world' in checkpoint_name:
            continue
        meta = parse_checkpoint_meta_info(checkpoint_name)
        fold = meta.fold
        val_roc_auc = meta.val_roc_auc
        if fold not in best_checkpoint_paths:
            best_checkpoint_paths[fold] = os.path.join(checkpoint_dir,
                                                       checkpoint_name)
        if fold not in best_val_roc_auc:
            best_val_roc_auc[fold] = val_roc_auc
        if best_val_roc_auc[fold] < val_roc_auc:
            best_checkpoint_paths[fold] = os.path.join(checkpoint_dir,
                                                       checkpoint_name)
            best_val_roc_auc[fold] = val_roc_auc
    return DotMap(best_checkpoint_paths)


def parse_checkpoint_path(checkpoint_path):
    checkpoint_name = os.path.basename(checkpoint_path)
    names = checkpoint_name.split('-')
    fold_i = 0
    for name in names:
        if 'fold' in name:
            fold_i = int(name.split('=')[-1])
            break
    return DotMap({fold_i: checkpoint_path})


def parse_checkpoint_paths(checkpoint_input):
    if os.path.isdir(checkpoint_input):
        return parse_checkpoint_dir(checkpoint_input)
    else:
        return parse_checkpoint_path(checkpoint_input)


def init_hparams(config_path=None):
    if config_path is None:
        parser: ArgumentParser = get_parser()
        args = parser.parse_args()
        hparams = get_args(args.config)
        hparams.config = args.config
    else:
        hparams = get_args(config_path)
        hparams.config = config_path
    if len(hparams.gpus) == 1:
        hparams.gpus = [int(hparams.gpus[0])]
    else:
        hparams.gpus = [int(gpu) for gpu in hparams.gpus]
    return hparams


# def init_hparams():
#     parser: ArgumentParser = get_parser()
#     args = parser.parse_args()
#     hparams = get_args(args)
#     if len(hparams.gpus) == 1:
#         hparams.gpus = [int(hparams.gpus[0])]
#     else:
#         hparams.gpus = [int(gpu) for gpu in hparams.gpus]
#     hparams.image_size = [int(size) for size in hparams.image_size]
#     hparams.config = args.config
#     return hparams


def backup_config(config_path, output_path):
    shutil.copy(config_path, output_path)


def init_training_config():
    """create training configuration from the command parameters, 
    or loading from history checkpoints.

    Returns:
        _type_: _description_
    """
    hparams = init_hparams()
    resume_from_checkpoint = hparams.resume_from_checkpoint
    if resume_from_checkpoint is None:
        timestamp = ts.strftime("%Y%m%d%-H%M", ts.localtime())
        exp_name = hparams.exp_name
        hparams.log_dir = os.path.join(hparams.log_dir,
                                       f'{timestamp}-{exp_name}')
        os.makedirs(hparams.log_dir, exist_ok=True)
        backup_config(hparams.config, hparams.log_dir)
    else:
        output = dirname(resume_from_checkpoint) if os.path.isdir(
            resume_from_checkpoint) else dirname(resume_from_checkpoint)
        config = retrival_yaml(output)
        hparams = init_hparams(config)
        hparams.log_dir = output
        hparams.resume_from_checkpoint = parse_checkpoint_paths(
            resume_from_checkpoint)
        print(f'Resume configuration from {hparams.log_dir}')
    # else:
    #     output = dirname(dirname(resume_from_checkpoint))
    #     config = retrival_yaml(output)
    #     hparams = init_hparams(config)
    #     hparams.log_dir = output
    #     hparams.resume_from_checkpoint = resume_from_checkpoint
    #     hparams.fold_i = get_fold_i(resume_from_checkpoint)
    #     print(
    #         f'Fold {hparams.fold_i}, Resume configuration from {hparams.log_dir}.'
    #     )
    checkpoint_dir = os.path.join(hparams.log_dir, 'checkpoints')
    os.makedirs(checkpoint_dir, exist_ok=True)
    hparams.checkpoint_dir = checkpoint_dir
    return hparams


def load_data(logger, frac=1):
    data, test_data = pd.read_csv("data/train.csv"), pd.read_csv(
        "data/sample_submission.csv")
    # Do fast experiment
    if frac < 1:
        logger.info(f"use frac : {frac}")
        data = data.sample(frac=frac).reset_index(drop=True)
        test_data = test_data.sample(frac=frac).reset_index(drop=True)
    return data, test_data


def load_training_data(logger, data_folder, frac=1):
    data, test_data = pd.read_csv(os.path.join(
        data_folder,
        'train_4_3.csv')), pd.read_csv("data/sample_submission.csv")
    # Do fast experiment
    if frac < 1:
        logger.info(f"use frac : {frac}")
        data = data.sample(frac=frac).reset_index(drop=True)
        test_data = test_data.sample(frac=frac).reset_index(drop=True)
    return data, test_data


def load_test_data(logger, data_folder, frac=1):
    data, test_data = pd.read_csv(os.path.join(
        data_folder,
        'test_4_1.csv')), pd.read_csv("data/sample_submission.csv")
    # Do fast experiment
    if frac < 1:
        logger.info(f"use frac : {frac}")
        data = data.sample(frac=frac).reset_index(drop=True)
        test_data = test_data.sample(frac=frac).reset_index(drop=True)
    return data, test_data


def load_test_data_with_header(logger, data_folder, header_names, frac=1):
    data, test_data = pd.read_csv(
        os.path.join(data_folder, 'test_4_1.csv'),
        names=header_names), pd.read_csv("data/sample_submission.csv")
    # Do fast experiment
    # if frac < 1:
    # logger.info(f"use frac : {frac}")
    # data = data.sample(frac=frac).reset_index(drop=True)
    # test_data = test_data.sample(frac=frac).reset_index(drop=True)
    return data, test_data


def init_logger(log_name, log_dir=None):
    """日志模块
    Reference: https://juejin.im/post/5bc2bd3a5188255c94465d31
    日志器初始化
    日志模块功能:
        1. 日志同时打印到到屏幕和文件
        2. 默认保留近一周的日志文件
    日志等级:
        NOTSET（0）、DEBUG（10）、INFO（20）、WARNING（30）、ERROR（40）、CRITICAL（50）
    如果设定等级为10, 则只会打印10以上的信息

    Parameters
    ----------
    log_name : str
        日志文件名
    log_dir : str
        日志保存的目录

    Returns
    -------
    RootLogger
        Python日志实例
    """

    mkdir(log_dir)

    # 若多处定义Logger，根据log_name确保日志器的唯一性
    if log_name not in Logger.manager.loggerDict:
        logging.root.handlers.clear()
        logger = logging.getLogger(log_name)
        logger.setLevel(logging.DEBUG)

        # 定义日志信息格式
        datefmt = "%Y-%m-%d %H:%M:%S"
        format_str = "[%(asctime)s] %(filename)s[%(lineno)4s] : %(levelname)s  %(message)s"
        formatter = logging.Formatter(format_str, datefmt)

        # 日志等级INFO以上输出到屏幕
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_dir is not None:
            # 日志等级INFO以上输出到{log_name}.log文件
            file_info_handler = TimedRotatingFileHandler(filename=os.path.join(
                log_dir, "%s.log" % log_name),
                                                         when="D",
                                                         backupCount=7)
            file_info_handler.setFormatter(formatter)
            file_info_handler.setLevel(logging.INFO)
            logger.addHandler(file_info_handler)

    logger = logging.getLogger(log_name)

    return logger


def read_image(image_path):
    """ 读取图像数据，并转换为RGB格式
        32.2 ms ± 2.34 ms -> self
        48.7 ms ± 2.24 ms -> plt.imread(image_path)
    """
    return cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)

class TerminalLogger(LightningLoggerBase):

    @property
    def name(self):
        return "TerminalLogger"

    @property
    def version(self):
        # Return the experiment version, int or str.
        return "1.0"


    @rank_zero_only
    def log_metrics(self, metrics, step):
        # metrics is a dictionary of metric names and values
        # your code to record metrics goes here
        pass


    @rank_zero_only
    def finalize(self, status):
        # Optional. Any code that needs to be run after training
        # finishes goes here
        pass