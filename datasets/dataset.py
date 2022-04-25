# @Author: yican, yelanlan
# @Date: 2020-05-27 22:58:45
# @Last Modified by:   yican
# @Last Modified time: 2020-05-27 22:58:45

# Standard libraries
import os
from time import time
import pytorch_lightning as pl
# Third party libraries
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import torch
import albumentations
from albumentations import (Compose, GaussianBlur, HorizontalFlip, MedianBlur,
                            MotionBlur, Normalize, OneOf, RandomBrightness,
                            RandomContrast, RandomBrightnessContrast, Resize,
                            ShiftScaleRotate, VerticalFlip, LongestMaxSize,
                            PadIfNeeded)
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# User defined libraries
from utils import *
from torchvision.utils import save_image
from torch.utils.data.distributed import DistributedSampler

# for fast read data
# from utils import NPY_FOLDER


class MySampler(DistributedSampler):

    def __init__(self,
                 dataset: Dataset,
                 num_replicas: Optional[int] = None,
                 rank: Optional[int] = None,
                 shuffle: bool = True,
                 seed: int = 0,
                 drop_last: bool = False) -> None:
        super().__init__(dataset, num_replicas, rank, shuffle, seed, drop_last)

    def __len__(self) -> int:
        # print(f'Pid {os.getpid()}, {super().__len__()}')
        return super().__len__()


class PlantDataset(Dataset):
    """ Do normal training
    """

    def __init__(self, data, soft_labels_filename=None, transforms=None):
        self.data = data
        self.transforms = transforms
        if soft_labels_filename == "":
            self.soft_labels = None
        else:
            self.soft_labels = pd.read_csv(soft_labels_filename)

    def __getitem__(self, index):
        start_time = time()
        # Read image
        # solution-1: read from raw image
        image = cv2.cvtColor(
            cv2.imread(
                os.path.join(IMAGE_FOLDER, self.data.iloc[index, 0] + ".jpg")),
            cv2.COLOR_BGR2RGB)
        # solution-2: read from npy file which can speed the data load time.
        # image = np.load(os.path.join(NPY_FOLDER, "raw", self.data.iloc[index, 0] + ".npy"))

        # Convert if not the right shape
        if image.shape != IMG_SHAPE:
            image = image.transpose(1, 0, 2)

        # Do data augmentation
        if self.transforms is not None:
            image = self.transforms(image=image)["image"].transpose(2, 0, 1)

        # Soft label
        if self.soft_labels is not None:
            label = torch.FloatTensor(
                (self.data.iloc[index, 1:].values * 0.7).astype(np.float) +
                (self.soft_labels.iloc[index, 1:].values *
                 0.3).astype(np.float))
        else:
            label = torch.FloatTensor(self.data.iloc[index, 1:].values.astype(
                np.int64))

        return image, label, time() - start_time

    def __len__(self):
        return len(self.data)


class OpticalCandlingDataset(Dataset):
    """ Do normal training
    """

    def __init__(self,
                 data_folder,
                 data,
                 soft_labels_filename=None,
                 transforms=None):
        self.data_folder = data_folder
        # self.data = data[-8:]
        self.data = data
        self.transforms = transforms
        if soft_labels_filename == "":
            self.soft_labels = None
        else:
            self.soft_labels = pd.read_csv(soft_labels_filename)

    def __getitem__(self, index):
        start_time = time()
        # Read image
        # solution-1: read from raw image
        filename = self.data.iloc[index, 0]
        path = os.path.join(self.data_folder, filename)
        # image = cv2.cvtColor(cv2.imread(path),cv2.COLOR_BGR2RGB)
        image = Image.open(path).convert("RGB")
        # solution-2: read from npy file which can speed the data load time.
        # image = np.load(os.path.join(NPY_FO21LDER, "raw", self.data.iloc[index, 0] + ".npy"))

        if image is None:
            raise Exception('')
        # Convert if not the right shape
        # if image.shape != IMG_SHAPE:
        #     image = image.transpose(1, 0, 2)
        #     print(image.shape)

        # Do data augmentation
        if self.transforms is not None and isinstance(self.transforms,
                                                      albumentations.Compose):
            image = np.array(image)
            image = self.transforms(image=image)["image"].transpose(2, 0, 1)
        elif self.transforms is not None and isinstance(
                self.transforms, transforms.Compose):
            image = self.transforms(image)

        # Soft label
        if self.soft_labels is not None:
            label = torch.FloatTensor(
                (self.data.iloc[index, 1:].values * 0.7).astype(np.float) +
                (self.soft_labels.iloc[index, 1:].values *
                 0.3).astype(np.float))
        else:
            label = torch.FloatTensor(self.data.iloc[index, 1:].values.astype(
                np.int64))

        return image, label, time() - start_time, filename

    def __len__(self):
        return len(self.data)


class AnchorSet(OpticalCandlingDataset):

    def __init__(self,
                 data_folder,
                 data,
                 soft_labels_filename=None,
                 transforms=None,
                 sample_num=10):
        super().__init__(data_folder, data, soft_labels_filename, transforms)
        # filename = self.data.iloc[index,0]
        self.data = pd.concat([
            self.data.loc[self.data['filename'].str.startswith(
                class_name)].head(sample_num) for class_name in CLASS_NAMES
        ])


def a1_transforms(hparams):
    """use for the baseline model, we don't use any additional data augmentation.

    Args:
        hparams (_type_): _description_

    Returns:
        _type_: _description_
    """
    return Compose([
        Resize(height=hparams.image_size[0], width=hparams.image_size[1]),
        Normalize(mean=hparams.norm['mean'],
                  std=hparams.norm['std'],
                  max_pixel_value=255.0,
                  p=1.0),
    ])


def a2_transforms(hparams):
    return Compose([
        Resize(height=hparams.image_size[0], width=hparams.image_size[1]),
        RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
        OneOf([
            MotionBlur(blur_limit=(3, 5)),
            MedianBlur(blur_limit=(3, 5)),
            GaussianBlur(blur_limit=(3, 5))
        ],
              p=0.5),
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=20,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_REFLECT_101,
            p=0.5,
        ),
        Normalize(mean=hparams.norm['mean'],
                  std=hparams.norm['std'],
                  max_pixel_value=255.0,
                  p=1.0),
    ])

def a3_transforms(hparams):
    return Compose([
        LongestMaxSize(max_size=hparams.image_size[0]),
        RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
        OneOf([
            MotionBlur(blur_limit=(3, 5)),
            MedianBlur(blur_limit=(3, 5)),
            GaussianBlur(blur_limit=(3, 5))
        ],
              p=0.5),
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=20,
            interpolation=cv2.INTER_LINEAR,
            border_mode=cv2.BORDER_CONSTANT,
            p=0.5,
        ),
        Normalize(mean=hparams.norm['mean'],
                  std=hparams.norm['std'],
                  max_pixel_value=255.0,
                  p=1.0),
        PadIfNeeded(min_height=hparams.image_size[0],
                    min_width=hparams.image_size[1],
                    border_mode=cv2.BORDER_CONSTANT)
    ])


def a4_transforms(hparams):
    return Compose([
        LongestMaxSize(max_size=hparams.image_size[0]),
        Normalize(mean=hparams.norm['mean'],
                  std=hparams.norm['std'],
                  max_pixel_value=255.0,
                  p=1.0),
        PadIfNeeded(min_height=hparams.image_size[0],
                    min_width=hparams.image_size[1],
                    border_mode=cv2.BORDER_CONSTANT)
    ])


def generate_transforms(hparams):
    if hparams.train_transforms == 'a1':
        train_transform = a1_transforms(hparams)
        val_transform = a1_transforms(hparams)
    elif hparams.train_transforms == 'a2':
        train_transform = a2_transforms(hparams)
        val_transform = a1_transforms(hparams)
    elif hparams.train_transforms == 'a3':
        train_transform = a3_transforms(hparams)
        val_transform = a4_transforms(hparams)
    elif hparams.train_transforms == 'a4':
        train_transform = a4_transforms(hparams)
        val_transform = a4_transforms(hparams)
    tensor_transform = transforms.Compose(
        [transforms.Resize(size=hparams.image_size),
         transforms.ToTensor()])

    return {
        "train_transforms": train_transform,
        "val_transforms": val_transform,
        'tensor_transforms': tensor_transform
    }


def generate_val_dataloaders(hparams, val_data, transforms):
    dataset = OpticalCandlingDataset(
        data_folder=hparams.data_folder,
        data=val_data,
        transforms=transforms["val_transforms"],
        soft_labels_filename=hparams.soft_labels_filename)

    sampler = MySampler(dataset, shuffle=False, drop_last=True)
    # val_dataloader = DataLoader(
    #     val_dataset,
    #     batch_size=hparams.val_batch_size,
    #     shuffle=False,
    #     num_workers=hparams.num_workers,
    #     pin_memory=True,
    #     drop_last=True,
    # )
    val_dataloader = DataLoader(dataset,
                                batch_size=hparams.val_batch_size,
                                num_workers=hparams.val_num_workers,
                                pin_memory=True,
                                sampler=sampler)
    return val_dataloader


def generate_train_dataloaders(hparams, data, transforms):
    dataset = OpticalCandlingDataset(
        data_folder=hparams.data_folder,
        data=data,
        transforms=transforms["train_transforms"],
        soft_labels_filename=hparams.soft_labels_filename)

    # dataloader = DataLoader(
    #     val_dataset,
    #     batch_size=hparams.train_batch_size,
    #     shuffle=True,
    #     num_workers=hparams.num_workers,
    #     pin_memory=True,
    #     drop_last=True,
    # )
    sampler = MySampler(dataset, shuffle=True, drop_last=True)
    dataloader = DataLoader(dataset,
                            batch_size=hparams.train_batch_size,
                            num_workers=hparams.train_num_workers,
                            pin_memory=True,
                            sampler=sampler)
    return dataloader


def generate_dataloaders(hparams, train_data, val_data, transforms):
    train_dataloader = generate_train_dataloaders(hparams, train_data,
                                                  transforms)
    val_dataloader = generate_val_dataloaders(hparams, val_data, transforms)
    return train_dataloader, val_dataloader


def generate_test_dataloaders(hparams, test_data, transforms):
    dataset = OpticalCandlingDataset(
        data_folder=hparams.data_folder,
        data=test_data,
        transforms=transforms["val_transforms"],
        soft_labels_filename=hparams.soft_labels_filename)
    sampler = MySampler(dataset, shuffle=False, drop_last=True)
    dataloader = DataLoader(dataset,
                            num_workers=hparams.test_num_workers,
                            batch_size=hparams.val_batch_size,
                            pin_memory=True,
                            sampler=sampler)
    # dataloader = DataLoader(
    #     test_dataset,
    #     batch_size=hparams.sample_num,
    #     shuffle=False,
    #     num_workers=hparams.num_workers,
    #     pin_memory=True,
    #     drop_last=True,
    # )
    return dataloader


def generate_anchor_dataloaders(hparams, test_data, transforms):
    dataset = AnchorSet(data_folder=hparams.data_folder,
                        data=test_data,
                        transforms=transforms["val_transforms"],
                        sample_num=hparams.sample_num,
                        soft_labels_filename=hparams.soft_labels_filename)
    sampler = MySampler(dataset, shuffle=False, drop_last=True)
    dataloader = DataLoader(dataset,
                            num_workers=0,
                            batch_size=hparams.sample_num,
                            sampler=sampler)
    # anchor_dataloader = DataLoader(
    #     test_dataset,
    #     batch_size=hparams.sample_num * len(hparams.gpus),
    #     shuffle=False,
    #     num_workers=hparams.num_workers,
    #     pin_memory=True,
    #     drop_last=False,
    # )
    return dataloader


def generate_tensor_dataloaders(hparams, test_data, transforms):
    test_dataset = OpticalCandlingDataset(
        data_folder=hparams.data_folder,
        data=test_data,
        transforms=transforms["tensor_transforms"],
        soft_labels_filename=hparams.soft_labels_filename)
    tensor_dataloader = DataLoader(
        test_dataset,
        batch_size=48,
        shuffle=False,
        num_workers=hparams.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return tensor_dataloader


def test_transform():
    hparams = init_hparams()
    hparams.image_size = [600, 600]
    # hparams.norm.mean = [
    # 0.4755111336708069,
    # 0.15864244103431702,
    # 0.09940344840288162
    # ]
    # hparams.norm.std= [
    # 0.33696553111076355,
    # 0.295562744140625,
    # 0.2568116784095764]
    # test_img = Image.open('/data/lxd/datasets/2022-04-15-Eggs/Weak/082409286_Egg1_(ok)_R_0_cam2.jpg')
    hparams.norm.mean = [0, 0, 0]
    hparams.norm.std = [1, 1, 1]
    test_img = Image.open(
        '/data/lxd/datasets/2022-04-15-Egg-Masks/Flower/egg_roi/082031737_Egg2_(ok)_R_0_cam2.jpg'
    )
    test_img = np.array(test_img)
    test_tf = a3_transforms(hparams)
    test_img = test_tf(image=test_img)["image"].transpose(2, 0, 1)
    test_img = torch.from_numpy(test_img)
    save_image(test_img, 'test.png')


def get_real_world_test_dataloaders(hparams, transforms):
    if 'test_real_world_set' not in hparams or hparams.test_real_world_set is None:
        return []
    test_paths = [
        os.path.join(hparams.data_folder, filename)
        for filename in os.listdir(hparams.data_folder)
        if filename.startswith(hparams.test_real_world_set)
        and filename.endswith('.csv')
    ]
    real_world_test_dataloaders = []
    for filepath in test_paths[:hparams.test_real_world_num]:
        test_data = pd.read_csv(filepath)
        real_world_test_dataloaders.append(
            generate_val_dataloaders(hparams, test_data, transforms))
    return real_world_test_dataloaders


class ProjectDataModule(pl.LightningDataModule):

    def __init__(self, hparams):
        # seed_reproducer(2022)
        super().__init__()
        self.transforms = generate_transforms(hparams)
        self.hparams.update(hparams)
        self.folds = KFold(n_splits=5,
                           shuffle=True,
                           random_state=self.hparams.seed)
        self.data = pd.read_csv(
            os.path.join(self.hparams.data_folder, self.hparams.training_set))
        self.test_data = pd.read_csv(
            os.path.join(self.hparams.data_folder, self.hparams.test_set))
        self.fold_indexes = {}
        for fold_i, (train_index,
                     val_index) in enumerate(self.folds.split(self.data)):
            self.fold_indexes[fold_i] = [train_index, val_index]

    def train_dataloader(self):
        # seed_reproducer(2022)
        train_data = self.data.iloc[
            self.fold_indexes[self.hparams.fold_i][0], :].reset_index(
                drop=True)
        train_dataloader = generate_train_dataloaders(self.hparams, train_data,
                                                      self.transforms)
        self.hparams.HEC_LOGGER.info(
            f'Pid {os.getpid()}, the batches of TRAIN dataloader are {len(train_dataloader)}'
        )
        return train_dataloader

    def val_dataloader(self):
        # seed_reproducer(2022)
        anchor_dataloader = generate_anchor_dataloaders(
            self.hparams, self.test_data, self.transforms)
        real_world_test_dataloaders = get_real_world_test_dataloaders(
            self.hparams, self.transforms)
        val_data = self.data.iloc[
            self.fold_indexes[self.hparams.fold_i][1], :].reset_index(
                drop=True)
        val_dataloader = generate_val_dataloaders(self.hparams, val_data,
                                                  self.transforms)
        val_dataloaders = [anchor_dataloader, val_dataloader
                           ] + real_world_test_dataloaders
        for idx, val_dataloader in enumerate(val_dataloaders):
            self.hparams.HEC_LOGGER.info(
                f'Pid {os.getpid()}, the batches of VAL dataloader {idx} are {len(val_dataloader)}'
            )
        return val_dataloaders

    def test_dataloader(self):
        # seed_reproducer(2022)
        test_dataloaders = [generate_test_dataloaders(self.hparams,
                                                    self.test_data,
                                                    self.transforms)]
        test_real_world_dataloaders = get_real_world_test_dataloaders(self.hparams, self.transforms) 
        test_dataloaders = test_dataloaders + test_real_world_dataloaders
        for test_dataloader in test_dataloaders:
            self.hparams.HEC_LOGGER.info(
                f'Pid {os.getpid()}, the batches of TEST dataloader {len(test_dataloader)}'
            )
        return test_dataloaders


if __name__ == '__main__':
    test_transform()
    # hparams = init_hparams()
    # # Make experiment reproducible
    # seed_reproducer(hparams.seed)
    # header_names = ['filename'] + class_names
    # test_data, _= load_test_data_with_header(None, hparams.data_folder, header_names=header_names)
    # transforms = generate_transforms(hparams.image_size)
    # anchor_dataloader = generate_anchor_dataloaders(hparams, test_data, transforms)
