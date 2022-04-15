# @Author: yican, yelanlan
# @Date: 2020-05-27 22:58:45
# @Last Modified by:   yican
# @Last Modified time: 2020-05-27 22:58:45

# Standard libraries
import os
from time import time

# Third party libraries
import cv2
import numpy as np
import pandas as pd
import torch
import albumentations
from albumentations import (
    Compose,
    GaussianBlur,
    HorizontalFlip,
    MedianBlur,
    MotionBlur,
    Normalize,
    OneOf,
    RandomBrightness,
    RandomContrast,
    RandomBrightnessContrast,
    Resize,
    ShiftScaleRotate,
    VerticalFlip,
)
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# User defined libraries
from utils import *

# for fast read data
# from utils import NPY_FOLDER

   

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
        filename = self.data.iloc[index,0]
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
        if self.transforms is not None and isinstance(self.transforms, albumentations.Compose) :
            image = np.array(image)
            image = self.transforms(image=image)["image"].transpose(2, 0, 1)
        elif self.transforms is not None and isinstance(self.transforms,transforms.Compose):
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
    def __init__(self, data_folder, data, soft_labels_filename=None, transforms=None, sample_num = 10):
        super().__init__(data_folder, data, soft_labels_filename, transforms)
        # filename = self.data.iloc[index,0]
        self.data = pd.concat([self.data.loc[
                               self.data['filename'].str.startswith(class_name)].head(sample_num) 
                              for class_name in class_names])

def get_non_trivial_transforms(hparams):
    """use for the baseline model, we don't use any additional data augmentation.

    Args:
        hparams (_type_): _description_

    Returns:
        _type_: _description_
    """
    return Compose([
            Resize(height=hparams.image_size[0], width=hparams.image_size[1]),
            Normalize(mean=hparams.norm['mean'],
                    std= hparams.norm['std'],
                    max_pixel_value=255.0,
                    p=1.0),
        ])
    

def generate_transforms(hparams):

    if hparams.train_transforms == 'non-trivial':
        train_transform = get_non_trivial_transforms(hparams)
    else:
        train_transform = Compose([
            Resize(height=hparams.image_size[0], width=hparams.image_size[1]),
            # OneOf(
                # [RandomBrightness(limit=0.1, p=1),
                #  RandomContrast(limit=0.1, p=1)]),
            RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1),
            OneOf([
                MotionBlur(blur_limit=(3,5)),
                MedianBlur(blur_limit=(3,5)),
                GaussianBlur(blur_limit=(3,5))
            ], p=0.5),
            VerticalFlip(p=0.5),
            HorizontalFlip(p=0.5),
            ShiftScaleRotate(
                shift_limit=0.2,
                scale_limit=0.2,
                rotate_limit=20,
                interpolation=cv2.INTER_LINEAR,
                border_mode=cv2.BORDER_REFLECT_101,
                p=1,
            ),
            Normalize(mean=hparams.norm['mean'],
                    std=hparams.norm['std'],
                    max_pixel_value=255.0,
                    p=1.0),
        ])

    val_transform = get_non_trivial_transforms(hparams)
    # val_transform = Compose([
    #     Resize(height=hparams.image_size[0], width=hparams.image_size[1]),
    #     Normalize(mean=hparams.norm['mean'],
    #               std= hparams.norm['std'],
    #               max_pixel_value=255.0,
    #               p=1.0),
    # ])

    tensor_transform = transforms.Compose([
        transforms.Resize(size=hparams.image_size),
        #    ShiftScaleRotate(
        #     shift_limit=0.2,
        #     scale_limit=0.2,
        #     rotate_limit=20,
        #     interpolation=cv2.INTER_LINEAR,
        #     border_mode=cv2.BORDER_REFLECT_101,
        #     p=1,
        # ),
        transforms.ToTensor()
    ])

    return {
        "train_transforms": train_transform,
        "val_transforms": val_transform,
        'tensor_transforms': tensor_transform
    }


def generate_dataloaders(hparams, train_data, val_data, transforms):
    train_dataset = OpticalCandlingDataset(
        data_folder=hparams.data_folder,
        data=train_data,
        transforms=transforms["train_transforms"],
        soft_labels_filename=hparams.soft_labels_filename)
    val_dataset = OpticalCandlingDataset(
        data_folder=hparams.data_folder,
        data=val_data,
        transforms=transforms["val_transforms"],
        soft_labels_filename=hparams.soft_labels_filename)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=hparams.train_batch_size,
        shuffle=True,
        num_workers=hparams.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=hparams.val_batch_size,
        shuffle=False,
        num_workers=hparams.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    return train_dataloader, val_dataloader


def generate_test_dataloaders(hparams, test_data, transforms):
    test_dataset = OpticalCandlingDataset(
        data_folder=hparams.data_folder,
        data=test_data,
        transforms=transforms["val_transforms"],
        soft_labels_filename=hparams.soft_labels_filename)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=hparams.sample_num,
        shuffle=False,
        num_workers=hparams.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return test_dataloader

def generate_anchor_dataloaders(hparams, test_data, transforms):
    test_dataset = AnchorSet(
        data_folder=hparams.data_folder,
        data=test_data,
        transforms=transforms["val_transforms"],
        sample_num=hparams.sample_num,
        soft_labels_filename=hparams.soft_labels_filename)
    anchor_dataloader = DataLoader(
        test_dataset,
        batch_size=hparams.sample_num,
        shuffle=False,
        num_workers=hparams.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return anchor_dataloader

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


if __name__ == '__main__':
    hparams = init_hparams()
    # Make experiment reproducible
    seed_reproducer(hparams.seed) 
    header_names = ['filename'] + class_names 
    test_data, _= load_test_data_with_header(None, hparams.data_folder, header_names=header_names)
    transforms = generate_transforms(hparams.image_size)
    anchor_dataloader = generate_anchor_dataloaders(hparams, test_data, transforms)
    