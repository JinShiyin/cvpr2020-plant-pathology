# @Author: yican, yelanlan
# @Date: 2020-07-07 14:48:03
# @Last Modified by:   yican
# @Last Modified time: 2020-07-07 14:48:03
# Standard libraries
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

# Third party libraries
import torch
from scipy.special import softmax
from torch.utils.data import DataLoader
from tqdm import tqdm

# User defined libraries
from dataset import OpticalCandlingDataset, generate_transforms, PlantDataset
from train import CoolSystem
from utils import init_hparams, init_logger, load_test_data, seed_reproducer, load_data


if __name__ == "__main__":
    # Init Hyperparameters
    hparams = init_hparams()

    # Make experiment reproducible
    seed_reproducer(hparams.seed)

    output_dir = "test_results"

    os.makedirs(output_dir, exist_ok=True)
    # init logger
    logger = init_logger("kun_out", log_dir=hparams.log_dir)

    # Load data
    test_data, data = load_test_data(logger, hparams.data_folder)

    # Generate transforms
    transforms = generate_transforms(hparams.image_size)

    early_stop_callback = EarlyStopping(monitor="val_roc_auc", patience=10, mode="max", verbose=True)

    # Instance Model, Trainer and train model
    model = CoolSystem(hparams)

    # [folds * num_aug, N, num_classes]
    submission = []
    PATH = [
        "logs_submit/fold=0-epoch=67-val_loss=0.0992-val_roc_auc=0.9951.ckpt",
        "logs_submit/fold=1-epoch=61-val_loss=0.1347-val_roc_auc=0.9928.ckpt",
        "logs_submit/fold=2-epoch=57-val_loss=0.1289-val_roc_auc=0.9968.ckpt",
        "logs_submit/fold=3-epoch=48-val_loss=0.1161-val_roc_auc=0.9980.ckpt",
        "logs_submit/fold=4-epoch=67-val_loss=0.1012-val_roc_auc=0.9979.ckpt"
    ]

    # ==============================================================================================================
    # Test Submit
    # ==============================================================================================================
    test_dataset = OpticalCandlingDataset(
        hparams.data_folder, test_data, transforms=transforms["train_transforms"], soft_labels_filename=hparams.soft_labels_filename
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=hparams.val_batch_size, shuffle=False, num_workers=hparams.num_workers, pin_memory=True, drop_last=False,
    )

    for path in PATH:
        model.load_state_dict(torch.load(path, map_location="cuda")["state_dict"])
        model.to("cuda")
        model.eval()

        for i in range(8):
            test_preds = []
            labels = []
            with torch.no_grad():
                for image, label, times in tqdm(test_dataloader):
                    test_preds.append(model(image.to("cuda")))
                    labels.append(label)

                labels = torch.cat(labels)
                test_preds = torch.cat(test_preds)
                # [8, N, num_classes]
                submission.append(test_preds.cpu().numpy())

    submission_ensembled = 0
    for sub in submission:
        # sub: N * num_classes
        submission_ensembled += softmax(sub, axis=1) / len(submission)
    test_data.iloc[:, 1:] = submission_ensembled
    test_data.to_csv(os.path.join(output_dir, "submission_distill.csv"), index=False)
