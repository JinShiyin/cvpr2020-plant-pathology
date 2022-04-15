# @Author: yican, yelanlan
# @Date: 2020-07-07 14:48:03
# @Last Modified by:   yican
# @Last Modified time: 2020-07-07 14:48:03
# Standard libraries
import os

# Third party libraries
import torch
from scipy.special import softmax
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

# User defined libraries
from dataset import OpticalCandlingDataset, generate_transforms
from train import CoolSystem
from utils import init_hparams, init_logger, load_test_data, seed_reproducer 
# from torchvision.transforms.functional import normalize, resize, to_pil_image
from torchcam.methods import SmoothGradCAMpp
import matplotlib.pyplot as plt
from utils import *
import time



    

if __name__ == "__main__":
    # Init Hyperparameters
    hparams = init_hparams()

    # Make experiment reproducible
    seed_reproducer(hparams.seed)

    timestamp = time.strftime("%Y%m%d-%H%M", time.localtime()) 

    output_dir = os.path.join("test_results", timestamp)

    os.makedirs(output_dir, exist_ok=True)
    # init logger
    logger = init_logger("kun_out", log_dir=hparams.log_dir)

    # Load data
    test_data, data = load_test_data(logger, hparams.data_folder)

    # Generate transforms
    transforms = generate_transforms(hparams.image_size)

    # Instance Model, Trainer and train model
    model = CoolSystem(hparams)

    # [folds * num_aug, N, num_classes]
    submission = []
    # PATH = [
    #     "logs_submit/fold=0-epoch=67-val_loss=0.0992-val_roc_auc=0.9951.ckpt",
    #     "logs_submit/fold=1-epoch=61-val_loss=0.1347-val_roc_auc=0.9928.ckpt",
    #     "logs_submit/fold=2-epoch=57-val_loss=0.1289-val_roc_auc=0.9968.ckpt",
    #     "logs_submit/fold=3-epoch=48-val_loss=0.1161-val_roc_auc=0.9980.ckpt",
    #     "logs_submit/fold=4-epoch=67-val_loss=0.1012-val_roc_auc=0.9979.ckpt"
    # ]
    PATH = [
        "logs_submit/20220305-0932/fold=0-epoch=59-val_loss=0.1946-val_roc_auc=0.9945.ckpt"
        # "logs_submit/20220305-0932/fold=1-epoch=39-val_loss=0.2358-val_roc_auc=0.9913.ckpt",
        # "logs_submit/20220305-0932/fold=2-epoch=49-val_loss=0.2395-val_roc_auc=0.9913.ckpt",
        # "logs_submit/20220305-0932/fold=3-epoch=48-val_loss=0.2291-val_roc_auc=0.9918.ckpt",
        # "logs_submit/20220305-0932/fold=4-epoch=59-val_loss=0.2246-val_roc_auc=0.9926.ckpt",
        ]

    # ==============================================================================================================
    # Test Submit
    # ==============================================================================================================
    test_dataset = OpticalCandlingDataset(
        hparams.data_folder, test_data, transforms=transforms["val_transforms"], soft_labels_filename=hparams.soft_labels_filename
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=hparams.val_batch_size, shuffle=False, num_workers=hparams.num_workers, pin_memory=True, drop_last=False,
    )

    # gt_data, data = load_test_data_with_header(logger, hparams.data_folder, header_names)
    # gt_labels = gt_data.iloc[:, 1:].to_numpy()

    for path in PATH:
        model.load_state_dict(torch.load(path, map_location="cuda")["state_dict"])
        model.to("cuda")
        model.eval()
        model.zero_grad()

        print(model)
        # cam_extractor = SmoothGradCAMpp(model, target_layer='model.model_ft.4.2.relu')
        cam_extractors = [SmoothGradCAMpp(model, target_layer=f'model.model_ft.{i}.0.downsample') for i in range(1, 5)]
        # cam_extractor = CAM(model, target_layer='model.model_ft.4.2.se_module.fc2')
        b = hparams.val_batch_size
        n = len(cam_extractors)

        for i in range(1):
            test_preds = []
            labels = []
            # with torch.no_grad():
            for batch_id, (images, label, times, filenames) in enumerate(tqdm(test_dataloader)):
                h, w = images.size()[-2:]
                label = label.cuda()
                pred = model(images.cuda()).detach()
                test_preds.append(pred)
                labels.append(label)
                

                # select the false positive indexes
                fn_indexes = select_fn_indexes(pred, label)

                visualization(batch_id, cam_extractors, images, pred, label, filenames,output_dir, save_batch=False, fp_indexes=fn_indexes) 

            labels = torch.cat(labels)
            test_preds = torch.cat(test_preds)
            # [8, N, num_classes]
            submission.append(test_preds.detach().cpu().numpy())

    submission_ensembled = 0
    for sub in submission:
        # sub: N * num_classes
        submission_ensembled += softmax(sub, axis=1) / len(submission)
    test_data.iloc[:, 1:] = submission_ensembled
    test_data.to_csv(os.path.join(output_dir, "submission_distill.csv"), index=False)
