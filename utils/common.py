import pdb
import traceback
from dotmap import DotMap
from matplotlib.pyplot import axis
# from sklearn.metrics import roc_auc_score
from sklearn import metrics
import torch
import sys
import os
# sys.path.append('./')
from typing import Any, List, Optional, Tuple, Union
from matplotlib import cm
import cv2
from torchvision.transforms.functional import to_pil_image
import numpy as np
from torchvision.transforms.functional import to_tensor
from PIL import Image
from torchvision.utils import save_image
import os
from torchcam.methods import SmoothGradCAMpp
import pandas as pd
from .report import *
from .constant import *
from pytorch_lightning.plugins import *
from pytorch_lightning.strategies import *
from sklearn.metrics import roc_auc_score, classification_report

class_label_to_name = {
    0: 'ok',
    1: 'qishihuangdong',
    2: 'sipei',
    3: 'kongliao',
    4: 'wuqishi',
    5: 'liewen',
    6: 'ruopei',
    7: 'huake'
}
name_to_class_label = {
    'ok': 0,
    'qishihuangdong': 1,
    'sipei': 1,
    'kongliao': 1,
    'wuqishi': 1,
    'liewen': 1,
    'ruopei': 1
}


def extract_activation_map(cam_extractor: SmoothGradCAMpp, images, preds):
    activation_map = cam_extractor(preds.argmax(dim=1).tolist(), preds)[0]
    # print(len(activation_map))
    activation_map = torch.nn.functional.interpolate(
        activation_map.unsqueeze(1),
        size=images.size()[2:],
        mode='bicubic',
        align_corners=True)
    # [b, 1, 3, h, w]
    return activation_map.to(images.device)


def generate_heatmaps(activation_maps, colormap='jet'):
    """generate the corresponding heatmaps using activation maps

    Args:
        activation_maps (_type_): [b, n, h, w], b is the batch size, n is the total layers, h and w are the image spatical size
        mask (_type_): _description_
        colormap (str, optional): _description_. Defaults to 'jet'.

    Returns:
        _type_: _description_
    """
    b, n, h, w = activation_maps.size()
    activation_maps = activation_maps.reshape(b * n, h, w)
    heat_maps = []
    for mask in activation_maps:
        cmap = cm.get_cmap(colormap)
        # generate a heat map, [h, w, 3]
        heat_map = torch.from_numpy(
            cmap((mask**2).detach().cpu().numpy())[:, :, :3]).to(mask.device)
        heat_maps.append(heat_map)
        # print(heat_map.size())
        # [b, n, h, w, 3] -> [b, n, 3, h, w]
    return torch.stack(heat_maps).reshape(b, n, h, w,
                                          -1).permute(0, 1, 4, 2, 3)


def overlay(images, heat_maps, alpha=0.5):
    """_summary_

    Args:
        images (_type_): [b, 1, 3, h, w]
        heat_maps (_type_): [b, n, 3, h, w] 
        alpha (float, optional): _description_. Defaults to 0.5.

    Returns:
        _type_: _description_
    """
    return images * alpha + (1 - alpha) * heat_maps


def render_label(image, label, pred):
    image = cv2.putText(image, f'Label: {label}', (50, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                        cv2.LINE_AA)
    image = cv2.putText(image, f'Pred: {pred}', (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2,
                        cv2.LINE_AA)
    # cv2.imshow('Classification Result',image)
    # cv2.waitKey(1)
    return image


def img_denorm(image, mean, std):
    #for ImageNet the mean and std are:
    #mean = np.asarray([ 0.485, 0.456, 0.406 ])
    #std = np.asarray([ 0.229, 0.224, 0.225 ])
    std = torch.tensor(std, device=image.device).reshape(1, -1, 1, 1)
    mean = torch.tensor(mean, device=image.device).reshape(-1, 1, 1)
    # mean = -1 * mean / std
    # std = 1.0 / std
    image = image * std + mean
    return torch.clamp(image, 0, 1)


def visualization(batch_id,
                  cam_extractors: List[SmoothGradCAMpp],
                  images,
                  preds,
                  labels,
                  filenames,
                  output_dir,
                  save_batch=True,
                  save_per_image=False,
                  fp_indexes=None,
                  norm=True,
                  mean=[0.485, 0.456, 0.406],
                  std=[0.229, 0.224, 0.225]):
    """render the convolution's activation in the images.

    Args:
        batch_id (_type_): batch idx
        cam_extractors (_type_): the each feature map extractor
        images (_type_): [b, 3, h, w]
        pred (_type_): [b, num_classes]
        label (_type_): [b, num_classes]
        output_dir (_type_): output directory
        path (_type_): the path of images 
    """
    # output_dir = os.path.join(output_dir, 'batch')
    os.makedirs(output_dir, exist_ok=True)
    b, _, h, w = images.size()
    n = len(cam_extractors)
    activation_maps = torch.cat(
        [extract_activation_map(cam, images, preds) for cam in cam_extractors],
        dim=1)
    heat_maps = generate_heatmaps(activation_maps, 'jet')
    # print(heat_maps.size())

    images = img_denorm(images, mean=mean, std=std)
    images = images.unsqueeze(1)
    mask_images = overlay(images, heat_maps)
    images = render_labels(images, labels, preds)
    results = torch.cat([images, mask_images],
                        dim=1).reshape(b * (n + 1), 3, h, w)
    # print(results.size())
    if save_batch:
        save_image(results,
                   os.path.join(output_dir, f'{batch_id}.jpeg'),
                   nrow=n + 1)
    if save_per_image:
        per_results = results.view(b, n + 1, 3, h, w)
        for filename, image in zip(filenames, per_results):
            prefix = os.path.splitext(filename)[0].replace('/', '_')
            save_image(image,
                       os.path.join(output_dir, f'{prefix}.jpeg'),
                       nrow=n + 1)

    # save false negative by class.
    if fp_indexes is not None:
        # if fn_indexes is None:
        results = results.reshape(b, n + 1, 3, h, w)
        fp_output_dir = os.path.join(output_dir, 'fp')
        os.makedirs(fp_output_dir, exist_ok=True)
        # selected the false positive
        labels = torch.argmax(labels, dim=1).detach().cpu().numpy(
        )  # [b, 1] transfer one-hot into class index
        fp_results = results[fp_indexes]
        if not len(fp_results):
            print(
                f'Batch id {batch_id}: Not found false negative samples in batch.'
            )
            return
        fn_filenames = np.array(filenames)[fp_indexes]
        fp_labels = labels[fp_indexes]
        for label, result, filename in zip(fp_labels, fp_results,
                                           fn_filenames):
            class_dir = os.path.join(fp_output_dir, class_label_to_name[label])
            if not os.path.exists(class_dir):
                os.makedirs(class_dir, exist_ok=True)
            filename = os.path.splitext(filename)[0].replace('/', '_')
            save_image(result,
                       os.path.join(class_dir, f'{filename}.jpeg'),
                       nrow=n + 1)


def render_labels(images, labels, preds):
    """

    Args:
        images (_type_): [b, 1, 3, h, w]
        labels (_type_): _description_
        preds (_type_): _description_

    Returns:
        _type_: _description_
    """
    cpu_images = images.detach().cpu().squeeze()
    # one-hot to class index
    preds = preds.argmax(dim=1).tolist()
    labels = labels.argmax(dim=1).tolist()
    new_images = []
    for image, label, pred in zip(cpu_images, labels, preds):
        image = np.array(to_pil_image(image))
        # class index to class name
        label = class_label_to_name[label]
        pred = class_label_to_name[pred]
        image = render_label(image, label, pred)
        new_images.append(to_tensor(image))
        # [b, 1, 3, h, w]
    return torch.stack(new_images).unsqueeze(1).to(images.device)


def select_fn_indexes(pred, label):
    label_class = torch.argmax(label, dim=1)
    pred_class = torch.argmax(pred, dim=1)
    fn_indexes = torch.ne(pred_class, label_class) & torch.eq(pred_class, 0)
    fn_indexes = fn_indexes.detach().cpu().numpy()
    return fn_indexes


def get_roc_auc(hparams, labels, scores):
    class_num = labels.argmax(dim=1).unique()
    metric_scores = 0
    label_class = labels.detach().cpu().numpy() # [n, num_classes], the second dim saves the one-hot vector.
    pred_class = scores.detach().cpu().numpy() # [n, num_classes]
    try:
        if len(class_num) == 1:
            metric_scores = torch.tensor(0)
        else:
            # print(labels)
            # label_class = np.argmax(labels.detach().cpu().numpy(), axis=1)
            # pred_class = np.argmax(scores.detach().cpu().numpy(), axis=1)
            if hparams.metrics == 'roc_auc': 
                metric_scores = metrics.roc_auc_score(label_class,
                                            pred_class,
                                            multi_class='ovo')
            elif hparams.metrics == 'mAP':
                metric_scores = metrics.average_precision_score(label_class,
                                            pred_class)
            elif hparams.metrics == 'bn_mAP':
                # Convert one-hot to class label.
                bn_labels = np.argmax(label_class, axis=1)
                bn_pred_labels = np.argmax(pred_class, axis=1)
                bn_labels[bn_labels != 0] = 1
                bn_pred_labels[bn_pred_labels != 0] = 1
                metric_scores = metrics.average_precision_score(bn_labels,
                                            bn_pred_labels)
                

    except Exception as e:
        traceback.print_exc()
        print('Unexpected auc scores error')
    return metric_scores


def save_false_positive(hparams, current_epoch, scores_all, labels_all,
                        filenames, output_dir):
    prefix = f'{hparams.fold_i}-{current_epoch}'
    fp_indexes = select_fn_indexes(scores_all, labels_all)
    fp_filenames = filenames[fp_indexes]
    fp_scores = torch.softmax(scores_all[fp_indexes], dim=1)
    fp_labels = labels_all[fp_indexes]  # one-hot label, [n, num_classes]
    fp_label_names = np.array(CLASS_NAMES)[torch.argmax(
        fp_labels, dim=1).detach().cpu().numpy()]  # label name [n, 1]
    fp_pred_names = np.array(CLASS_NAMES)[torch.argmax(
        fp_scores, dim=1).detach().cpu().numpy()]  # label name [n, 1]
    output_dir = os.path.join(output_dir, 'fp')
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'{prefix}-fp.csv')
    df = pd.DataFrame({
        'filename': fp_filenames.reshape(-1),
        'label': fp_label_names.reshape(-1),
        'pred': fp_pred_names.reshape(-1)
    })
    fp_pred = pd.DataFrame(fp_scores.detach().cpu().numpy(),
                           columns=CLASS_NAMES)
    fp_df = pd.concat([df, fp_pred], axis=1)
    if not os.path.exists(save_path):
        fp_df.to_csv(save_path, index=False)
    else:
        fp_df.to_csv(save_path, index=False, mode='a', header=False)


def generate_classification_report(hparams, current_epoch, scores_all,
                                   labels_all, filenames, output_dir):
    output_dir = os.path.join(output_dir, 'report')
    os.makedirs(output_dir, exist_ok=True)
    prefix = f'{hparams.fold_i}-{current_epoch}'
    pred_save_path = os.path.join(output_dir, f'{prefix}-pred.csv')
    # scores = torch.softmax(scores_all, dim=1)
    file_name_df = pd.DataFrame({
        'filename': filenames,
    })
    score_df = pd.DataFrame(scores_all.detach().cpu().numpy(), columns=CLASS_NAMES)
    gt_df = pd.DataFrame(labels_all.detach().cpu().numpy(),
                         columns=CLASS_NAMES)
    pred_df = pd.concat([file_name_df, score_df], axis=1)
    gt_df = pd.concat([file_name_df, gt_df], axis=1)
    pred_df.to_csv(pred_save_path, index=False)
    generate_report(pred_df, gt_df, prefix, output_dir)


def post_report(hparams,
                current_epoch,
                scores_all,
                labels_all,
                filenames,
                output_dir,
                ger_report=False):
    """ postprocessing classification results, producing classification report or 
        saving false positive to the csv files.
    Args:
        scores_all (_type_): _description_
        labels_all (_type_): _description_
        filenames (_type_): _description_
        output_dir (_type_): _description_
    """
    save_false_positive(hparams, current_epoch, scores_all, labels_all,
                        filenames, output_dir)
    if ger_report:
        generate_classification_report(hparams, current_epoch, scores_all,
                                       labels_all, filenames, output_dir)


def cat_image_in_ddp(val_epoch_out_path, cat_epoch_out_path):
    """In ddp mode, the each intermidiate output are saved by different processes. This function collect them and cat 
    these images together for better visualization.
    """
    os.makedirs(cat_epoch_out_path, exist_ok=True)
    filenames = os.listdir(val_epoch_out_path)
    filenames = sorted(filenames)
    for class_name in CLASS_NAMES:
        imgs = []
        for filename in filenames:
            if filename.startswith(class_name):
                img = cv2.imread(os.path.join(val_epoch_out_path, filename))
                if img is None:
                    continue
                imgs.append(img)
        if len(imgs):
            imgs = cv2.vconcat(imgs)
            cv2.imwrite(os.path.join(cat_epoch_out_path, f'{class_name}.jpeg'),
                        imgs)


def collect_distributed_info(hparams, outputs):
    val_loss_mean = torch.stack([output["loss"] for output in outputs]).mean()
    data_load_times = torch.stack(
        [output["data_load_time"] for output in outputs]).sum()
    batch_run_times = torch.stack(
        [output["batch_run_time"] for output in outputs]).sum()
    scores_all = torch.cat([output["scores"] for output in outputs])
    labels_all = torch.cat([output["labels"] for output in outputs])
    val_roc_auc = get_roc_auc(hparams, labels_all, scores_all)
    filenames = np.concatenate([output["filenames"] for output in outputs])
    # print(f'Pid {os.getpid()} sample filename, {filenames[0]}')
    return DotMap({
        'loss_mean': val_loss_mean,
        'scores': torch.softmax(scores_all, axis=1),
        'labels': labels_all,
        'filenames': filenames,
        'roc_auc': val_roc_auc,
        'data_load_times': data_load_times,
        'batch_run_times': batch_run_times
    })


def collect_other_distributed_info(hparams, outputs, created=True):
    other_roc_auc = 0.0
    other_loss = torch.tensor(0.0, dtype=torch.float)
    if len(outputs) > 2 and created:
        other_infos = [
            collect_distributed_info(hparams, output) for output in outputs[2:]
        ]
        other_loss = torch.stack([info.loss_mean
                                  for info in other_infos]).mean()
        other_roc_auc = np.array([info.roc_auc for info in other_infos]).mean()
    return DotMap({'loss_mean': other_loss, 'roc_auc': other_roc_auc})


def write_distributed_records(global_rank, fold, epoch, step, val_info,
                              other_info, selection_record_path,
                              performance_record_path):
    try:
        labels = np.argmax(val_info.labels.detach().cpu().numpy(), axis=1)
        preds = np.argmax(val_info.scores.detach().cpu().numpy(), axis=1)
        # labels = np.array([0, 1, 2, 3, 4, 5, 6, 7])
        # preds = np.array([1, 0, 2, 4, 3, 5, 6, 7])
        classification_results = classification_report(
            labels, preds, target_names=CLASS_NAMES, output_dict=True)

        base_row = {
            'Fold': fold,
            'Rank': global_rank,
            'Epoch': epoch,
            'Step': step,
        }

        selection_row = {
            **base_row, 'Loss': val_info.loss_mean.detach().cpu().numpy(),
            'ROC_AUC': val_info.roc_auc,
            'Other_Loss': other_info.loss_mean.detach().cpu().numpy(),
            'OTHER_ROC_AUC': other_info.roc_auc
        }

        new_selection_record = pd.DataFrame([selection_row])

        new_selection_record.to_csv(selection_record_path,
                                    mode='a',
                                    header=False,
                                    index=False,
                                    float_format='%.4f')
        new_performance_records = []
        for class_name in CLASS_NAMES:
            row = base_row.copy()
            row['class'] = class_name
            for k, v in classification_results[class_name].items():
                row[k] = v
                if k == 'precision':
                    row['in_precision'] = 1 - v
            # self.val_performance_records = self.val_performance_records.append(
            # row, ignore_index=True)
            new_performance_records.append(row)
        new_performance_records = pd.DataFrame(new_performance_records)
        new_performance_records.to_csv(performance_record_path,
                                       mode='a',
                                       header=False,
                                       index=False,
                                       float_format='%.4f')
    except Exception as e:
        traceback.print_exc()


def get_training_strategy(hparams):
    tm = hparams.training_mode
    if tm == 'ddp':
        return DDPStrategy(find_unused_parameters=False)
    elif tm == 'ddp2':
        return DDP2Strategy(find_unused_parameters=False)
    elif tm == 'dp':
        return DataParallelStrategy()


def test_render_labels():
    images = Image.open(
        '/data/lxd/datasets/2022-03-02-Eggs/OK/000445139_Egg6_(ok)_L_0_cam5.bmp'
    )
    images = to_tensor(images).reshape(1, 3, 700, 600)
    images = render_labels(images, torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0]]),
                           torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0]]))
    save_image(images, os.path.join('test_results', 'test_render_label.jpeg'))


if __name__ == '__main__':
    pass
