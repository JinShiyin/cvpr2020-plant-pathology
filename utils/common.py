from sklearn.metrics import roc_auc_score
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
class_names = ['OK', 'AirRoomShake', 'Dead', 'Empty', 'NoAirRoom', 'Split', 'Weak', 'Flower']

bn_class_names = ['OK', 'NoOK']

header_names = ['filename'] + class_names 

class_label_to_name = {0: 'ok', 1: 'qishihuangdong', 2 : 'sipei', 3: 'kongliao', 4: 'wuqishi', 5: 'liewen', 6: 'ruopei', 7: 'huake'}
name_to_class_label = {'ok': 0 , 'qishihuangdong' : 1, 'sipei' : 1, 'kongliao' : 1, 'wuqishi' : 1, 'liewen' : 1, 'ruopei' : 1}

def extract_activation_map(cam_extractor: SmoothGradCAMpp, images, preds):
    activation_map = cam_extractor(preds.argmax(dim=1).tolist(), preds)[0]
    # print(len(activation_map))
    activation_map = torch.nn.functional.interpolate(activation_map.unsqueeze(1), 
                                                     size = images.size()[2:], 
                                                     mode ='bicubic',
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
        heat_map = torch.from_numpy(cmap((mask ** 2).detach().cpu().numpy())[:, :, :3]).to(mask.device)
        heat_maps.append(heat_map)
        # print(heat_map.size())
        # [b, n, h, w, 3] -> [b, n, 3, h, w]
    return torch.stack(heat_maps).reshape(b, n, h, w, -1).permute(0, 1, 4, 2, 3)
    
def overlay(images, heat_maps, alpha = 0.5):
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
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    image = cv2.putText(image, f'Pred: {pred}', (50, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
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
 

def visualization(batch_id, cam_extractors : List[SmoothGradCAMpp], images, preds, labels, filenames, output_dir, save_batch=True,save_per_image=False, fp_indexes= None, norm=True, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """render the convolutional activation in the images.

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
    activation_maps = torch.cat([extract_activation_map(cam, images, preds) for cam in cam_extractors]
                                , dim=1)
    heat_maps = generate_heatmaps(activation_maps, 'jet')
    # print(heat_maps.size())

    images = img_denorm(images, 
                    mean=mean,
                    std=std) 
    images = images.unsqueeze(1)
    mask_images = overlay(images, heat_maps)
    images = render_labels(images, labels, preds)
    results = torch.cat([images, mask_images], dim=1).reshape(b*(n+1), 3, h, w)
    # print(results.size())
    if save_batch:
        save_image(results, os.path.join(output_dir, f'{batch_id}.jpeg'), nrow=n+1)
    if save_per_image:
       per_results = results.view(b , n+1, 3, h, w)
       for filename, image in zip(filenames, per_results):
           prefix = os.path.splitext(filename)[0].replace('/', '_')
           save_image(image, os.path.join(output_dir, f'{prefix}.jpeg'), nrow =n+1)

    # save false negative by class.
    if fp_indexes is not None:
        # if fn_indexes is None:
        results = results.reshape(b , n+1, 3, h, w)
        fp_output_dir = os.path.join(output_dir, 'fp')
        os.makedirs(fp_output_dir, exist_ok=True)
        # selected the false positive
        labels = torch.argmax(labels, dim=1).detach().cpu().numpy() # [b, 1] transfer one-hot into class index
        fp_results = results[fp_indexes]
        if not len(fp_results):
            print(f'Batch id {batch_id}: Not found false negative samples in batch.')
            return
        fn_filenames = np.array(filenames)[fp_indexes]
        fp_labels = labels[fp_indexes]
        for label, result, filename in zip(fp_labels, fp_results, fn_filenames):
            class_dir = os.path.join(fp_output_dir, class_label_to_name[label])
            if not os.path.exists(class_dir):
                os.makedirs(class_dir, exist_ok=True)
            filename = os.path.splitext(filename)[0].replace('/', '_')
            save_image(result, os.path.join(class_dir, f'{filename}.jpeg'), nrow =n+1)
        

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
    pred_class = torch.argmax(pred, dim =1)
    fn_indexes = torch.ne(pred_class, label_class) & torch.eq(pred_class, 0)
    fn_indexes = fn_indexes.detach().cpu().numpy()
    return fn_indexes


def get_roc_auc(labels, scores):
    class_num = labels.argmax(dim=1).unique() 
    val_roc_auc = 0
    try:
        if len(class_num) == 1:
            val_roc_auc = 0
        else:
            # print(labels)
            val_roc_auc = roc_auc_score(labels.cpu().numpy(), scores.cpu().numpy())
    except Exception as e:
        print('Unexpected auc scores error')
    return val_roc_auc

def test_render_labels():
    images = Image.open('/data/lxd/datasets/2022-03-02-Eggs/OK/000445139_Egg6_(ok)_L_0_cam5.bmp')
    images = to_tensor(images).reshape(1, 3, 700, 600)
    images = render_labels(images, torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0]]), torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0]]))
    save_image(images, os.path.join('test_results', 'test_render_label.jpeg'))
    

        

if __name__ == '__main__':
    pass

        
        