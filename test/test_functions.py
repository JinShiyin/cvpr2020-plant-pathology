import sys
import os
sys.path.append('./')
import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping

# Third party libraries
import torch
from scipy.special import softmax
from torch.utils.data import DataLoader
from torchcam.methods.activation import CAM
from torchvision.utils import save_image
from tqdm import tqdm

from torchvision.transforms.functional import to_tensor
from utils import *

def test_generate_activation_maps():
    test_dir = 'test_results' 
    os.makedirs(test_dir, exist_ok=True)
    b = 2
    n =4
    h = 700
    w = 600

    images = Image.open('/data/lxd/datasets/2022-03-02-Eggs/OK/000445139_Egg6_(ok)_L_0_cam5.bmp')
    images = to_tensor(images).reshape(1, 1, 3, h, w).expand(b, 1, 3, h, w)
    # images = torch.rand((b, 3, h, w))
    # images = images.unsqueeze(1) # [b, 1, 3, h, w]
    activation_maps = torch.rand((b, n, h, w))
    heat_maps = generate_heatmaps(activation_maps, 'jet')
    mask_images = overlay(images, heat_maps) 
    results = torch.cat([images, mask_images], dim=1)
    results = results.reshape(b * (n+1), 3,  h, w)
    save_image(results, os.path.join(test_dir,'test_saliency.png'), nrow=n+1)

def test_eq():
    pred_class = torch.rand((10, 8))
    label_class = torch.rand((10, 8))
    preds = torch.argmax(pred_class, dim = 1)
    labels = torch.argmax(label_class, dim = 1)
    results = torch.ne(preds, labels)
    indexes = results & torch.eq(labels, 0)
    print(preds)
    print(indexes)
    print(preds[indexes])
    
if __name__ == '__main__':
    test_generate_activation_maps()
    # test_eq()
