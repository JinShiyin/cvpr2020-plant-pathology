from ntpath import join
from queue import Empty, Queue
import traceback
import cv2

# Third party libraries
from scipy.special import softmax
from tqdm import tqdm
import gc

# User defined libraries
from dataset import OpticalCandlingDataset, generate_transforms, PlantDataset
from utils import init_hparams, init_logger, load_test_data, seed_reproducer, load_data
from multiprocessing import Process, Manager, Pool
import numpy as np

# ['OK', 'qishihuangdong','sipei', 'kongliao', 'wuqishi','liewen', 'ruopei']
# label_dict = {0: '好', 1: '气室抖动', 2 : '死胚', 3: '空', 4: '无气室', 5: '裂纹', 6: '弱胚', }
label_dict = {0: 'ok', 1: 'qishihuangdong', 2 : 'sipei', 3: 'kongliao', 4: 'wuqishi', 5: 'liewen', 6: 'ruopei'}

def worker(model_index, checkpoint, 
                  recv_pipe: Queue,
                  output_pipe: Queue):
    from train import CoolSystem
    from torch.utils.data import DataLoader
    import os
    os.environ["CUDA_VISIBLE_DEVICES"]= str(model_index % 4)
    import torch
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    hparams = init_hparams()

    # Make experiment reproducible
    seed_reproducer(hparams.seed)

    logger = init_logger("kun_out", log_dir=hparams.log_dir)

    # Generate transforms
    # transforms = generate_transforms(hparams.image_size)['val_transforms']
    transforms = generate_transforms(hparams.image_size)['train_transforms']

    # Instance Model, Trainer and train model
    model = CoolSystem(hparams).cuda()

    model.load_state_dict(torch.load(checkpoint, map_location="cuda")["state_dict"])
    model.eval()
    logger.info(
        f'Classifier [{model_index}]: Running Inference Services: checkpoint from {checkpoint}')
    while True:
        try:
            image = recv_pipe.get(timeout=5)
            image = transforms(image=image)["image"].transpose(2, 0, 1)
            logger.info(
                f'Classifier [{model_index}]: receive signal.')
            image = torch.from_numpy(image).cuda().unsqueeze(0)
            # image = torch.from_numpy(image).to(device).unsqueeze(0)
            preds = model(image)
            preds = preds.detach().cpu().numpy()
            output_pipe.put(preds)
            # torch.cuda.empty_cache()
            logger.info(
                f'Classifier [{model_index}]: Done.')
        except Empty as e:
            pass
        except Exception as e:
            traceback.print_exc()
            return

def assemble_result(submission):
    submission_ensembled = 0
    for sub in submission:
        # sub: N * num_classes
        submission_ensembled += softmax(sub, axis=1) / len(submission)
    class_label = np.argmax(submission_ensembled, axis=1)[0]
    # return {class_label : label_dict[class_label]}
    return label_dict[class_label]

if __name__ == "__main__":
    # Init Hyperparameters
    # image_path = '/data/lxd/datasets/2021-12-12-Eggs/OK/000445139_Egg6_(ok)_R_0_cam6.bmp'
    # image_path = '/data/lxd/datasets/2021-12-12-Eggs/AirRoomShake/093941599_Egg3_(qishihuangdong)_R_0_cam4.bmp'
    # image_path = '/data/lxd/datasets/2021-12-12-Eggs/Dead/174427969_Egg5_(sipei)_R_0_cam6.bmp'
    # image_path = '/data/lxd/datasets/2021-12-12-Eggs/Empty/175008598_Egg1_(kongliao)_L_0_cam1.bmp'
    # image_path = '/data/lxd/datasets/2021-12-12-Eggs/NoAirRoom/165445307_Egg6_(wuqishi)_L_0_cam5.bmp'
    # image_path = '/data/lxd/datasets/2021-12-12-Eggs/Split/204805659_Egg2_(liewen)_R_0_cam2.bmp'
    image_path = '/data/lxd/datasets/2021-12-12-Eggs/Weak/174409520_Egg3_(ruopei--ok)_L_0_cam3.bmp'

    image = cv2.cvtColor(cv2.imread(image_path),cv2.COLOR_BGR2RGB)
    # image = image.transpose(1, 0, 2)

    submission = []
    PATH = [
        "logs_submit/fold=0-epoch=67-val_loss=0.0992-val_roc_auc=0.9951.ckpt",
        "logs_submit/fold=1-epoch=61-val_loss=0.1347-val_roc_auc=0.9928.ckpt",
        "logs_submit/fold=2-epoch=57-val_loss=0.1289-val_roc_auc=0.9968.ckpt",
        "logs_submit/fold=3-epoch=48-val_loss=0.1161-val_roc_auc=0.9980.ckpt",
        "logs_submit/fold=4-epoch=67-val_loss=0.1012-val_roc_auc=0.9979.ckpt"
    ]

    manager = Manager()
    num_workers = len(PATH)
    rec_pipes = [manager.Queue() for idx in range(num_workers)]
    output_pipes = [manager.Queue() for idx in range(num_workers)]
    proc_instances = []

    with Pool(num_workers) as pool:
        for idx, checkpoint in enumerate(PATH):
            p = pool.apply_async(worker,
                args=(idx, checkpoint, rec_pipes[idx],
                        output_pipes[idx])),
        pool.close()

        for idx in range(num_workers):
            rec_pipes[idx].put(image)
        
        for idx in range(num_workers):
            res = output_pipes[idx].get()
            submission.append(res)

        submission_ensembled = 0
        for sub in submission:
            # sub: N * num_classes
            submission_ensembled += softmax(sub, axis=1) / len(submission)
        class_label = np.argmax(submission_ensembled, axis=1)[0]
        print(label_dict[class_label])
        exit(0)
        pool.join()
