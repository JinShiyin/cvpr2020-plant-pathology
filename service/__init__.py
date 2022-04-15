from multiprocessing import Process, Manager, Pool
import cv2
from .inference import worker, assemble_result

PATH = [
"logs_submit/fold=0-epoch=67-val_loss=0.0992-val_roc_auc=0.9951.ckpt",
"logs_submit/fold=1-epoch=61-val_loss=0.1347-val_roc_auc=0.9928.ckpt",
"logs_submit/fold=2-epoch=57-val_loss=0.1289-val_roc_auc=0.9968.ckpt",
"logs_submit/fold=3-epoch=48-val_loss=0.1161-val_roc_auc=0.9980.ckpt",
"logs_submit/fold=4-epoch=67-val_loss=0.1012-val_roc_auc=0.9979.ckpt"
]

num_workers = len(PATH)
manager = Manager()
input_entry = [manager.Queue() for idx in range(num_workers)]
output_entry = [manager.Queue() for idx in range(num_workers)]

def run():
    with Pool(num_workers) as pool:
        for idx, checkpoint in enumerate(PATH):
            p = pool.apply_async(worker,args=(idx, checkpoint, input_entry[idx],output_entry[idx]))
        pool.close()
        pool.join()