# python train.py --train_batch_size 96 --gpus 0 1 --min_epochs 70 --max_epochs 70 --num_workers 12
# python train.py --train_batch_size 12 --val_batch_size 6 --gpus 2 3 --min_epochs 70 --max_epochs 70 --num_workers 8
# CUDA_VISIBLE_DEVICES=0,1 python train.py --config config/train_[original]_small_batch_v2.yaml
python train.py --config config/train_[original]_small_batch_v2.yaml
# CUDA_VISIBLE_DEVICES=1,2,3 python train.py --config /data/lxd/project/cvpr2020-plant-pathology/logs_submit/20220420-1307-roi/checkpoints/fold=1-epoch=8-val_loss=0.3057-val_roc_auc=0.9842.ckpt
