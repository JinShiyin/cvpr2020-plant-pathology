# python train.py --train_batch_size 96 --gpus 0 1 --min_epochs 70 --max_epochs 70 --num_workers 12
# python train.py --train_batch_size 12 --val_batch_size 6 --gpus 2 3 --min_epochs 70 --max_epochs 70 --num_workers 8
# CUDA_VISIBLE_DEVICES=0,1 python train.py --config config/train_[original]_small_batch_v2.yaml
CUDA_VISIBLE_DEVICES=1,2,3 python train.py --config config/train_[roi]_small_batch_v2.yaml
