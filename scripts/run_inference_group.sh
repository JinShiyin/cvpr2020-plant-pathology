#CUDA_VISIBLE_DEVICES=0,1 python inference_groups.py  --gpus 0 1 --min_epochs 70 --max_epochs 70 --val_batch_size 6
CUDA_VISIBLE_DEVICES=1 python inference_groups.py  --config config/test2.yaml
