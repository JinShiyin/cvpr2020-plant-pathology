# CUDA_VISIBLE_DEVICES=0,1 python generate_distill_submission2.py   --gpus 0 1 --min_epochs 70 --max_epochs 70 --val_batch_size 232
CUDA_VISIBLE_DEVICES=1 python test.py  --config config/test2.yaml
