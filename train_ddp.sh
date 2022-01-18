#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node 4 --master_port=2333 \
    train_ddp.py --conf_file=configs/config_final_4view.yaml --scene=mit-16  \
    --log_dir=test_code/mit-16



