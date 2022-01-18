#!/bin/bash

CUDA_VISIBLE_DEVICES=9 python generate_result.py --model_path test_code/mit-16 \
    --out_path test_result/mit-16 \
    --n 1 \
    --save_background



