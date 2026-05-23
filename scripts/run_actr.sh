#!/bin/bash

# ACTR without BPR
CUDA_VISIBLE_DEVICES=0 python -m au2actr eval --verbose -p configs/deezer/baselines/actr.json

# Train BPR
CUDA_VISIBLE_DEVICES=0 python -m au2actr train --verbose -p configs/deezer/baselines/actr_bpr.json \
>& exp/logs/deezer_min300sess/actr_bpr/actr_bpr_lr0.001_batch512_dim128_l2emb1e-5_neguni.txt

CUDA_VISIBLE_DEVICES=0 python -m au2actr eval --verbose -p configs/deezer/baselines/actr_bpr.json