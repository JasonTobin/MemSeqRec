#!/bin/bash

# Last Session
CUDA_VISIBLE_DEVICES=0 python -m au2actr train --verbose -p configs/deezer/baselines/pisa.json \
>& exp/logs/deezer_min300sess/pisa/pisa_lr0.001_batch512_seqlen30_dim128_nblocks2_nheads2_dropout0_l2emb0_ldatask0.9_ldapos0.9_ldals0.4_nfavs20_negpop0.2.txt

CUDA_VISIBLE_DEVICES=0 python -m au2actr eval --verbose -p configs/deezer/baselines/pisa.json
