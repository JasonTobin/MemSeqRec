#!/bin/bash

# Pisa-au: audio is used only in output
CUDA_VISIBLE_DEVICES=0 python -m au2actr train --verbose -p configs/deezer/au2actr.json \
>& exp/logs/deezer_min300sess/au2actr/au2actr_lr0.001_ldapredactr1_actrdrop0_ldaenc0.6_audrop0_bprenc_hdim512_relu_pop0.5.txt

CUDA_VISIBLE_DEVICES=0 python -m au2actr eval --verbose -p configs/deezer/au2actr.json
