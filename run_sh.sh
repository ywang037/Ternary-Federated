#!/bin/sh

# CUDA_VISIBLE_DEVICES=0 python Ternary_Fed_2_resnets_sea.py --fedmdl=s2 --lr=0.0001 --batch_size=64 --local_e=5 --save_record=True

CUDA_VISIBLE_DEVICES=0 python ./Ternary_Fed_2_resnets_sea_dummy.py --fedmdl=s2 --lr=0.0001 --batch_size=64 --local_e=5 --save_record=True

CUDA_VISIBLE_DEVICES=0 python ./Ternary_Fed_2_resnets_sea_dummy.py --fedmdl=s2 --lr=0.0001 --batch_size=64 --local_e=10 --save_record=True

CUDA_VISIBLE_DEVICES=0 python ./Ternary_Fed_2_resnets_sea_dummy.py --fedmdl=s2 --lr=0.0001 --batch_size=64 --local_e=20 --save_record=True