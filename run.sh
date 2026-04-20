#!/bin/bash


# CUDA_VISIBLE_DEVICES=1 python ./DualGCN/train.py \
#     --model_name dapgcn_bert \
#     --dataset laptop \
#     --seed 705 \
#     --bert_lr 2e-5 \
#     --num_epoch 15 \
#     --hidden_dim 768 \
#     --max_length 120 \
#     --cuda 1 \
#     --losstype doubleloss \
#     --alpha 0.4 \
#     --beta 0.3 \
#     --bert_dropout 0.3 \
#     --parseadj


# * restaurant

# CUDA_VISIBLE_DEVICES=0 \
#     python ./DualGCN/train.py \
#     --model_name dapgcn_bert \
#     --dataset restaurant \
#     --seed 705 \
#     --bert_lr 2e-5 \
#     --num_epoch 10 \
#     --hidden_dim 768 \
#     --max_length 120 \
#     --cuda 0 \
#     --losstype doubleloss \
#     --alpha 0.2 \
#     --beta 0.7 \
#     --parseadj




# CUDA_VISIBLE_DEVICES=1 \
#     python ./DualGCN/train.py \
#     --model_name dapgcn_bert \
#     --dataset twitter2015 \
#     --seed 705 \
#     --bert_lr 2e-5 \
#     --num_epoch 15 \
#     --hidden_dim 768 \
#     --max_length 120 \
#     --cuda 0 \
#     --losstype doubleloss \
#     --alpha 0.5 \
#     --beta 0.9 \
#     --parseadj



# CUDA_VISIBLE_DEVICES=1 \
#     python ./DualGCN/train.py \
#     --model_name dapgcn_bert \
#     --dataset twitter2017 \
#     --seed 705 \
#     --bert_lr 2e-5 \
#     --num_epoch 15 \
#     --hidden_dim 768 \
#     --max_length 120 \
#     --cuda 0 \
#     --losstype doubleloss \
#     --alpha 0.3 \
#     --beta 0.9 \
#     --parseadj



CUDA_VISIBLE_DEVICES=0 \
    python ./DualGCN/train.py \
    --model_name dapgcn_bert \
    --dataset MAMS \
    --seed 705 \
    --bert_lr 2e-5 \
    --num_epoch 15 \
    --hidden_dim 768 \
    --max_length 120 \
    --cuda 0 \
    --losstype doubleloss \
    --alpha 0.3 \
    --beta 1.0 \
    --parseadj







