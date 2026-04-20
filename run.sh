#!/bin/bash

# * laptop

# * DualGCN

# CUDA_VISIBLE_DEVICES=1 python ./DualGCN/train.py --model_name semgcn --dataset laptop --seed 123 --num_epoch 50 --vocab_dir ./DualGCN/dataset/Laptops_corenlp --cuda 1 --alpha 0.2 --beta 0.2 --parseadj
# CUDA_VISIBLE_DEVICES=0 python ./DualGCN/train.py --model_name syngcn --dataset laptop --seed 818 --num_epoch 50 --vocab_dir ./DualGCN/dataset/Laptops_corenlp --cuda 0 --alpha 0.2 --beta 0.2 --parseadj


# --pretrained_model_path /home/liugaofei/wyj/DualGCN-ABSA-main3/DualGCN/no_posAtten_7864_7552_weight_best_train_acc
# CUDA_VISIBLE_DEVICES=1 python ./DualGCN/train_correct_average_evaldata.py \
#     --model_name no_posAtten_gcn_bert \
#     --dataset laptop \
#     --seed 612 \
#     --num_epoch 50 \
#     --vocab_dir ./DualGCN/dataset/Laptops_corenlp \
#     --cuda 1 \
#     --losstype doubleloss \
#     --alpha 0.2 \
#     --beta 0.2 \
#     --parseadj \
    # --pretrained_model_path /home/liugaofei/wyj/DualGCN-ABSA-main3/DualGCN/state_dict_correct_testdap/dapgcn_laptop_acc_0.7911_f1_0.7587



# * DualGCN with Bert (为了适应prompt长度，max_length由100改为了120)(放fakePolarity的length是125)
# 目前表现最好的alpha：0.4  beta: 0.3  dep:attention pos；gate
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

# * DualGCN
# CUDA_VISIBLE_DEVICES=0 \
#     python ./DualGCN/train.py \
#     --model_name dapgcn \
#     --dataset restaurant \
#     --seed 818 \
#     --num_epoch 50 \
#     --vocab_dir ./DualGCN/dataset/Restaurants_corenlp \
#     --cuda 0 \
#     --losstype doubleloss \
#     --alpha 0.2 \
#     --beta 0.3 \
#     --parseadj
# * DualGCN with Bert     --alpha 0.6 \--beta 0.9 \ (原来的alpha和beta) (带prompt的是0.2 0.7)
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


# * twitter

# * DualGCN
# CUDA_VISIBLE_DEVICES=1 \
#     python ./DualGCN/train.py \
#     --model_name dapgcn \
#     --dataset twitter \
#     --seed 81 \
#     --num_epoch 50 \
#     --vocab_dir ./DualGCN/dataset/Tweets_corenlp \
#     --cuda 1 \
#     --losstype doubleloss \
#     --alpha 0.3 \
#     --beta 0.2 \
#     --parseadj
# * DualGCN with Bert    best: 1:1:0.9   dep:attention pos；gate
# CUDA_VISIBLE_DEVICES=1 \
#     python ./DualGCN/train.py \
#     --model_name dapgcn_bert \
#     --dataset twitter \
#     --seed 705 \
#     --bert_lr 2e-5 \
#     --num_epoch 15 \
#     --hidden_dim 768 \
#     --max_length 120 \
#     --cuda 1 \
#     --losstype doubleloss \
#     --alpha 0.5 \
#     --beta 0.9 \
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







