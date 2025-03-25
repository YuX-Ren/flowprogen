#!/bin/bash

# 设置环境变量
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0  # 根据您的服务器网络接口来调整

# 获取可用GPU数量
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "using $NUM_GPUS GPUs for distributed training"

# 设置分布式训练参数
MASTER_PORT=$((RANDOM % 10000 + 10000))  # 随机端口避免冲突

# 启动分布式训练
torchrun --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    train_latent_only_dillm_esmfold_fsdp.py \
    --num_gpus=$NUM_GPUS

echo "FSDP training completed" 