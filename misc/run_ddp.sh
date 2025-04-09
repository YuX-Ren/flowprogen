#!/bin/bash

# 设置NCCL_DEBUG以获取更多信息
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=eth0  # 这里根据你的网络接口名称进行修改

# 使用所有可见GPU
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
echo "using $NUM_GPUS GPUs for distributed training"

# 启动分布式训练
OMP_NUM_THREADS=1 torchrun --nproc_per_node=$NUM_GPUS \
    --master_port=29500 \
    train_latent_only_dillm_esmfold_ddp.py