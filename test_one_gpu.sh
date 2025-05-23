# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export NCCL_SOCKET_IFNAME=eth0  # 根据实际网络接口修改
# export NCCL_IB_DISABLE=1  # 如果使用 InfiniBand，可以尝试禁用
# export NCCL_P2P_DISABLE=1  # 如果使用多机训练，可以尝试禁用 P2P


CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 train_seq_cheap.py \
    --val_freq 1 \
    --lr 5e-4 \
    --epochs 10000 \
    --train_epoch_len 1000 \
    --train_data_dir ./data \
    --mode transflow \
    --run_name transflow_seq_only \
    --print_freq 1000 \
    --wandb \
    --normal_validate
    # --no_ema \