# export NCCL_DEBUG=INFO
# export NCCL_DEBUG_SUBSYS=ALL
# export NCCL_SOCKET_IFNAME=eth0  # 根据实际网络接口修改
# export NCCL_IB_DISABLE=1  # 如果使用 InfiniBand，可以尝试禁用
# export NCCL_P2P_DISABLE=1  # 如果使用多机训练，可以尝试禁用 P2P

torchrun --nproc_per_node=8 train.py \
    --lr 5e-4 \
    --noise_prob 0.8 \
    --accumulate_grad 8 \
    --epochs 13 \
    --train_epoch_len 40000 \
    --train_cutoff 2020-05-01 \
    --filter_chains \
    --train_data_dir /share/project/xiaohongwang/Datasets/pdb_mmcif_data_npz \
    --train_msa_dir /share/project/xiaohongwang/Datasets/openfold/pdb \
    --mmcif_dir /share/project/xiaohongwang/Datasets/pdb_mmcif \
    --pdb_chains ./pdb_mmcif_msa.csv \
    --pdb_clusters ./pdb_clusters \
    --val_csv ./splits/cameo2022.csv \
    --val_msa_dir ./alignment_dir/cameo2022 \
    --mode transflow \
    --run_name transflow_esmfold \
    --print_freq 1000 \
    --wandb