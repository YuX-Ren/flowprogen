python train.py --lr 5e-4 --noise_prob 0.8 --accumulate_grad 8 --train_epoch_len 20000 --train_cutoff 2020-05-01 --filter_chains \
    --train_data_dir /share/project/xiaohongwang/Datasets/pdb_mmcif_data_npz \
    --train_msa_dir /share/project/xiaohongwang/Datasets/openfold/pdb \
    --mmcif_dir /share/project/xiaohongwang/Datasets/pdb_mmcif \
    --pdb_chains ./pdb_mmcif_msa.csv \
    --pdb_clusters ./pdb_clusters \
    --mode transflow \
    --run_name train_latent_only_transflow_esmfold

    