from flowprogen.utils.parsing import parse_train_args
args = parse_train_args()

import os
# import torch
import wandb
import pandas as pd
from shutil import rmtree
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader

# torch.set_float32_matmul_precision("high")
from flowprogen.config import model_config
from flowprogen.data.data_modules import OpenFoldSingleDataset, OpenFoldBatchCollator, OpenFoldDataset
from flowprogen.model.wrapper import LLMFlowWrapper, TransFlowWrapper
from flowprogen.utils.logging import get_logger
logger = get_logger(__name__)



is_main_process = pl.utilities.rank_zero_only.rank == 0

if is_main_process:
    wandb.init(
        project="dillm", 
        name=args.run_name,
    )
    rmtree(f'./results_llmflow/{args.run_name}', ignore_errors=True)
    results_folder = Path(f'./results_llmflow/{args.run_name}')
    results_folder.mkdir(exist_ok=True, parents=True)

# device = torch.device('cuda:0')

config = model_config(
    'initial_training',
    train=True, 
    low_prec=True
) 
data_cfg = config.data
data_cfg.common.use_templates = False
data_cfg.common.max_recycling_iters = 0

if is_main_process:
    logger.info(f'config: {config.keys()}')

trainer = pl.Trainer(
    accelerator="gpu",
    max_epochs=args.epochs,
    limit_train_batches=args.limit_batches or 1.0,
    limit_val_batches=args.limit_batches or 1.0,
    num_sanity_val_steps=0,
    enable_progress_bar=not args.wandb,
    gradient_clip_val=args.grad_clip,
    callbacks=[ModelCheckpoint(
        dirpath=os.environ["MODEL_DIR"], 
        save_top_k=-1,
        every_n_epochs=args.ckpt_freq,
    )],
    accumulate_grad_batches=args.accumulate_grad,
    check_val_every_n_epoch=args.val_freq,
    logger=False,
)

trainset = OpenFoldSingleDataset(
    data_dir = args.train_data_dir,
    alignment_dir = args.train_msa_dir,
    pdb_chains = pdb_chains,
    config = data_cfg,
    mode = 'train',
    subsample_pos = args.sample_train_confs,
    first_as_template = args.first_as_template,
)

if args.filter_chains:
    trainset = OpenFoldDataset([trainset], [1.0], epoch_len=args.train_epoch_len)

train_loader = DataLoader(
    trainset,
    batch_size=args.batch_size,
    collate_fn=OpenFoldBatchCollator(),
    num_workers=args.num_workers,
    shuffle=not args.filter_chains,
)


model_cfg = config.model
trunk_cfg = config.model.trunk

model = LLMFlowWrapper(config, args)

val_loader = None
trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt)