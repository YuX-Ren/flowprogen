# from flowprogen.utils.parsing import parse_train_args
from argparse import ArgumentParser

from flowprogen.utils.logging import get_logger
logger = get_logger(__name__)

import torch, tqdm, os, wandb
import pandas as pd

from functools import partial
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
# from lightning.pytorch.callbacks import ModelCheckpoint
from openfold.utils.exponential_moving_average import ExponentialMovingAverage
from flowprogen.model.wrapper import ESMFoldWrapper, AlphaFoldWrapper, LLMFlowWrapper, TransFlowWrapper
from openfold.utils.import_weights import import_jax_weights_

torch.set_float32_matmul_precision("high")
from flowprogen.config import model_config
from flowprogen.data.data_modules import OpenFoldSingleDataset, OpenFoldBatchCollator, OpenFoldDataset
from flowprogen.data.inference import CSVDataset, AlphaFoldCSVDataset

config = model_config(
    'initial_training',
    train=True, 
    low_prec=True
) 

loss_cfg = config.loss
data_cfg = config.data
data_cfg.common.use_templates = False
data_cfg.common.max_recycling_iters = 0

def load_clusters(path):
    cluster_size = []
    with open(args.pdb_clusters) as f:
        for line in f:
            names = line.split()
            for name in names:
                cluster_size.append({'name': name, 'cluster_size': len(names)})
    return pd.DataFrame(cluster_size).set_index('name')
    
def main(args):
    
    pdb_chains = pd.read_csv(args.pdb_chains, index_col='name')

    if args.filter_chains:
        clusters = load_clusters(args.pdb_clusters)
        pdb_chains = pdb_chains.join(clusters)
        pdb_chains = pdb_chains[pdb_chains.release_date < args.train_cutoff]
    
    trainset = OpenFoldSingleDataset(
        data_dir = args.train_data_dir,
        alignment_dir = args.train_msa_dir,
        pdb_chains = pdb_chains,
        config = data_cfg,
        mode = 'train',
        subsample_pos = args.sample_train_confs,
        first_as_template = args.first_as_template,
    )
    if args.normal_validate:
        val_pdb_chains = pd.read_csv(args.val_csv, index_col='name')
        valset = OpenFoldSingleDataset(
            data_dir = args.train_data_dir,
            alignment_dir = args.train_msa_dir,
            pdb_chains = val_pdb_chains,
            config = data_cfg,
            mode = 'train',
            subsample_pos = args.sample_val_confs,
            num_confs = args.num_val_confs,
            first_as_template = args.first_as_template,
        )   
    else:
        valset = AlphaFoldCSVDataset(
            data_cfg,
            args.val_csv,
            mmcif_dir=args.mmcif_dir,
            # data_dir=args.train_data_dir,
            msa_dir=args.val_msa_dir,
        )
    if args.filter_chains:
        trainset = OpenFoldDataset([trainset], [1.0], args.train_epoch_len)
    
    val_loader = torch.utils.data.DataLoader(
        valset,
        batch_size=args.batch_size,
        collate_fn=OpenFoldBatchCollator(),
        num_workers=args.num_workers,
    )
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        collate_fn=OpenFoldBatchCollator(),
        num_workers=args.num_workers,
        shuffle=not args.filter_chains,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        strategy="deepspeed_stage_2",
        max_epochs=args.epochs,
        limit_train_batches=args.limit_batches or 1.0,
        limit_val_batches=args.limit_batches or 1.0,
        num_sanity_val_steps=0,
        enable_progress_bar=not args.wandb,
        gradient_clip_val=args.grad_clip,
        callbacks=[ModelCheckpoint(
            dirpath=os.environ["MODEL_DIR"], 
            save_top_k=2,
            monitor='val/min_ref_rmsd',
            mode='min',
            every_n_epochs=args.ckpt_freq,
        )],
        accumulate_grad_batches=args.accumulate_grad,
        check_val_every_n_epoch=args.val_freq,
        logger=False,
    )
    
    if args.wandb and trainer.is_global_zero:
        wandb.init(
            # entity=os.environ["WANDB_ENTITY"],
            # settings=wandb.Settings(start_method="fork"),
            project="flowprogen",
            name=args.run_name,
            config=args,
        )
    if args.mode == 'esmfold':
        model = ESMFoldWrapper(config, args)
        if args.ckpt is None:
            logger.info("Loading the model")
            path = "/share/project/xiaohongwang/Routine_ckpts/esm_pretrained_models/esmfold_3B_v1.pt"
            model_data = torch.load(path)
            model_state = model_data["model"]
            model.model.load_state_dict(model_state, strict=False)
            logger.info("Model has been loaded")
            
            if not args.no_ema:
                model.ema = ExponentialMovingAverage(
                    model=model.model, decay=config.ema.decay
                ) # need to initialize EMA this way at the beginning
    elif args.mode == 'alphafold':
        model = AlphaFoldWrapper(config, args)
        if args.ckpt is None:
            logger.info("Loading the model")
            import_jax_weights_(model.model, '/share/project/database/database/params_v3/params_model_1.npz', version='model_3')
            if not args.no_ema:
                model.ema = ExponentialMovingAverage(
                    model=model.model, decay=config.ema.decay
                ) # need to initialize EMA this way at the beginning
    elif args.mode == 'llmflow':
        model = LLMFlowWrapper(config, args)
    elif args.mode == 'transflow':
        model = TransFlowWrapper(config, args)
        
    if args.restore_weights_only:
        model.load_state_dict(torch.load(args.ckpt, map_location='cpu')['state_dict'], strict=False)
        args.ckpt = None
        if not args.no_ema:
            model.ema = ExponentialMovingAverage(
                model=model.model, decay=config.ema.decay
            ) # need to initialize EMA this way at the beginning
    
    if args.validate: # only validate
        trainer.validate(model, val_loader, ckpt_path=args.ckpt)
    else: # train and validate
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt)

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--mode", choices=['esmfold', 'alphafold', 'llmflow', 'transflow'], default='transflow')
    
    ## Trainer settings
    parser.add_argument("--ckpt", type=str, default=None)
    parser.add_argument("--restore_weights_only", action='store_true')
    parser.add_argument("--validate", action='store_true')
    
    ## Epoch settings
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--train_epoch_len", type=int, default=40000)
    parser.add_argument("--limit_batches", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)

    ## Optimization settings
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--check_grad", action="store_true")
    parser.add_argument("--accumulate_grad", type=int, default=1)
    parser.add_argument("--grad_clip", type=float, default=1.)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--no_ema", type=bool, default=True)
    
    ## Training data 
    parser.add_argument("--train_data_dir", type=str, default='./data')
    parser.add_argument("--pdb_chains", type=str, default='./pdb_mmcif_msa.csv')
    parser.add_argument("--train_msa_dir", type=str, default='./msa_dir')
    parser.add_argument("--pdb_clusters", type=str, default='./pdb_clusters')
    parser.add_argument("--train_cutoff", type=str, default='2021-10-01')
    parser.add_argument("--mmcif_dir", type=str, default='./mmcif_dir')
    parser.add_argument("--filter_chains", action='store_true')
    parser.add_argument("--sample_train_confs", action='store_true')
    
    ## Validation data
    parser.add_argument("--val_csv", type=str, default='splits/cameo2022.csv')
    parser.add_argument("--val_samples", type=int, default=5)
    parser.add_argument("--val_msa_dir", type=str, default='./alignment_dir/cameo2022')
    parser.add_argument("--sample_val_confs", action='store_true')
    parser.add_argument("--num_val_confs", type=int, default=None)
    parser.add_argument("--normal_validate", action='store_true')
    
    ## Flow matching
    parser.add_argument("--noise_prob", type=float, default=0.5)
    parser.add_argument("--self_cond_prob", type=float, default=0.5)
    parser.add_argument("--extra_input", action='store_true')
    parser.add_argument("--extra_input_prob", type=float, default=0.5)
    parser.add_argument("--first_as_template", action='store_true')
    parser.add_argument("--distillation", action='store_true')
    parser.add_argument("--distill_self_cond", action='store_true')
    
    ## Logging args
    parser.add_argument("--print_freq", type=int, default=100)
    parser.add_argument("--val_freq", type=int, default=1)
    parser.add_argument("--ckpt_freq", type=int, default=1)
    parser.add_argument("--wandb", action='store_true')
    parser.add_argument("--run_name", type=str, default="default")
    
    args = parser.parse_args()
    if not os.path.exists(os.path.join("workdir", args.run_name)):
        os.makedirs(os.path.join("workdir", args.run_name))
    os.environ["MODEL_DIR"] = os.path.join("workdir", args.run_name)
    # os.environ["WANDB_LOGGING"] = str(int(args.wandb))
    # # if args.wandb:
    #     if subprocess.check_output(["git", "status", "-s"]):
    #         print("Warning: git status is not clean, skipping wandb logging")
    #         exit()
    # args.commit = (
    #     subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    # )

    main(args)