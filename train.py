# from flowprogen.utils.parsing import parse_train_args
from argparse import ArgumentParser

from flowprogen.utils.logging import get_logger
logger = get_logger(__name__)

import torch, tqdm, os, wandb
import pandas as pd
import torch.distributed as dist
from functools import partial
import pytorch_lightning as pl
# from lightning.pytorch.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DeepSpeedStrategy, DDPStrategy
from openfold.utils.exponential_moving_average import ExponentialMovingAverage
from flowprogen.model.wrapper import TransFlowWrapper, LLMFlowWrapper

# torch.set_float32_matmul_precision("high")
from flowprogen.config import model_config
from flowprogen.data.data_modules import OpenFoldSingleDataset, OpenFoldBatchCollator, OpenFoldDataset
from flowprogen.data.inference import CSVDataset, AlphaFoldCSVDataset
torch._dynamo.config.optimize_ddp=False


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
    with open(path) as f:
        for line in f:
            names = line.split()
            for name in names:
                cluster_size.append({'name': name, 'cluster_size': len(names)})
    return pd.DataFrame(cluster_size).set_index('name')
    
def init_distributed_training(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ['RANK'])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                           world_size=args.world_size, rank=args.rank)
    dist.barrier(device_ids=[torch.cuda.current_device()])

def main(args):
    # Initialize distributed training
    init_distributed_training(args)

    is_global_zero = not dist.is_initialized() or dist.get_rank() == 0

    if args.wandb and is_global_zero:
        wandb_logger = WandbLogger(
                project="flowprogen",
                name=args.run_name,             
                save_dir="./",            
                log_model=False,                   
            )  
    else:
        wandb_logger = False

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

    
    # Create distributed sampler
    # train_sampler = torch.utils.data.DistributedSampler(trainset, shuffle=not args.filter_chains)
    train_loader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        collate_fn=OpenFoldBatchCollator(),
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=not args.filter_chains,
        # persistent_workers=True,
        # prefetch_factor=2,
        # drop_last=True,
        # sampler=train_sampler
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        # strategy=DeepSpeedStrategy(stage=2, config='./zero2.json'),
        strategy=DDPStrategy(find_unused_parameters=True,
                             gradient_as_bucket_view=True),
        max_epochs=args.epochs,
        limit_train_batches=args.limit_batches or 1.0,
        limit_val_batches=args.limit_batches or 1.0,
        num_sanity_val_steps=0,
        enable_progress_bar=True,
        gradient_clip_val=args.grad_clip,
        callbacks=[ModelCheckpoint(
            dirpath=os.environ["MODEL_DIR"], 
            save_top_k=1,
            monitor='val/min_ref_rmsd',
            mode='min',
            every_n_epochs=args.ckpt_freq,
        )],
        accumulate_grad_batches=args.accumulate_grad,
        check_val_every_n_epoch=args.val_freq,
        logger=wandb_logger,
        log_every_n_steps=10,
        use_distributed_sampler=True,
    )
    
    if args.mode == 'transflow':
        model = TransFlowWrapper(config, args)
    elif args.mode == 'llmflow':
        model = LLMFlowWrapper(config, args)

    if args.restore_weights_only:
        model.load_state_dict(torch.load(args.ckpt, map_location='cpu')['state_dict'], strict=False)
        args.ckpt = None
        if not args.no_ema:
            model.ema = ExponentialMovingAverage(
                model=model.model, decay=config.ema.decay
            ) # need to initialize EMA this way at the beginning
    
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    
    if args.validate: # only validate
        trainer.validate(model, val_loader, ckpt_path=args.ckpt)
    else: # train and validate
        trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt)
    
    wandb.finish()

if __name__ == "__main__":
    parser = ArgumentParser()

    parser.add_argument("--mode", choices=['transflow', 'llmflow'], default='transflow')
    parser.add_argument("--local_rank", type=int, default=-1)
    
    # Add distributed training arguments
    parser.add_argument("--dist_url", type=str, default="env://",
                       help="url used to set up distributed training")
    parser.add_argument("--dist_backend", type=str, default="nccl",
                       help="distributed backend")
    
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
    parser.add_argument("--no_ema", action='store_true')
    
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