import argparse
from decode_cheap import DecodeLatent
import torch
from ml_collections.config_dict import config_dict

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['transflow', 'llmflow'], default='transflow')
parser.add_argument('--samples', type=int, default=1)
parser.add_argument('--steps', type=int, default=10)
parser.add_argument('--outpdb', type=str, default='./outpdb/default')
parser.add_argument('--weights', type=str, default=None)
parser.add_argument('--ckpt', type=str, default=None)
parser.add_argument('--original_weights', action='store_true')
parser.add_argument('--pdb_id', nargs='*', default=[])
parser.add_argument('--subsample', type=int, default=None)
parser.add_argument('--resample', action='store_true')
parser.add_argument('--tmax', type=float, default=1.0)
parser.add_argument('--no_flow', action='store_true', default=False)
parser.add_argument('--self_cond', action='store_true', default=False)
parser.add_argument('--noisy_first', action='store_true', default=False)
parser.add_argument('--runtime_json', type=str, default=None)
parser.add_argument('--no_overwrite', action='store_true', default=False)
parser.add_argument('--flow_steps', type=int, default=1)
parser.add_argument('--sample_len', type=int, default=100)
args = parser.parse_args()

import torch, tqdm, os, wandb, json, time
import pandas as pd
import pytorch_lightning as pl
import numpy as np
from collections import defaultdict
from flowprogen.data.data_modules import collate_fn
from flowprogen.model.wrapper import TransFlowWrapper, LLMFlowWrapper
from flowprogen.utils.tensor_utils import tensor_tree_map
import flowprogen.utils.protein as protein
from collections import defaultdict
from openfold.utils.import_weights import import_jax_weights_
from flowprogen.config import model_config
from flowprogen.model.esmfold import ESMFold

import torch._dynamo
torch._dynamo.config.suppress_errors = True

from flowprogen.utils.logging import get_logger
logger = get_logger(__name__)
torch.set_float32_matmul_precision("high")

config = model_config(
    'initial_training',
    train=True, 
    low_prec=True
) 
schedule = np.linspace(args.tmax, 0, args.steps+1)
if args.tmax != 1.0:
    schedule = np.array([1.0] + list(schedule))
loss_cfg = config.loss
data_cfg = config.data
data_cfg.common.use_templates = False
data_cfg.common.max_recycling_iters = 0

torch.serialization.add_safe_globals([
    config_dict.ConfigDict,
    config_dict.FieldReference
])
@torch.no_grad()
def main():

    # valset[0]
    logger.info("Loading the model")
    model_class = {'transflow': TransFlowWrapper, 'llmflow': LLMFlowWrapper}[args.mode]
    with torch.serialization.safe_globals([
        config_dict.ConfigDict,
        config_dict.FieldReference
    ]):
        model = model_class.load_from_checkpoint(args.ckpt, map_location='cpu', weights_only=False)
    model.load_ema_weights()
    model = model.cuda()
    model.eval()
    
    logger.info("Model has been loaded")
    
    results = defaultdict(list)
    os.makedirs(args.outpdb, exist_ok=True)
    runtime = defaultdict(list)
    decode_latent = DecodeLatent()
    result = []
    batch = None
    for j in tqdm.trange(args.samples):
        latents = model.inference(batch, fixed_modality_shape=(args.sample_len,),  modality_steps=args.flow_steps, as_protein=False, no_flow=False)
        print(latents.shape)
        seq_strs, pdb_strs = decode_latent.run(latents)
    # print(seq_strs,pdb_strs)
    for i in range(len(pdb_strs)):
        with open(f'test_{i}.pdb', 'w') as f:
            f.write(pdb_strs[i])

    if args.runtime_json:
        with open(args.runtime_json, 'w') as f:
            f.write(json.dumps(dict(runtime)))
if __name__ == "__main__":
    main()
