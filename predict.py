import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['transflow', 'llmflow'], default='transflow')
parser.add_argument('--samples', type=int, default=10)
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


@torch.no_grad()
def main():

    valset = {
        # 'transflow': AlphaFoldCSVDataset,
        'transflow': CSVDataset,
    }[args.mode](
        data_cfg,
        args.input_csv,
        msa_dir=args.msa_dir,
        templates_dir=args.templates_dir,
    )
    # valset[0]
    logger.info("Loading the model")
    model_class = {'transflow': TransFlowWrapper, 'llmflow': LLMFlowWrapper}[args.mode]

    if args.weights:
        ckpt = torch.load(args.weights, map_location='cpu',weights_only=False)
        model = model_class(**ckpt['hyper_parameters'])
        # Add 'model.' prefix to keys in state_dict
        state_dict = {f"model.{k}" if not k.startswith('model.') else k: v for k, v in ckpt['state_dict'].items()}
        model.load_state_dict(state_dict, strict=False)
        model = model.cuda()
    
    elif args.original_weights:
        model = ESMFold(config.model, extra_input=None)
        path = "/share/project/xiaohongwang/Routine_ckpts/esm_pretrained_models/esmfold_3B_v1.pt"
        model_data = torch.load(path, map_location='cpu')
        model_state = model_data["model"]
        model.model.load_state_dict(model_state, strict=False)
        model = model.to(torch.float).cuda()
        
    else:
        model = model_class.load_from_checkpoint(args.ckpt, map_location='cpu')
        model.load_ema_weights()
        model = model.cuda()
    model.eval()
    
    logger.info("Model has been loaded")
    
    results = defaultdict(list)
    os.makedirs(args.outpdb, exist_ok=True)
    runtime = defaultdict(list)
    for i, item in enumerate(valset):
        if args.pdb_id and item['name'] not in args.pdb_id:
            continue
        if args.no_overwrite and os.path.exists(f'{args.outpdb}/{item["name"]}.pdb'):
            continue
        result = []
        for j in tqdm.trange(args.samples):
            if args.subsample or args.resample:
                item = valset[i] # resample MSA
            
            batch = collate_fn([item])
            batch = tensor_tree_map(lambda x: x.cuda(), batch)  
            start = time.time()
            prots = model.inference(batch, fixed_modality_shape=(len(item['seqres']), len(item['seqres'])),  modality_steps=args.flow_steps, as_protein=True, no_flow=args.no_flow)
            # prots = model.inference(batch, as_protein=True, noisy_first=args.noisy_first,
            #             no_diffusion=args.no_diffusion, schedule=schedule, self_cond=args.self_cond)
            runtime[item['name']].append(time.time() - start)
            result.append(prots[-1])
            


        with open(f'{args.outpdb}/{item["name"]}.pdb', 'w') as f:
            f.write(protein.prots_to_pdb(result))

    if args.runtime_json:
        with open(args.runtime_json, 'w') as f:
            f.write(json.dumps(dict(runtime)))
if __name__ == "__main__":
    main()
