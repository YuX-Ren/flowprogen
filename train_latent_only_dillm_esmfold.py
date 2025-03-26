from shutil import rmtree
from pathlib import Path
import os
import gc
import torch
import random
import numpy as np
import wandb
import pandas as pd

import typing as T
from torch import nn, tensor, Tensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F

from einops import rearrange

from llmflow import LLMFlow, print_modality_sample

# hf related
from datasets import load_dataset
from diffusers.models import AutoencoderKL

from Bio import SeqIO
from Bio.PDB import *
from Bio.PDB import PDBParser, PDBIO, Polypeptide
from Bio.PDB.StructureBuilder import StructureBuilder
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

import sys; sys.path.append('.')
torch.set_float32_matmul_precision("high")
from llmflow.config import model_config
from llmflow.data.data_modules import OpenFoldSingleDataset, OpenFoldBatchCollator, OpenFoldDataset
from llmflow.model.wrapper import LLMFlowWrapper, TransFlowWrapper
from llmflow.utils.parsing import parse_train_args
args = parse_train_args()
from llmflow.utils.logging import get_logger
logger = get_logger(__name__)

from openfold.utils.exponential_moving_average import ExponentialMovingAverage
from openfold.utils.feats import atom14_to_atom37, pseudo_beta_fn
from openfold.utils.checkpointing import checkpoint_blocks
from openfold.np import residue_constants

RUN_NAME = "train_latent_only_dillm_esmfold"

is_main_process = pl.utilities.rank_zero_only.rank == 0

if is_main_process:
    wandb.init(
        project="dillm", 
        name=RUN_NAME,
    )
    rmtree(f'./results_dillm/{RUN_NAME}', ignore_errors=True)
    results_folder = Path(f'./results_dillm/{RUN_NAME}')
    results_folder.mkdir(exist_ok=True, parents=True)

# device = torch.device('cuda:0')

# 内存管理函数
def clear_memory():
    """清理显存和内存"""
    gc.collect()
    torch.cuda.empty_cache()
    
    # 尝试释放更多显存
    if torch.cuda.is_available():
        # 打印当前显存使用情况
        print(f"GPU memory before clearing: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        # 遍历所有CUDA张量并尝试释放
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) and obj.is_cuda:
                    del obj
            except:
                pass
        
        # 再次清理
        gc.collect()
        torch.cuda.empty_cache()
        
        # 打印清理后的显存使用情况
        print(f"GPU memory after clearing: {torch.cuda.memory_allocated() / 1e9:.2f} GB")


def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
set_seed()

# constants
SAMPLE_EVERY = 100

# functions
def divisible_by(num, den):
    return (num % den) == 0

def save_protein_structure(filename, sequence, coordinates):
    """Save generated protein structure in PDB format with full backbone"""
    
    sb = StructureBuilder()
    sb.init_structure("prot")
    sb.init_model(0)
    sb.init_chain("A")
    
    three_letter_codes = {
        'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU',
        'F': 'PHE', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
        'K': 'LYS', 'L': 'LEU', 'M': 'MET', 'N': 'ASN',
        'P': 'PRO', 'Q': 'GLN', 'R': 'ARG', 'S': 'SER',
        'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR',
        'X': 'UNK'  # 未知氨基酸
    }
    
    # 标准骨架原子的位置（相对于CA的位置）
    # backbone_offsets = {
    #     'N': np.array([-1.458, 0.0, 0.0]),   # 典型N-CA距离 ~1.46Å
    #     'C': np.array([1.524, 0.0, 0.0]),    # 典型CA-C距离 ~1.52Å
    #     'O': np.array([2.4, 1.0, 0.0])       # C=O键 ~1.24Å，并稍微偏离
    # }
    
    # 遍历每个氨基酸
    for i, (aa, coord) in enumerate(zip(sequence, coordinates)):
        if aa == 'X':  # 跳过未知残基
            continue
            
        # 确保使用有效的氨基酸代码
        three_letter = three_letter_codes.get(aa, "UNK")
        
        # 设置残基ID
        sb.init_seg(' ')
        sb.init_residue(three_letter, " ", i+1, " ")
        
        # 将张量转换为numpy数组
        if isinstance(coord, torch.Tensor):
            coord = coord.detach().cpu().numpy()
        
        # 从12维坐标向量中提取各个原子的坐标
        # 顺序是 [CA_x, CA_y, CA_z, N_x, N_y, N_z, C_x, C_y, C_z, O_x, O_y, O_z]
        ca_coord = coord[0:3]
        n_coord = coord[3:6]
        c_coord = coord[6:9]
        o_coord = coord[9:12]
        
        sb.init_atom("CA", ca_coord, 0.0, 1.0, " ", "CA", element="C")
        sb.init_atom("N", n_coord, 0.0, 1.0, " ", "N", element="N")
        sb.init_atom("C", c_coord, 0.0, 1.0, " ", "C", element="C")
        sb.init_atom("O", o_coord, 0.0, 1.0, " ", "O", element="O")
    structure = sb.get_structure()
    
    io = PDBIO()
    io.set_structure(structure)
    io.save(filename)

def cycle(iter_dl):
    while True:
        for batch in iter_dl:
            yield batch

config = model_config(
    'initial_training',
    train=True, 
    low_prec=True
) 
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
    data_dir='/share/project/xiaohongwang/Datasets/pdb_mmcif_data_npz',
    alignment_dir='/share/project/xiaohongwang/Datasets/openfold/pdb',
    pdb_chains=pd.read_csv('./pdb_mmcif_msa.csv', index_col='name'),
    config=config.data
)
trainset = OpenFoldDataset([trainset], [1.0], epoch_len=args.train_epoch_len)

train_loader = DataLoader(
    trainset,
    batch_size=args.batch_size,
    collate_fn=OpenFoldBatchCollator(),
    num_workers=args.num_workers,
    shuffle=not args.filter_chains,
)

iter_dl = cycle(train_loader)

model_cfg = config.model
trunk_cfg = config.model.trunk

model = LLMFlowWrapper(config, args)
# ProtEncoder = model.evoformer_stack
# ProtDecoder = model.structure_module

val_loader = None
trainer.fit(model, train_loader, val_loader, ckpt_path=args.ckpt)


ema_model = model.create_ema(0.9)


# 确保所有参数都是float32
# print("Converting all parameters to float32 to avoid precision issues...")
# for param in model.parameters():
#     param.data = param.data.to(torch.float32)

optimizer = AdamW(model.parameters(), lr = 5e-4)
for param_group in optimizer.param_groups:
    for param in param_group['params']:
        param.data = param.data.to(torch.float32)

# scaler = GradScaler()
# train loop
scheduler = CosineAnnealingLR(optimizer, T_max=100_000, eta_min=1e-6)


# # clear_memory()


model.train()
for step in range(1, 10_000 + 1):
    

    if step % 100 == 0:
        clear_memory()
    

    grad_accum_steps = min(4, 2 + step // 1000)
    

    loss_computed = False
    
    for _ in range(grad_accum_steps):
        # try:
        # Get batch
        batch = next(iter_dl)
        print('batch:', batch)        
        '''
        batch: ['aatype', 'between_segment_residues', 'domain_name', 
               'residue_index', 'seq_length', 'sequence', 'all_atom_positions', 
               'all_atom_mask', 'resolution', 'release_date', 'is_distillation']
        '''

        if coords_tensor.dtype != torch.float32:
            coords_tensor = coords_tensor.to(torch.float32)

        result = model.forward_modality(
            modalities=batch,
            times=None,
            modality_type=0,
            return_loss=True,
            return_loss_breakdown=True
        )
        

        if isinstance(result, tuple) and len(result) == 2:
            total_loss, (flow_loss, velocity_loss, recon_loss) = result
        else:

            total_loss = result
            flow_loss = total_loss * 0.9  # 假设重建损失约占总损失的90%
            recon_loss = total_loss * 0.1     # 假设KL损失约占总损失的10%
            print("Warning: forward_seq_coord didn't return loss breakdown, using approximations.")
        
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            print(f"Warning: Loss is {total_loss.item()}, skipping this batch")
            optimizer.zero_grad()
            continue
            

        if total_loss.item() > 1e5:
            print(f"Warning: Loss is very large: {total_loss.item()}, scaling down")
            total_loss = torch.log1p(total_loss)  # 使用log(1+x)缩放大损失
            

        if isinstance(recon_loss, torch.Tensor) and recon_loss.item() > 1e4:
            print(f"KL loss too large: {recon_loss.item()}, clamping")
            recon_loss_scale = min(1.0, 1e4 / recon_loss.item())
            recon_loss = recon_loss * recon_loss_scale

            if isinstance(result, tuple):
                total_loss = flow_loss + velocity_loss + recon_loss
        
        # Calculate losses
        total_loss = total_loss / grad_accum_steps
        
        # 直接反向传播，不使用scaler
        total_loss.backward()
        
        # 立即进行梯度裁剪，防止梯度爆炸
        clip_grad_norm_(model.parameters(), 1.0)
        
        # 检查梯度是否有NaN
        has_nan_grad = False
        for param in model.parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    has_nan_grad = True
                    break
        
        if has_nan_grad:
            print("Warning: NaN in gradients detected, skipping this batch")
            optimizer.zero_grad()
            continue
        
        # 标记成功计算了损失
        loss_computed = True
        
        # 成功计算损失后跳出循环
        break
            
        # except Exception as e:
        #     print(f"Error during forward/backward pass: {e}")
        #     optimizer.zero_grad()  # 确保梯度被清零
        #     continue
    
    # 只有成功计算了损失才执行优化器更新
    if loss_computed:
        try:
            # 梯度裁剪和优化器更新，不使用scaler
            clip_grad_norm_(model.parameters(), 0.5)
            
            # 检查参数更新前的范数，如果过大则进一步约束
            param_norm = 0.0
            for param in model.parameters():
                if param.grad is not None:
                    param_norm += param.grad.data.norm(2).item() ** 2
            param_norm = param_norm ** 0.5
            
            if param_norm > 10.0:
                print(f"Warning: Gradient norm is large: {param_norm}, scaling down")
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.data.mul_(10.0 / param_norm)
            
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            ema_model.update()
            
            # 更新日志记录
            log_dict = {
                "total_loss": total_loss.item(),
                "flow_loss": flow_loss.item(),
                "velocity_loss": velocity_loss.item(),
                "recon_loss": recon_loss.item(),
                "grad_accum_steps": grad_accum_steps,
                "lr": scheduler.get_last_lr()[0]
            }
            

            if isinstance(recon_loss, torch.Tensor) and not torch.isnan(recon_loss) and not torch.isinf(recon_loss):
                log_dict["recon_loss"] = recon_loss.item()
            if isinstance(flow_loss, torch.Tensor) and not torch.isnan(flow_loss) and not torch.isinf(flow_loss):
                log_dict["flow_loss"] = flow_loss.item()
            if isinstance(velocity_loss, torch.Tensor) and not torch.isnan(velocity_loss) and not torch.isinf(velocity_loss):
                log_dict["velocity_loss"] = velocity_loss.item()
            
            print(f'{step}: total_loss={total_loss.item():.3f}, flow_loss={flow_loss.item() if isinstance(flow_loss, torch.Tensor) else "N/A":.3f}, velocity_loss={velocity_loss.item() if isinstance(velocity_loss, torch.Tensor) else "N/A":.3f}, recon_loss={recon_loss.item() if isinstance(recon_loss, torch.Tensor) else "N/A":.3f}, grad_accum_steps={grad_accum_steps}')
                
            wandb.log(log_dict, step=step)
            
        except Exception as e:
            print(f"Error during optimizer step: {e}")
            optimizer.zero_grad()  # 确保梯度被清零
    else:
        print(f"Step {step}: Failed to compute loss, skipping optimizer update")
    
    # 只有在成功计算了损失的情况下才执行采样
    if loss_computed and divisible_by(step, SAMPLE_EVERY):
        model.eval()
        try:
            with torch.no_grad():
                # 清理内存前将模型转为float32
                ema_model.to(torch.float32)
                
                # 减少生成的batch_size
                # z = torch.randn(1, 512, 32).to(device)  # [batch=1, length, latent_dim]
                
                structure = ema_model.generate_modality_only(
                    batch_size=1,
                    modality_type=0,
                    modality_steps=100,
                    return_unprocessed_modalities=True
                )
                print('structure', structure)
                # Convert to amino acid sequence
                # seq_pred = torch.argmax(seq_logits, dim=-1)
                # # print(seq_pred.shape)
                
                # # Use the idx_to_aa mapping to convert indices to amino acids
                # sequences = []
                # for seq in seq_pred:
                #     aa_sequence = []
                #     for i in seq:
                #         idx = i.item()
                #         aa = dataset.idx_to_aa.get(idx, 'X')  # Default to 'X' if index not found
                #         aa_sequence.append(aa)
                #     sequences.append(''.join(aa_sequence))
                
                # # Save generated sequences and coordinates
                # for i, (seq, coord) in enumerate(zip(sequences, coords_tensor)):
                #     filename = str(results_folder / f'{step}_sample_{i}.pdb')
                #     save_protein_structure(filename, seq, coord)
                #     print(f"Generated sequence {i}: {seq[:100]}...")
                
                # clear_memory()
        except Exception as e:
            print(f"Error during sampling: {e}")
            # clear_memory()
    
    # 每50步保存一次模型以便断点续训
    # if divisible_by(step, 50):
    #     try:
    #         # 保存模型
    #         torch.save({
    #             'step': step,
    #             'model_state_dict': model.state_dict(),
    #             'optimizer_state_dict': optimizer.state_dict(),
    #             'scheduler_state_dict': scheduler.state_dict(),
    #             'ema_model_state_dict': ema_model.state_dict()
    #         }, str(results_folder / f'checkpoint_{step}.pt'))
            
    #         # 仅保留最新的3个检查点
    #         checkpoints = sorted(list(results_folder.glob('checkpoint_*.pt')))
    #         if len(checkpoints) > 3:
    #             for old_ckpt in checkpoints[:-3]:
    #                 old_ckpt.unlink()
    #     except Exception as e:
    #         print(f"Error saving checkpoint: {e}")

print("Training complete!") 