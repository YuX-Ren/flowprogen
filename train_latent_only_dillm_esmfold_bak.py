from shutil import rmtree
from pathlib import Path
import os
import gc
import torch
import random
import numpy as np
import wandb

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

import sys; sys.path.append('.')
torch.set_float32_matmul_precision("high")
from llmflow.model.esmfold import ESMFold
from llmflow.model.trunk import FoldingTrunk
from llmflow.config import model_config
from llmflow.data.data_modules import OpenFoldSingleDataset, OpenFoldBatchCollator, OpenFoldDataset

from openfold.utils.exponential_moving_average import ExponentialMovingAverage
from openfold.utils.feats import atom14_to_atom37, pseudo_beta_fn
from openfold.utils.checkpointing import checkpoint_blocks
from openfold.np import residue_constants

RUN_NAME = "train_latent_only_dillm_esmfold"
wandb.init(
    project="dillm", 
    name=RUN_NAME,
)
device = torch.device('cuda:0')
rmtree(f'./results_llmflow/{RUN_NAME}', ignore_errors = True)
results_folder = Path(f'./results_llmflow/{RUN_NAME}')
results_folder.mkdir(exist_ok = True, parents = True)


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

class ProteinEncoder(ESMFold):
    def __init__(self, cfg):
        super(ProteinEncoder, self).__init__(cfg)
        
        self.mytrunk = myFoldingTrunk(cfg.trunk)
        
    def forward(
        self,
        batch,
        prev_outputs=None,
    ):
        """Runs a forward pass given input tokens. Use `model.infer` to
        run inference from a sequence.

        Args:
            aa (torch.Tensor): Tensor containing indices corresponding to amino acids. Indices match
                openfold.np.residue_constants.restype_order_with_x.
            mask (torch.Tensor): Binary tensor with 1 meaning position is unmasked and 0 meaning position is masked.
            residx (torch.Tensor): Residue indices of amino acids. Will assume contiguous if not provided.
            masking_pattern (torch.Tensor): Optional masking to pass to the input. Binary tensor of the same size
                as `aa`. Positions with 1 will be masked. ESMFold sometimes produces different samples when
                different masks are provided.
            num_recycles (int): How many recycle iterations to perform. If None, defaults to training max
                recycles, which is 3.
        """

        aa = batch['aatype']
        mask = batch['seq_mask']
        residx = batch['residue_index']
       
        # === ESM ===
        
        esmaa = self._af2_idx_to_esm_idx(aa, mask)
        print('esmaa', esmaa.shape)
        esm_s, _ = self._compute_language_model_representations(esmaa)
        print('esm_s', esm_s.shape)
        # Convert esm_s to the precision used by the trunk and
        # the structure module. These tensors may be a lower precision if, for example,
        # we're running the language model in fp16 precision.
        esm_s = esm_s.to(self.esm_s_combine.dtype)
        esm_s = esm_s.detach()

        # === preprocessing ===
        esm_s = (self.esm_s_combine.softmax(0).unsqueeze(0) @ esm_s).squeeze(2)
        s_s_0 = self.esm_s_mlp(esm_s)
        s_s_0 += self.embedding(aa)
        #######################
        if 'noised_pseudo_beta_dists' in batch:
            inp_z = self._get_input_pair_embeddings(
                batch['noised_pseudo_beta_dists'], 
                batch['pseudo_beta_mask']
            )
            inp_z = inp_z + self.input_time_embedding(self.input_time_projection(batch['t']))[:,None,None]
        else: # have to run the module, else DDP wont work
            B, L = batch['aatype'].shape
            inp_z = self._get_input_pair_embeddings(
                s_s_0.new_zeros(B, L, L), 
                batch['pseudo_beta_mask'] * 0.0
            )
            inp_z = inp_z + self.input_time_embedding(self.input_time_projection(inp_z.new_zeros(B)))[:,None,None]
        ##########################
        #############################
        if self.extra_input:
            if 'extra_all_atom_positions' in batch:
                extra_pseudo_beta = pseudo_beta_fn(batch['aatype'], batch['extra_all_atom_positions'], None)
                extra_pseudo_beta_dists = torch.sum((extra_pseudo_beta.unsqueeze(-2) - extra_pseudo_beta.unsqueeze(-3)) ** 2, dim=-1)**0.5
                extra_inp_z = self._get_extra_input_pair_embeddings(
                    extra_pseudo_beta_dists, 
                    batch['pseudo_beta_mask'],
                )
                
            else: # otherwise DDP complains
                B, L = batch['aatype'].shape
                extra_inp_z = self._get_extra_input_pair_embeddings(
                    inp_z.new_zeros(B, L, L), 
                    inp_z.new_zeros(B, L),
                ) * 0.0
    
            inp_z = inp_z + extra_inp_z
        ########################

        
        s_z_0 = inp_z 
        if prev_outputs is not None:
            s_s_0 = s_s_0 + self.trunk.recycle_s_norm(prev_outputs['s_s'])
            s_z_0 = s_z_0 + self.trunk.recycle_z_norm(prev_outputs['s_z'])
            s_z_0 = s_z_0 + self.trunk.recycle_disto(FoldingTrunk.distogram(
                prev_outputs['sm']["positions"][-1][:, :, :3],
                3.375,
                21.375,
                self.trunk.recycle_bins,
            ))

        else:
            s_s_0 = s_s_0 + self.trunk.recycle_s_norm(torch.zeros_like(s_s_0)) * 0.0
            s_z_0 = s_z_0 + self.trunk.recycle_z_norm(torch.zeros_like(s_z_0)) * 0.0
            s_z_0 = s_z_0 + self.trunk.recycle_disto(s_z_0.new_zeros(s_z_0.shape[:-2], dtype=torch.long)) * 0.0
        
        s_s, s_z = self.mytrunk(s_s_0, s_z_0, aa, residx, mask, no_recycles=3)
        return s_s, s_z
    
class myFoldingTrunk(FoldingTrunk):
    def __init__(self, cfg):
        super(myFoldingTrunk, self).__init__(cfg)
        
    def forward(self, seq_feats, pair_feats, true_aa, residx, mask, no_recycles: T.Optional[int] = None):
        """
        Inputs:
          seq_feats:     B x L x C            tensor of sequence features
          pair_feats:    B x L x L x C        tensor of pair features
          residx:        B x L                long tensor giving the position in the sequence
          mask:          B x L                boolean tensor indicating valid residues

        Output:
          predicted_structure: B x L x (num_atoms_per_residue * 3) tensor wrapped in a Coordinates object
        """

        device = seq_feats.device
        s_s_0 = seq_feats
        s_z_0 = pair_feats

        
        def trunk_iter(s, z, residx, mask):
            z = z + self.pairwise_positional_embedding(residx, mask=mask)
            blocks = self._prep_blocks(mask=mask, residue_index=residx, chunk_size=self.chunk_size)

            s, z = checkpoint_blocks(
                blocks,
                args=(s, z),
                blocks_per_ckpt=1,
            )
            return s, z


        s_s, s_z = trunk_iter(s_s_0, s_z_0, residx, mask)
        return s_s, s_z

class ProteinDecoder(FoldingTrunk):
    def __init__(self, cfg):
        super(ProteinDecoder, self).__init__(cfg)

    def forward(self, s_s, s_z, batch):
        # === Structure module ===
        structure = {}
        true_aa = batch['aatype']
        mask = batch['seq_mask']
        structure["sm"] = self.structure_module(
            {"single": self.trunk2sm_s(s_s), "pair": self.trunk2sm_z(s_z)},
            true_aa,
            mask.float(),
        )
        
        assert isinstance(structure, dict)  # type: ignore
        structure["s_s"] = s_s
        structure["s_z"] = s_z

        return structure


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
print('config:', config.keys())
# dataset = ProteinDataset("/share/project/linwenjun/swissprot_pdb_v4", max_length = 256)
dataset = OpenFoldSingleDataset(data_dir='/share/project/xiaohongwang/Datasets/pdb_mmcif_data_npz',
                                alignment_dir='/share/project/xiaohongwang/Datasets/openfold/pdb',
                                pdb_chains=pd.read_csv('./pdb_mmcif_msa.csv',
                                                       index_col='name'),
                                config=config.data
                                )


# Use custom collate function instead of model.create_dataloader
# dataloader = model.create_dataloader(dataset, batch_size = 2, shuffle = True)
dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=True,
    # collate_fn=collate_protein_batch,
    collate_fn=OpenFoldBatchCollator(),
    num_workers=4,
    pin_memory=True,
    persistent_workers=True
)

iter_dl = cycle(dataloader)

model_cfg = config.model
trunk_cfg = config.model.trunk
model = LLMFlow(
    num_text_tokens = 21,  # Number of amino acids
    dim_latent = 21,  # Latent dimension for protein representation
    channel_first_latent = False,  # Protein data is not channel-first
    modality_default_shape = (512,),  # Maximum sequence length
    modality_encoder = ProteinEncoder(cfg=model_cfg),
    modality_decoder = ProteinDecoder(cfg=trunk_cfg),
    pre_post_transformer_enc_dec = (
        nn.Linear(21, 2048),  # Adapt latent dimension to transformer dimension
        nn.Linear(2048, 21),
    ),
    add_pos_emb = True,  # Important for sequence data
    modality_num_dim = 1,  # 1D sequence data
    fallback_to_default_shape_if_invalid = True,
    reconstruction_loss_weight = 1.0,
    transformer={
        'use_llama': True,
        'dim': 2048,
        'model_name_or_path': '/share/project/xiaohongwang/LLM_checkpoints/Llama3.2/Llama-3.2-1B-Instruct',
        'use_gradient_checkpointing': True
    },
).to(device)

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
        
        # 检查损失是否有效（不是NaN或无穷大）
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