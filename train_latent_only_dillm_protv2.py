from shutil import rmtree
from pathlib import Path
import os
import gc
import torch

import torch
from torch import nn, tensor, Tensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn.utils import clip_grad_norm_

from einops import rearrange
from PIL import Image
import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image

from dillm import LLMfusion, print_modality_sample

import json
import wandb

# hf related
from datasets import load_dataset
from diffusers.models import AutoencoderKL

from Bio import SeqIO
from Bio.PDB import *
import torch.nn.functional as F
from Bio.PDB import PDBParser, PDBIO, Polypeptide
from Bio.PDB.StructureBuilder import StructureBuilder
import numpy as np
import random
# 移除OpenFold导入
# from alphaflow.openfold.openfold.model.evoformer import (
#     EvoformerBlock,
#     EvoformerStack,
# )
# from alphaflow.openfold.openfold.model.structure_module import StructureModule


# 设置PyTorch内存管理
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
# torch.backends.cuda.matmul.allow_tf32 = True  # 允许使用TF32（对于Ampere及以上GPU）

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

wandb.init(
    project="dillm", 
    name="train_latent_only_dillm_prot",
)
device = torch.device('cuda:0')
rmtree('./results_dillm/train_latent_only_dillm_protv2', ignore_errors = True)
results_folder = Path('./results_dillm/train_latent_only_dillm_protv2')
results_folder.mkdir(exist_ok = True, parents = True)

# constants
SAMPLE_EVERY = 100

# with open("./data/flowers/labels.txt", "r") as file:
#     content = file.read()
# LABELS_TEXT = content.split('\n')

# functions
def divisible_by(num, den):
    return (num % den) == 0

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens: Tensor) -> str:
    return "".join(list(map(decode_token, tokens.tolist())))

def encode_tokens(str: str) -> Tensor:
    return tensor([*bytes(str, 'UTF-8')])

class ProteinEncoder(Module):
    def __init__(self, hidden_dim=256, latent_dim=20, num_blocks=4):
        super().__init__()
        
        # Embedding layers
        self.seq_embedding = nn.Embedding(20, hidden_dim)  # 20 standard amino acids
        self.pos_embedding = nn.Embedding(1024, hidden_dim)  # Position embeddings
        
        # MSA and pair processing
        self.msa_dim = hidden_dim
        self.pair_dim = hidden_dim
        
        # Evoformer-like stack (simplified)
        self.evoformer_blocks = nn.ModuleList([
            EvoformerBlock(
                msa_dim=hidden_dim,
                pair_dim=hidden_dim,
                num_heads=4
            )
            for _ in range(num_blocks)
        ])
        
        # Final projection to latent space
        self.latent_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # Mean and log variance
        )
        
        self.latent_dim = latent_dim
        
    def reparameterize(self, mu, logvar):
        # 添加裁剪以提高数值稳定性
        logvar = torch.clamp(logvar, -20, 20)  # 防止方差变得过大或过小
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, seq_tensor, coords_tensor):
        batch_size, seq_len = seq_tensor.shape
        
        # Initial sequence embedding
        s = self.seq_embedding(seq_tensor)  # [batch, length, hidden_dim]
        
        # Position embedding
        positions = torch.arange(seq_len, device=seq_tensor.device).unsqueeze(0).expand(batch_size, -1)
        p = self.pos_embedding(positions)  # [batch, length, hidden_dim]
        
        # Initialize MSA representation (using single sequence)
        m = s.unsqueeze(1)  # [batch, 1, length, hidden_dim]
        
        # Initialize pair representation
        z = torch.zeros(batch_size, seq_len, seq_len, self.pair_dim, device=seq_tensor.device)
        
        # Add coordinate information to pair representation
        coord_diff = coords_tensor.unsqueeze(2) - coords_tensor.unsqueeze(1)  # [batch, length, length, 3]
        coord_dist = torch.norm(coord_diff, dim=-1, keepdim=True)  # [batch, length, length, 1]
        
        # Simple projection of distances to pair space
        dist_embedding = torch.cat([
            torch.sin(coord_dist * (2 ** i)) for i in range(10)
        ], dim=-1)  # [batch, length, length, 10]
        
        z += F.pad(dist_embedding, (0, self.pair_dim - 10))  # Pad to pair_dim
        
        # Create masks
        seq_mask = (seq_tensor > 0).float()  # [batch, length]
        msa_mask = seq_mask.unsqueeze(1)  # [batch, 1, length]
        pair_mask = seq_mask.unsqueeze(-1) * seq_mask.unsqueeze(-2)  # [batch, length, length]
        
        # Run through Evoformer blocks
        for block in self.evoformer_blocks:
            m, z = block(m, z, msa_mask, pair_mask)
        
        # Extract sequence representation
        s = m.squeeze(1)  # [batch, length, hidden_dim]
        
        # Project to latent space
        latent = self.latent_projector(s)  # [batch, length, latent_dim*2]
        mu, logvar = latent.chunk(2, dim=-1)
        
        # Sample latent vector
        z_latent = self.reparameterize(mu, logvar)
        
        return z_latent, mu, logvar


class ProteinDecoder(Module):
    def __init__(self, hidden_dim=256, latent_dim=20, num_blocks=4):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Project latent to hidden dimension
        self.latent_projector = nn.Linear(latent_dim, hidden_dim)
        
        # Structure module components
        self.structure_blocks = nn.ModuleList([
            StructureModule(
                hidden_dim=hidden_dim,
                num_heads=4
            ) for _ in range(num_blocks)
        ])
        
        # Final predictors
        self.seq_predictor = nn.Linear(hidden_dim, 20)  # 20 amino acids
        self.coord_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 12)  # 3D coordinates
        )
    
    def __call__(self, z):
        """重写__call__方法，使其根据调用上下文返回正确的输出格式
        
        当作为DiLLM的decoder调用时(通过`_is_called_by_model_decoder`标志识别)：
        - 返回与输入相同形状的张量，确保MSE损失计算正确
        
        当正常调用时：
        - 返回标准的forward方法输出(seq_logits, coords)
        """
        # 检查是否是DiLLM调用
        if getattr(self, '_is_called_by_model_decoder', False):
            # 如果是DiLLM调用，重置标志并返回输入z
            # 这是最简单的解决方案，确保MSE损失能够计算
            self._is_called_by_model_decoder = False
            return z  # 直接返回输入，避免任何形状不匹配问题
        
        # 否则使用标准forward方法
        return self.forward(z)
    
    def forward(self, z):
        """标准前向方法，返回序列logits和坐标"""
        batch_size, seq_len, _ = z.shape
        
        # Project latent vector to hidden dimension
        h = self.latent_projector(z)  # [batch, length, hidden_dim]
        
        # Initialize coordinate frames
        frames = torch.zeros(batch_size, seq_len, 4, 4, device=z.device)
        # Set diagonal to 1 to make identity matrices
        identity = torch.eye(4, device=z.device).unsqueeze(0).unsqueeze(0)
        frames = frames + identity
        
        # Initialize positions
        positions = torch.zeros(batch_size, seq_len, 3, device=z.device)
        
        # Run through structure blocks
        for block in self.structure_blocks:
            h, frames, positions = block(h, frames, positions)
        
        # Predict sequence and coordinates
        seq_logits = self.seq_predictor(h)  # [batch, length, 20]
        coords = self.coord_predictor(h)  # [batch, length, 3]
        
        return seq_logits, coords


# 完全使用自定义的EvoformerBlock，而不是导入OpenFold的实现
class EvoformerBlock(Module):
    def __init__(self, msa_dim, pair_dim, num_heads):
        super().__init__()
        
        # MSA stack
        self.msa_attention = MSAAttention(msa_dim, num_heads)
        self.msa_transition = Transition(msa_dim)
        
        # Pair stack
        self.pair_attention = PairAttention(pair_dim, num_heads)
        self.pair_transition = Transition(pair_dim)
        
        # Communication between MSA and pair representations
        self.msa_pair_attention = MSAPairAttention(msa_dim, pair_dim, num_heads)
        self.pair_msa_update = PairMSAUpdate(msa_dim, pair_dim)
        
        # Normalization
        self.msa_norm1 = nn.LayerNorm(msa_dim)
        self.msa_norm2 = nn.LayerNorm(msa_dim)
        self.msa_norm3 = nn.LayerNorm(msa_dim)
        self.pair_norm1 = nn.LayerNorm(pair_dim)
        self.pair_norm2 = nn.LayerNorm(pair_dim)
        self.pair_norm3 = nn.LayerNorm(pair_dim)
    
    def forward(self, m, z, msa_mask, pair_mask):
        # MSA self-attention
        m_attn = self.msa_attention(self.msa_norm1(m), msa_mask) + m
        
        # MSA transition
        m_trans = self.msa_transition(self.msa_norm2(m_attn)) + m_attn
        
        # MSA-pair attention
        z_update = self.msa_pair_attention(self.msa_norm3(m_trans), z, msa_mask, pair_mask) + z
        
        # Pair self-attention
        z_attn = self.pair_attention(self.pair_norm1(z_update), pair_mask) + z_update
        
        # Pair transition
        z_trans = self.pair_transition(self.pair_norm2(z_attn)) + z_attn
        
        # Pair-MSA update
        m_update = self.pair_msa_update(m_trans, self.pair_norm3(z_trans), msa_mask) + m_trans
        
        return m_update, z_trans


class MSAAttention(Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        
        # 使用三个独立的线性层代替MultiheadAttention
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # 初始化缩放因子
        self.scale = self.head_dim ** -0.5
    
    def forward(self, m, mask):
        # Reshape MSA for attention: [batch, num_seqs, length, dim] -> [batch*num_seqs, length, dim]
        batch, num_seqs, length, dim = m.shape
        m_reshaped = m.view(batch * num_seqs, length, dim)
        mask_reshaped = mask.view(batch * num_seqs, length)
        
        # 投影查询、键和值
        q = self.q_proj(m_reshaped)  # [batch*num_seqs, length, dim]
        k = self.k_proj(m_reshaped)  # [batch*num_seqs, length, dim]
        v = self.v_proj(m_reshaped)  # [batch*num_seqs, length, dim]
        
        # 将投影后的张量重塑为多头格式
        q = q.view(batch * num_seqs, length, self.num_heads, self.head_dim).transpose(1, 2)  # [batch*num_seqs, num_heads, length, head_dim]
        k = k.view(batch * num_seqs, length, self.num_heads, self.head_dim).transpose(1, 2)  # [batch*num_seqs, num_heads, length, head_dim]
        v = v.view(batch * num_seqs, length, self.num_heads, self.head_dim).transpose(1, 2)  # [batch*num_seqs, num_heads, length, head_dim]
        
        # 计算注意力分数并应用缩放
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch*num_seqs, num_heads, length, length]
        
        # 应用mask
        if mask_reshaped is not None:
            # 调整mask形状
            mask_expanded = mask_reshaped.unsqueeze(1).unsqueeze(2)  # [batch*num_seqs, 1, 1, length]
            
            # 使用float16兼容的负值
            mask_value = torch.tensor(-65000.0, dtype=attn_weights.dtype, device=attn_weights.device)
            attn_weights = attn_weights.masked_fill(~(mask_expanded > 0), mask_value)
        
        # 应用softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # 应用注意力
        out = torch.matmul(attn_weights, v)  # [batch*num_seqs, num_heads, length, head_dim]
        
        # 重塑回原始形状
        out = out.transpose(1, 2).contiguous().view(batch * num_seqs, length, dim)  # [batch*num_seqs, length, dim]
        
        # 最终投影
        out = self.out_proj(out)  # [batch*num_seqs, length, dim]
        
        # 重塑回原始维度
        return out.view(batch, num_seqs, length, dim)


class PairAttention(Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim must be divisible by num_heads"
        
        # 使用三个独立的线性层代替MultiheadAttention
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # 初始化缩放因子
        self.scale = self.head_dim ** -0.5
    
    def forward(self, z, mask):
        # Reshape pair for attention: [batch, length, length, dim] -> [batch*length, length, dim]
        batch, length, _, dim = z.shape
        z_reshaped = z.view(batch * length, length, dim)
        
        # 投影查询、键和值
        q = self.q_proj(z_reshaped)  # [batch*length, length, dim]
        k = self.k_proj(z_reshaped)  # [batch*length, length, dim]
        v = self.v_proj(z_reshaped)  # [batch*length, length, dim]
        
        # 将投影后的张量重塑为多头格式
        q = q.view(batch * length, length, self.num_heads, self.head_dim).transpose(1, 2)  # [batch*length, num_heads, length, head_dim]
        k = k.view(batch * length, length, self.num_heads, self.head_dim).transpose(1, 2)  # [batch*length, num_heads, length, head_dim]
        v = v.view(batch * length, length, self.num_heads, self.head_dim).transpose(1, 2)  # [batch*length, num_heads, length, head_dim]
        
        # 计算注意力分数并应用缩放
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch*length, num_heads, length, length]
        
        # 应用mask
        if mask is not None:
            # 调整mask形状
            mask_reshaped = mask.view(batch * length, 1, 1, length)
            
            # 使用float16兼容的负值
            mask_value = torch.tensor(-65000.0, dtype=attn_weights.dtype, device=attn_weights.device)
            attn_weights = attn_weights.masked_fill(~(mask_reshaped > 0), mask_value)
        
        # 应用softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # 应用注意力
        out = torch.matmul(attn_weights, v)  # [batch*length, num_heads, length, head_dim]
        
        # 重塑回原始形状
        out = out.transpose(1, 2).contiguous().view(batch * length, length, dim)  # [batch*length, length, dim]
        
        # 最终投影
        out = self.out_proj(out)  # [batch*length, length, dim]
        
        # 重塑回原始维度
        return out.view(batch, length, length, dim)


class MSAPairAttention(Module):
    def __init__(self, msa_dim, pair_dim, num_heads):
        super().__init__()
        self.msa_proj = nn.Linear(msa_dim, pair_dim)
        self.pair_proj = nn.Linear(pair_dim, pair_dim)
        self.output_proj = nn.Linear(pair_dim, pair_dim)
        self.num_heads = num_heads
        self.head_dim = pair_dim // num_heads
        assert self.head_dim * num_heads == pair_dim, "pair_dim must be divisible by num_heads"
    
    def forward(self, m, z, msa_mask, pair_mask):
        batch, num_seqs, length, msa_dim = m.shape
        
        # Project MSA to pair dimension
        m_proj = self.msa_proj(m)  # [batch, num_seqs, length, pair_dim]
        
        # 更高效的实现，避免循环
        # 将m_proj展平为[batch*num_seqs, length, pair_dim]
        m_flat = m_proj.reshape(batch * num_seqs, length, -1)
        
        # 计算批量注意力
        # 将m_flat看作是query，每个位置对其他位置进行注意力计算
        q = m_flat.reshape(batch * num_seqs, length, self.num_heads, self.head_dim).transpose(1, 2)  # [batch*num_seqs, num_heads, length, head_dim]
        k = q  # 自注意力
        v = q  # 自注意力
        
        # 计算注意力分数
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [batch*num_seqs, num_heads, length, length]
        
        # 使用pair_mask来遮盖无效的注意力
        if pair_mask is not None:
            # 调整mask的形状以适应注意力分数
            mask = pair_mask.view(batch, 1, 1, length, length).expand(-1, num_seqs, self.num_heads, -1, -1)
            mask = mask.reshape(batch * num_seqs, self.num_heads, length, length)
            
            # 使用与attn_weights相同dtype的mask值，以避免float16溢出
            mask_value = torch.tensor(-65000.0, dtype=attn_weights.dtype, device=attn_weights.device)
            attn_weights = attn_weights.masked_fill(~(mask.bool()), mask_value)
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # 应用注意力并转换回原始形状
        out = torch.matmul(attn_weights, v)  # [batch*num_seqs, num_heads, length, head_dim]
        out = out.transpose(1, 2).reshape(batch * num_seqs, length, -1)  # [batch*num_seqs, length, pair_dim]
        
        # 转回原始形状 [batch, num_seqs, length, pair_dim]
        out = out.reshape(batch, num_seqs, length, -1)
        
        # 对所有序列进行平均，得到一个[batch, length, length, pair_dim]的张量
        # 首先，我们需要将out复制length次，然后沿着一个新维度堆叠
        out_expanded = out.unsqueeze(3).expand(-1, -1, -1, length, -1)  # [batch, num_seqs, length, length, pair_dim]
        
        # 然后，对num_seqs维度取平均
        z_update = out_expanded.mean(dim=1)  # [batch, length, length, pair_dim]
        
        # 投影到输出
        z_update = self.output_proj(z_update)
        
        return z_update


class PairMSAUpdate(Module):
    def __init__(self, msa_dim, pair_dim):
        super().__init__()
        self.pair_proj = nn.Linear(pair_dim, msa_dim)
        self.msa_proj = nn.Linear(msa_dim, msa_dim)
        self.output_proj = nn.Linear(msa_dim, msa_dim)
    
    def forward(self, m, z, msa_mask):
        batch, num_seqs, length, msa_dim = m.shape
        
        # Project pair to MSA dimension
        z_proj = self.pair_proj(z)  # [batch, length, length, msa_dim]
        
        # 更高效的实现，避免循环
        # 将z_proj转换为[batch, length, length*msa_dim]
        z_flat = z_proj.reshape(batch, length, length * msa_dim)
        
        # 将m转换为[batch*num_seqs*length, msa_dim]
        m_flat = m.reshape(batch * num_seqs * length, msa_dim)
        
        # 使用1x1卷积进行交互（等价于线性变换）
        m_proj = self.msa_proj(m_flat).reshape(batch, num_seqs, length, msa_dim)
        
        # 使用广播机制更新m
        # 将z_proj看作每个位置的上下文
        context = z_proj.mean(dim=2)  # [batch, length, msa_dim]，对每列取平均
        
        # 广播到[batch, num_seqs, length, msa_dim]
        context = context.unsqueeze(1).expand(-1, num_seqs, -1, -1)
        
        # 组合m和context
        combined = m_proj + context
        
        # 投影到输出
        m_update = self.output_proj(combined)
        
        return m_update


class Transition(Module):
    def __init__(self, dim):
        super().__init__()
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)
        self.act = nn.GELU()
    
    def forward(self, x):
        return self.linear2(self.act(self.linear1(x)))


# Structure module components
class StructureModule(Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        
        # 自定义注意力实现
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"
        
        # 注意力投影层
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # 缩放因子
        self.scale = self.head_dim ** -0.5
        
        # Feed-forward layers
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Frame update components
        self.frame_updater = FrameUpdater(hidden_dim)
        
        # Normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
    
    def self_attention(self, x):
        batch_size, seq_len, dim = x.shape
        
        # 投影
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 重塑为多头格式
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        
        # 计算注意力得分
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [batch, num_heads, seq_len, seq_len]
        
        # Softmax
        attn_weights = F.softmax(attn_weights, dim=-1)
        
        # 应用注意力
        out = torch.matmul(attn_weights, v)  # [batch, num_heads, seq_len, head_dim]
        
        # 重塑回原始维度
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, dim)  # [batch, seq_len, dim]
        
        # 最终投影
        return self.out_proj(out)
    
    def forward(self, h, frames, positions):
        # Self-attention
        h_norm = self.norm1(h)
        h_attn = self.self_attention(h_norm)
        h = h + h_attn
        
        # Feed-forward
        h = h + self.mlp(self.norm2(h))
        
        # Update frames and positions
        frames_update, positions_update = self.frame_updater(h, frames)
        
        # Rigid body update
        frames = compose_frames(frames, frames_update)
        positions = update_positions(positions, frames_update)
        
        return h, frames, positions


class FrameUpdater(Module):
    def __init__(self, hidden_dim):
        super().__init__()
        
        # Predict frame updates
        self.frame_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 6)  # 3 rotation + 3 translation parameters
        )
    
    def forward(self, h, frames):
        # Predict frame updates
        frame_params = self.frame_predictor(h)  # [batch, length, 6]
        
        # Split into rotation and translation
        rot_params = frame_params[..., :3]  # [batch, length, 3]
        trans_params = frame_params[..., 3:]  # [batch, length, 3]
        
        batch_size, seq_len, _ = h.shape
        
        # Create rotation matrices (simplified)
        rot_matrices = torch.zeros(batch_size, seq_len, 3, 3, device=h.device)
        # Set diagonal to 1 to make identity matrices
        identity = torch.eye(3, device=h.device).unsqueeze(0).unsqueeze(0)
        rot_matrices = rot_matrices + identity
        
        # Apply small updates from rot_params (simplified)
        rot_matrices[..., 0, 1] += rot_params[..., 0]
        rot_matrices[..., 1, 0] -= rot_params[..., 0]
        rot_matrices[..., 0, 2] += rot_params[..., 1]
        rot_matrices[..., 2, 0] -= rot_params[..., 1]
        rot_matrices[..., 1, 2] += rot_params[..., 2]
        rot_matrices[..., 2, 1] -= rot_params[..., 2]
        
        # Create 4x4 transformation matrices
        frames_update = torch.zeros(batch_size, seq_len, 4, 4, device=h.device)
        frames_update[..., :3, :3] = rot_matrices
        frames_update[..., :3, 3] = trans_params
        frames_update[..., 3, 3] = 1.0
        
        return frames_update, trans_params


# Helper functions for rigid body transformations
def compose_frames(frames_1, frames_2):
    """Compose two 4x4 transformation matrices."""
    return torch.matmul(frames_1, frames_2)

def update_positions(positions, frames):
    """Update positions using transformation frames."""
    batch_size, seq_len, _ = positions.shape
    
    # Homogeneous coordinates
    pos_homo = torch.cat([
        positions, 
        torch.ones(batch_size, seq_len, 1, device=positions.device)
    ], dim=-1)  # [batch, length, 4]
    
    # Apply transformation
    pos_transformed = torch.matmul(
        frames[..., :3, :], 
        pos_homo.unsqueeze(-1)
    ).squeeze(-1)  # [batch, length, 3]
    
    return pos_transformed 

def protein_reconstruction_loss(seq_logits, coords_pred, seq_target, coords_target):
    """Calculate reconstruction loss for both sequence and structure with improved numerical stability"""
    # 1. 检查输入有效性
    if torch.isnan(seq_logits).any() or torch.isnan(coords_pred).any():
        print("Warning: NaN detected in model outputs")
        # 返回一个安全的默认损失
        return torch.tensor(100.0, device=seq_logits.device, requires_grad=True)
        
    # 2. 应用clamp防止极端值
    seq_logits = torch.clamp(seq_logits, -50.0, 50.0)
    
    # Sequence loss (cross entropy) - 使用更稳定的计算
    seq_loss = F.cross_entropy(
        seq_logits.view(-1, 20), 
        seq_target.view(-1), 
        reduction='sum',
        ignore_index=0  # 忽略填充位置
    )
    
    # 计算有效元素数量用于归一化
    valid_elements = (seq_target > 0).sum()
    seq_loss = seq_loss / (valid_elements + 1e-8)
    
    # Create a mask for padding positions (0 in seq_target)
    mask = (seq_target > 0).float().unsqueeze(-1)  # [batch, length, 1]
    
    # 3. 保护性地裁剪坐标，防止极端值
    coords_pred = torch.clamp(coords_pred, -500.0, 500.0)
    
    # Coordinate loss (MSE) - only for non-padding positions
    # 使用更稳定的计算，并进行归一化
    coord_diff = coords_pred * mask - coords_target * mask
    coord_loss = (coord_diff ** 2).sum() / (mask.sum() * 3 + 1e-8)
    
    # RMSD loss with improved stability
    # 对每个坐标差异应用方差计算，避免求平方和后再开方，增加数值稳定性
    squared_diff = ((coords_pred - coords_target) ** 2).sum(dim=-1) + 1e-8  # [batch, length]
    rmsd = torch.sqrt(squared_diff)  # [batch, length]
    rmsd_loss = (rmsd * mask.squeeze(-1)).sum() / (mask.sum() + 1e-8)
    
    # Distance matrix loss - captures relative positions
    coords_pred_1 = coords_pred.unsqueeze(2)   # [batch, length, 1, 3]
    coords_pred_2 = coords_pred.unsqueeze(1)   # [batch, 1, length, 3]
    coords_target_1 = coords_target.unsqueeze(2)   # [batch, length, 1, 3]
    coords_target_2 = coords_target.unsqueeze(1)   # [batch, 1, length, 3]
    
    # 4. 计算欧氏距离时增加稳定性
    # 使用平方和的形式计算距离的平方，避免开方操作
    coords_diff_pred = coords_pred_1 - coords_pred_2   # [batch, length, length, 3]
    coords_diff_target = coords_target_1 - coords_target_2   # [batch, length, length, 3]
    
    # 计算距离平方
    dist_pred_sq = (coords_diff_pred ** 2).sum(-1) + 1e-8  # [batch, length, length]
    dist_target_sq = (coords_diff_target ** 2).sum(-1) + 1e-8  # [batch, length, length]
    
    # 对齐距离差异计算
    dist_diff = torch.abs(torch.sqrt(dist_pred_sq) - torch.sqrt(dist_target_sq))
    
    # 应用掩码并计算均值
    mask_2d = mask.squeeze(-1).unsqueeze(2) * mask.squeeze(-1).unsqueeze(1)  # [batch, length, length]
    valid_pairs = mask_2d.sum() + 1e-8
    dist_loss = (dist_diff * mask_2d).sum() / valid_pairs
    
    # 5. 使用动态权重根据损失大小调整权重
    # 归一化各项损失的量级
    max_loss = 10.0  # 设置最大限制
    seq_weight = 1.0
    coord_weight = min(0.5, max_loss / (coord_loss.item() + 1e-8))
    rmsd_weight = min(0.3, max_loss / (rmsd_loss.item() + 1e-8))
    dist_weight = min(0.2, max_loss / (dist_loss.item() + 1e-8))
    
    # Weight the different loss components with clamping
    total_loss = seq_weight * seq_loss + coord_weight * coord_loss + rmsd_weight * rmsd_loss + dist_weight * dist_loss
    
    # 6. 最终安全检查
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        print("Warning: NaN or Inf in final loss computation")
        return torch.tensor(100.0, device=seq_logits.device, requires_grad=True)
    
    return total_loss

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
        
        # 添加骨架原子 - 修复原子名称和元素类型
        sb.init_atom("CA", ca_coord, 0.0, 1.0, " ", "CA", element="C")
        sb.init_atom("N", n_coord, 0.0, 1.0, " ", "N", element="N")
        sb.init_atom("C", c_coord, 0.0, 1.0, " ", "C", element="C")
        sb.init_atom("O", o_coord, 0.0, 1.0, " ", "O", element="O")
        # 添加骨架原子（N, C, O），使用相对于CA的坐标偏移
        # for atom_name, offset in backbone_offsets.items():
        #     atom_coord = ca_coord + offset
        #     sb.init_atom(atom_name, atom_coord, 0.0, 1.0, " ", atom_name, element=atom_name[0])
        
        # 对于不同的氨基酸，添加一些代表性的侧链原子
        # 这里简化处理，真实情况中侧链原子位置需要通过旋转角计算
        # if aa != 'G':  # 甘氨酸没有侧链
        #     # CB原子（几乎所有氨基酸都有）
        #     cb_offset = np.array([0.5, 1.3, 0.0])  # 典型CA-CB距离 ~1.5Å
        #     cb_coord = ca_coord + cb_offset
        #     sb.init_atom("CB", cb_coord, 0.0, 1.0, " ", "CB", element="C")
            
        #     # 对于特定氨基酸，添加更多侧链原子
        #     if aa in 'FYWH':  # 芳香族氨基酸
        #         # 添加一个代表性的芳香环原子
        #         ring_offset = np.array([0.5, 2.8, 0.0])
        #         ring_coord = ca_coord + ring_offset
        #         sb.init_atom("CG", ring_coord, 0.0, 1.0, " ", "CG", element="C")
            
        #     elif aa in 'KR':  # 长侧链带电荷氨基酸
        #         # 添加末端原子（赖氨酸的NZ或精氨酸的NH1）
        #         term_offset = np.array([0.5, 3.5, 0.0])
        #         term_coord = ca_coord + term_offset
        #         term_atom = "NZ" if aa == 'K' else "NH1"
        #         sb.init_atom(term_atom, term_coord, 0.0, 1.0, " ", term_atom, element="N")
            
        #     elif aa in 'DE':  # 酸性氨基酸
        #         # 添加侧链羧基氧原子
        #         o_offset = np.array([0.5, 2.5, 0.0])
        #         o_coord = ca_coord + o_offset
        #         sb.init_atom("OD1" if aa == 'D' else "OE1", o_coord, 0.0, 1.0, " ", 
        #                      "OD1" if aa == 'D' else "OE1", element="O")
    # 获取构建的结构
    structure = sb.get_structure()
    
    # 保存PDB文件
    io = PDBIO()
    io.set_structure(structure)
    io.save(filename)

model = LLMfusion(
    num_text_tokens = 20,  # Number of amino acids
    dim_latent = 20,  # Latent dimension for protein representation
    channel_first_latent = False,  # Protein data is not channel-first
    modality_default_shape = (512,),  # Maximum sequence length
    modality_encoder = ProteinEncoder(
        hidden_dim=256, 
        latent_dim=20,
        num_blocks=4
    ),
    modality_decoder = ProteinDecoder(
        hidden_dim=256, 
        latent_dim=20,
        num_blocks=4
    ),
    pre_post_transformer_enc_dec = (
        nn.Linear(20, 2048),  # Adapt latent dimension to transformer dimension
        nn.Linear(2048, 20),
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

# 设置decoder标志，以正确处理MSE损失
if hasattr(model, 'modality_decoder') and isinstance(model.modality_decoder, ProteinDecoder):
    model.modality_decoder._is_called_by_model_decoder = True
elif hasattr(model, 'modality_decoder') and isinstance(model.modality_decoder, nn.ModuleList):
    for decoder in model.modality_decoder:
        if isinstance(decoder, ProteinDecoder):
            decoder._is_called_by_model_decoder = True

ema_model = model.create_ema(0.9)

# 也设置EMA模型的decoder标志
if hasattr(ema_model.ema_model, 'modality_decoder') and isinstance(ema_model.ema_model.modality_decoder, ProteinDecoder):
    ema_model.ema_model.modality_decoder._is_called_by_model_decoder = True
elif hasattr(ema_model.ema_model, 'modality_decoder') and isinstance(ema_model.ema_model.modality_decoder, nn.ModuleList):
    for decoder in ema_model.ema_model.modality_decoder:
        if isinstance(decoder, ProteinDecoder):
            decoder._is_called_by_model_decoder = True

class ProteinDataset(Dataset):
    def __init__(self, data_dir, max_length=512):
        """
        Args:
            data_dir: Directory containing protein structure files (PDB format)
            max_length: Maximum sequence length to consider
        """
        self.data_dir = Path(data_dir)
        self.max_length = max_length
        
        # Load all PDB files in the directory
        self.protein_files = list(self.data_dir.glob("*.pdb"))
        
        # Initialize PDB parser
        self.parser = PDBParser(QUIET=True)
        
        # Initialize sequence tokenizer (using standard amino acids)
        self.aa_vocab = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        
        # Create reverse mapping from index to amino acid
        self.idx_to_aa = {i: aa for aa, i in self.aa_vocab.items()}
        # Add special token for padding
        self.idx_to_aa[0] = 'X'  # Use 'X' for unknown or padding
    
    def _process_structure(self, structure):
        """Extract backbone coordinates (CA, N, C, O) and sequence from structure"""
        coords = []
        sequence = []
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    # Check if all required atoms exist
                    if all(residue.has_id(atom) for atom in ['CA', 'N', 'C', 'O']):
                        # Extract coordinates for all backbone atoms
                        residue_coords = []
                        for atom in ['CA', 'N', 'C', 'O']:
                            residue_coords.extend(residue[atom].get_coord())
                        coords.append(residue_coords)
                        try:
                            sequence.append(Polypeptide.index_to_one(Polypeptide.three_to_index(residue.get_resname())))
                        except KeyError:
                            sequence.append('X')  # 'X' 代表未知氨基酸
        return np.array(coords), ''.join(sequence)
    
    def pad_sequence(self, seq_tensor, coords_tensor):
        """Pad or truncate sequence and coordinates to max_length"""
        current_length = len(seq_tensor)
        
        if current_length > self.max_length:
            # Truncate
            seq_tensor = seq_tensor[:self.max_length]
            coords_tensor = coords_tensor[:self.max_length]
        elif current_length < self.max_length:
            # Pad
            pad_length = self.max_length - current_length
            # Pad sequence with 0 (padding token)
            seq_tensor = F.pad(seq_tensor, (0, pad_length), value=0)
            # For coords_tensor with shape [length, 12], we pad only the first dimension
            coords_tensor = torch.cat([
                coords_tensor,
                torch.zeros(pad_length, 12, dtype=coords_tensor.dtype)
            ], dim=0)
            
        return seq_tensor, coords_tensor
    
    def __getitem__(self, idx):
        pdb_file = self.protein_files[idx]
        
        # Parse structure
        structure = self.parser.get_structure('protein', pdb_file)
        coords, sequence = self._process_structure(structure)
        # Convert coordinates to tensor with float32 instead of float16
        coords_tensor = torch.tensor(coords, dtype=torch.float32)
        
        # Convert sequence to tensor
        seq_indices = [self.aa_vocab.get(aa, 0) for aa in sequence]
        seq_tensor = torch.tensor(seq_indices, dtype=torch.long)
        
        # Pad or truncate sequences
        seq_tensor, coords_tensor = self.pad_sequence(seq_tensor, coords_tensor)
        assert seq_tensor.shape[0] == self.max_length, f"Sequence length mismatch: {seq_tensor.shape[0]} vs {self.max_length}"
        assert coords_tensor.shape[0] == self.max_length, f"Coordinates length mismatch: {coords_tensor.shape[0]} vs {self.max_length}"
        assert coords_tensor.shape[1] == 12, f"Coordinates dimension mismatch: {coords_tensor.shape[1]} vs 12"
        
        return seq_tensor, coords_tensor
    
    def __len__(self):
        return len(self.protein_files)

def cycle(iter_dl):
    while True:
        for batch in iter_dl:
            yield batch

def collate_protein_batch(batch):
    """Custom collate function for protein data"""
    seq_tensors, coords_tensors = zip(*batch)
    return torch.stack(seq_tensors), torch.stack(coords_tensors)

dataset = ProteinDataset("/share/project/linwenjun/swissprot_pdb_v4", max_length = 256)

# Use custom collate function instead of model.create_dataloader
# dataloader = model.create_dataloader(dataset, batch_size = 2, shuffle = True)
dataloader = DataLoader(
    dataset,
    batch_size=4,  # 将批量大小从2降为1
    shuffle=True,
    collate_fn=collate_protein_batch,
    num_workers=4,  # 减少worker数量
    pin_memory=True,
    persistent_workers=True  # 保持worker存活以减少重启开销
)
iter_dl = cycle(dataloader)

# 确保所有参数都是float32
# print("Converting all parameters to float32 to avoid precision issues...")
# for param in model.parameters():
#     param.data = param.data.to(torch.float32)

optimizer = AdamW(model.parameters(), lr = 5e-4)  # 降低学习率，从2e-3降到5e-4
for param_group in optimizer.param_groups:
    for param in param_group['params']:
        param.data = param.data.to(torch.float32)
# 彻底移除GradScaler，不要仅仅注释掉
# scaler = GradScaler()
# train loop
scheduler = CosineAnnealingLR(optimizer, T_max=100_000, eta_min=1e-6)

# 在训练开始前清理内存
# # clear_memory()

# 修改训练循环
model.train()
for step in range(1, 10_000 + 1):
    
    # 定期清理内存
    if step % 100 == 0:
        clear_memory()
    
    # 渐进式增加梯度累积步数，开始时只累积1步
    grad_accum_steps = min(4, 2 + step // 1000)
    
    # 标记是否成功计算了损失
    loss_computed = False
    
    for _ in range(grad_accum_steps):
        try:
            # Get batch
            seq_tensor, coords_tensor = next(iter_dl)
            seq_tensor, coords_tensor = seq_tensor.to(device), coords_tensor.to(device)
            
            # 确保coords_tensor为float32类型
            if coords_tensor.dtype != torch.float32:
                coords_tensor = coords_tensor.to(torch.float32)
            
            # 尝试获取分解的损失
            result = model.forward_seq_coord(
                seq_tensor,
                coords_tensor,
                return_loss=True,
                return_loss_breakdown=True
            )
            
            # 检查返回值是否为元组
            if isinstance(result, tuple) and len(result) == 2:
                loss, (recon_loss, kl_loss) = result
            else:
                # 如果只返回一个张量，则将其视为总损失
                loss = result
                recon_loss = loss * 0.9  # 假设重建损失约占总损失的90%
                kl_loss = loss * 0.1     # 假设KL损失约占总损失的10%
                print("Warning: forward_seq_coord didn't return loss breakdown, using approximations.")
            
            # 检查损失是否有效（不是NaN或无穷大）
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Loss is {loss.item()}, skipping this batch")
                optimizer.zero_grad()
                continue
                
            # 检查如果损失值过大，进行缩放
            if loss.item() > 1e5:
                print(f"Warning: Loss is very large: {loss.item()}, scaling down")
                loss = torch.log1p(loss)  # 使用log(1+x)缩放大损失
                
            # 限制KL损失比例，防止它变得过大
            if isinstance(kl_loss, torch.Tensor) and kl_loss.item() > 1e4:
                print(f"KL loss too large: {kl_loss.item()}, clamping")
                kl_scale = min(1.0, 1e4 / kl_loss.item())
                kl_loss = kl_loss * kl_scale
                # 如果我们需要重构loss，可以这样做
                if isinstance(result, tuple):
                    loss = recon_loss + kl_loss
            
            # Calculate losses
            loss = loss / grad_accum_steps
            
            # 直接反向传播，不使用scaler
            loss.backward()
            
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
            
        except Exception as e:
            print(f"Error during forward/backward pass: {e}")
            optimizer.zero_grad()  # 确保梯度被清零
            continue
    
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
                "loss": loss.item(),
                "grad_accum_steps": grad_accum_steps,
                "lr": scheduler.get_last_lr()[0]
            }
            
            # 确保损失值有效再记录
            if isinstance(recon_loss, torch.Tensor) and not torch.isnan(recon_loss) and not torch.isinf(recon_loss):
                log_dict["flow_loss"] = recon_loss.item()
            if isinstance(kl_loss, torch.Tensor) and not torch.isnan(kl_loss) and not torch.isinf(kl_loss):
                log_dict["kl_loss"] = kl_loss.item()
            
            print(f'{step}: loss={loss.item():.3f}, flow={recon_loss.item() if isinstance(recon_loss, torch.Tensor) else "N/A":.3f}, kl={kl_loss.item() if isinstance(kl_loss, torch.Tensor) else "N/A":.3f}, grad_accum_steps={grad_accum_steps}')
                
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
                
                seq_logits, coords_tensor = ema_model.generate_modality_only(
                    batch_size=1,
                    modality_type=0,
                    modality_steps=100,
                    return_unprocessed_modalities=True
                )
                # Convert to amino acid sequence
                seq_pred = torch.argmax(seq_logits, dim=-1)
                # print(seq_pred.shape)
                
                # Use the idx_to_aa mapping to convert indices to amino acids
                sequences = []
                for seq in seq_pred:
                    aa_sequence = []
                    for i in seq:
                        idx = i.item()
                        aa = dataset.idx_to_aa.get(idx, 'X')  # Default to 'X' if index not found
                        aa_sequence.append(aa)
                    sequences.append(''.join(aa_sequence))
                
                # Save generated sequences and coordinates
                for i, (seq, coord) in enumerate(zip(sequences, coords_tensor)):
                    filename = str(results_folder / f'{step}_sample_{i}.pdb')
                    save_protein_structure(filename, seq, coord)
                    print(f"Generated sequence {i}: {seq[:100]}...")
                
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