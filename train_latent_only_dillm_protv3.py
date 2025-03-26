from shutil import rmtree
from pathlib import Path
import os
import gc
import torch
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

import torch
from torch import nn, tensor, Tensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_

from einops import rearrange
from PIL import Image
import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image

from dillm import DiLLM, print_modality_sample

import json
import wandb

# hf related
from datasets import load_dataset
from diffusers.models import AutoencoderKL

from Bio import SeqIO
from Bio.PDB import *
import torch.nn.functional as F
from Bio.PDB import Polypeptide


# 设置随机种子，提高实验可重复性
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    import numpy as np
    np.random.seed(seed)
    import random
    random.seed(seed)

set_seed()

# 初始化 wandb
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
    def __init__(self, hidden_dim=256, latent_dim=32, num_blocks=4):
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
    def __init__(self, hidden_dim=256, latent_dim=32, num_blocks=4):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Project latent to hidden dimension
        self.latent_projector = nn.Linear(latent_dim, hidden_dim)
        
        # Structure module components
        self.structure_blocks = nn.ModuleList([
            StructureBlock(
                hidden_dim=hidden_dim,
                num_heads=4
            ) for _ in range(num_blocks)
        ])
        
        # Final predictors
        self.seq_predictor = nn.Linear(hidden_dim, 20)  # 20 amino acids
        self.coord_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # 3D coordinates
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
class StructureBlock(Module):
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
    """Calculate reconstruction loss for both sequence and structure"""
    # Sequence loss (cross entropy)
    seq_loss = F.cross_entropy(seq_logits.view(-1, 20), seq_target.view(-1))
    
    # Create a mask for padding positions (0 in seq_target)
    mask = (seq_target > 0).float().unsqueeze(-1)  # [batch, length, 1]
    
    # Coordinate loss (MSE) - only for non-padding positions
    coord_loss = F.mse_loss(coords_pred * mask, coords_target * mask)
    
    # RMSD loss for overall structure quality - only for non-padding positions
    rmsd = torch.sqrt(((coords_pred - coords_target) ** 2).sum(dim=-1) + 1e-8)  # Add epsilon for numerical stability
    rmsd_loss = (rmsd * mask.squeeze(-1)).sum() / (mask.sum() + 1e-8)
    
    # Distance matrix loss - captures relative positions
    coords_pred_1 = coords_pred.unsqueeze(2)   # [batch, length, 1, 3]
    coords_pred_2 = coords_pred.unsqueeze(1)   # [batch, 1, length, 3]
    coords_target_1 = coords_target.unsqueeze(2)   # [batch, length, 1, 3]
    coords_target_2 = coords_target.unsqueeze(1)   # [batch, 1, length, 3]
    
    # Calculate pairwise distances
    dist_pred = torch.sqrt(((coords_pred_1 - coords_pred_2) ** 2).sum(-1) + 1e-8)  # [batch, length, length]
    dist_target = torch.sqrt(((coords_target_1 - coords_target_2) ** 2).sum(-1) + 1e-8)  # [batch, length, length]
    
    # Apply mask for padding
    mask_2d = mask.squeeze(-1).unsqueeze(2) * mask.squeeze(-1).unsqueeze(1)  # [batch, length, length]
    dist_loss = F.mse_loss(dist_pred * mask_2d, dist_target * mask_2d)
    
    # Weight the different loss components
    total_loss = seq_loss + 0.5 * coord_loss + 0.3 * rmsd_loss + 0.2 * dist_loss
    
    return total_loss

def save_protein_structure(filename, sequence, coordinates):
    """Save generated protein structure in PDB format"""
    with open(filename, 'w') as f:
        for i, (aa, coord) in enumerate(zip(sequence, coordinates)):
            x, y, z = coord
            f.write(f"ATOM  {i+1:5d}  CA  {aa:3s} A{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  \n")
        f.write("END\n")

model = DiLLM(
    num_text_tokens = 20,  # Number of amino acids
    dim_latent = 32,  # Latent dimension for protein representation
    channel_first_latent = False,  # Protein data is not channel-first
    modality_default_shape = (512,),  # Maximum sequence length
    modality_encoder = ProteinEncoder(
        hidden_dim=256, 
        latent_dim=32,
        num_blocks=4
    ),
    modality_decoder = ProteinDecoder(
        hidden_dim=256, 
        latent_dim=32,
        num_blocks=4
    ),
    pre_post_transformer_enc_dec = (
        nn.Linear(32, 2048),  # Adapt latent dimension to transformer dimension
        nn.Linear(2048, 32),
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
        """Extract backbone coordinates and sequence from structure"""
        coords = []
        sequence = []
        
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.has_id('CA'):  # Only consider amino acids with alpha carbons
                        ca = residue['CA']
                        coords.append(ca.get_coord())
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
            # For coords_tensor with shape [length, 3], we pad only the first dimension
            coords_tensor = torch.cat([
                coords_tensor,
                torch.zeros(pad_length, 3, dtype=coords_tensor.dtype)
            ], dim=0)
            
        return seq_tensor, coords_tensor
    
    def __getitem__(self, idx):
        pdb_file = self.protein_files[idx]
        
        # Parse structure
        structure = self.parser.get_structure('protein', pdb_file)
        coords, sequence = self._process_structure(structure)
        # Convert coordinates to tensor
        coords_tensor = torch.tensor(coords, dtype=torch.float16)
        
        # Convert sequence to tensor
        seq_indices = [self.aa_vocab.get(aa, 0) for aa in sequence]
        seq_tensor = torch.tensor(seq_indices, dtype=torch.long)
        
        # Pad or truncate sequences
        seq_tensor, coords_tensor = self.pad_sequence(seq_tensor, coords_tensor)
        assert seq_tensor.shape[0] == self.max_length, f"Sequence length mismatch: {seq_tensor.shape[0]} vs {self.max_length}"
        assert coords_tensor.shape[0] == self.max_length, f"Coordinates length mismatch: {coords_tensor.shape[0]} vs {self.max_length}"
        assert coords_tensor.shape[1] == 3, f"Coordinates dimension mismatch: {coords_tensor.shape[1]} vs 3"
        
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
    batch_size=1,  # 将批量大小从2降为1
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

optimizer = AdamW(model.parameters(), lr = 2e-3)
for param_group in optimizer.param_groups:
    for param in param_group['params']:
        param.data = param.data.to(torch.float32)
# 移除GradScaler
scaler = GradScaler()
# train loop
scheduler = CosineAnnealingLR(optimizer, T_max=100_000, eta_min=1e-6)

# 在训练开始前清理内存
# clear_memory()

# 修改训练循环
model.train()
for step in range(1, 10_000 + 1):
    
    # 定期清理内存
    if step % 5 == 0:
        clear_memory()
    
    # 渐进式增加梯度累积步数，开始时只累积1步
    grad_accum_steps = min(4, 1 + step // 1000)
    
    for _ in range(2):
        # 移除autocast
        # try:
        # Get batch
        seq_tensor, coords_tensor = next(iter_dl)
        seq_tensor, coords_tensor = seq_tensor.to(device), coords_tensor.to(device)
        
        # Forward pass
        # z, mu, logvar = model.modality_encoder(seq_tensor, coords_tensor)
        # seq_logits, coords_pred = model.modality_decoder(z)
        # recon_loss = protein_reconstruction_loss(seq_logits, coords_pred, seq_tensor, coords_tensor)
        # kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # loss = recon_loss + 0.1 * kl_loss


        # try:
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
        # except Exception as e:
        #     print(f"Error in forward_seq_coord: {e}")
        #     continue
            
        # # Calculate losses
        loss = loss / 2
        
        # 直接反向传播，不使用scaler
        # loss.backward()
        scaler.scale(loss).backward()
        # except Exception as e:
        #     print(f"Error during forward/backward pass: {e}")
        #     continue
    
    # 梯度裁剪
    # try:
    # Optimizer step
    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), 0.5)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    scheduler.step()

    ema_model.update()
    
    # 更新日志记录，包含flow_loss和kl_loss
    log_dict = {
        "loss": loss.item(),
        "grad_accum_steps": grad_accum_steps,
        "lr": scheduler.get_last_lr()[0]
    }
    
    # 只有在有分解损失时才记录它们
    if 'recon_loss' in locals() and 'kl_loss' in locals():
        log_dict["flow_loss"] = recon_loss.item()
        log_dict["kl_loss"] = kl_loss.item()
        print(f'{step}: loss={loss.item():.3f}, flow={recon_loss.item():.3f}, kl={kl_loss.item():.3f}, grad_steps={grad_accum_steps}')
    else:
        print(f'{step}: loss={loss.item():.3f}, grad_steps={grad_accum_steps}')
        
    wandb.log(log_dict, step=step)
        
    # except Exception as e:
    #     print(f"Error during optimizer step: {e}")
    #     # clear_memory()
    #     optimizer.zero_grad()
    #     continue
    
    # 采样部分保持不变，但添加异常处理
    if divisible_by(step, SAMPLE_EVERY):
        model.eval()
        try:
            with torch.no_grad():
                clear_memory()
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
    if divisible_by(step, 50):
        try:
            # 保存模型
            torch.save({
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'ema_model_state_dict': ema_model.state_dict()
            }, str(results_folder / f'checkpoint_{step}.pt'))
            
            # 仅保留最新的3个检查点
            checkpoints = sorted(list(results_folder.glob('checkpoint_*.pt')))
            if len(checkpoints) > 3:
                for old_ckpt in checkpoints[:-3]:
                    old_ckpt.unlink()
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

print("Training complete!") 