from shutil import rmtree
from pathlib import Path
import os

# Set PyTorch memory allocator settings
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

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

from dillm import LLMfusion, print_modality_sample

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


vae = AutoencoderKL.from_pretrained("/share/project/xiaohongwang/LLM_checkpoints/stable-diffusion-v1-4", subfolder = "vae")
# vae = AutoencoderKL.from_pretrained(
#     "/share/project/xiaohongwang/LLM_checkpoints/stable-diffusion-v1-4", 
#     subfolder="vae",
#     torch_dtype=torch.float16,  # Use float16 for VAE
#     use_safetensors=True
# )
# vae.enable_gradient_checkpointing()

device = torch.device('cuda:0')
vae.to(device)

# class Encoder(Module):
#     def __init__(self, vae):
#         super().__init__()
#         self.vae = vae

#     def forward(self, image):
#         with torch.no_grad():
#             # Convert to float16 for VAE
#             # image = image.half()
#             latent = self.vae.encode(image * 2 - 1)
#         return 0.18215 * latent.latent_dist.sample()

# class Decoder(Module):
#     def __init__(self, vae):
#         super().__init__()
#         self.vae = vae

#     def forward(self, latents):
#         latents = (1 / 0.18215) * latents
#         with torch.no_grad():
#             image = self.vae.decode(latents).sample
#         return (image / 2 + 0.5).clamp(0, 1)

# # Simple identity encoder/decoder for text
# class TextIdentityEncoder(Module):
#     def forward(self, text):
#         # Just return the text as is
#         return text

# class TextIdentityDecoder(Module):
#     def forward(self, text):
#         # Just return the text as is
#         return text

# results folder

rmtree('./results_dillm/train_latent_only_dillm_prot', ignore_errors = True)
results_folder = Path('./results_dillm/train_latent_only_dillm_prot')
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
    def __init__(self, hidden_dim=256, latent_dim=32):
        super().__init__()
        
        # Sequence embedding
        self.aa_embedding = nn.Embedding(20, hidden_dim)  # 20 standard amino acids
        
        # Coordinate processing
        self.coord_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Combined processing
        self.encoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim * 2)  # Mean and log variance
        )
        
        self.latent_dim = latent_dim
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def forward(self, seq_tensor, coord_tensor):
        # Process sequence
        seq_embed = self.aa_embedding(seq_tensor)  # [batch, length, hidden_dim]
        
        # Process coordinates
        coord_embed = self.coord_encoder(coord_tensor)  # [batch, length, hidden_dim]
        
        # Combine embeddings
        combined = torch.cat([seq_embed, coord_embed], dim=-1)
        
        # Generate latent representation
        hidden = self.encoder(combined)
        mu, logvar = hidden.chunk(2, dim=-1)
        
        # Sample latent vector
        z = self.reparameterize(mu, logvar)
        
        return z, mu, logvar

class ProteinDecoder(Module):
    def __init__(self, hidden_dim=256, latent_dim=32):
        super().__init__()
        
        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Sequence generation
        self.seq_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 20)  # 20 standard amino acids
        )
        
        # Coordinate generation
        self.coord_decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3)  # 3D coordinates
        )
        
    def forward(self, z):
        # Project latent vector
        hidden = self.latent_proj(z)
        
        # Generate sequence probabilities and coordinates
        seq_logits = self.seq_decoder(hidden)
        coords = self.coord_decoder(hidden)
        
        return seq_logits, coords

model = LLMfusion(
    num_text_tokens = 20,  # Number of amino acids
    dim_latent = 32,  # Latent dimension for protein representation
    channel_first_latent = False,  # Protein data is not channel-first
    modality_default_shape = (512,),  # Maximum sequence length
    modality_encoder = ProteinEncoder(hidden_dim=256, latent_dim=32),
    modality_decoder = ProteinDecoder(hidden_dim=256, latent_dim=32),
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
        'torch_dtype': torch.float16,
        'use_gradient_checkpointing': True
    },
).to(device)

ema_model = model.create_ema(0.9)

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
        
    def __len__(self):
        return len(self.protein_files)
    
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
            # coords_tensor = torch.cat([
            #     coords_tensor,
            #     torch.zeros(pad_length, 3, dtype=coords_tensor.dtype)
            # ], dim=0)
            coords_tensor = F.pad(coords_tensor, (0, 0, 0, pad_length), value=0)
            
        return seq_tensor, coords_tensor
    
    def __getitem__(self, idx):
        pdb_file = self.protein_files[idx]
        
        # Parse structure
        structure = self.parser.get_structure('protein', pdb_file)
        coords, sequence = self._process_structure(structure)
        # Convert coordinates to tensor
        coords_tensor = torch.tensor(coords, dtype=torch.float32)
        
        # Convert sequence to tensor
        seq_indices = [self.aa_vocab.get(aa, 0) for aa in sequence]
        seq_tensor = torch.tensor(seq_indices, dtype=torch.long)
        
        # Pad or truncate if necessary
        # if len(seq_tensor) > self.max_length:
        #     seq_tensor = seq_tensor[:self.max_length]
        #     coords_tensor = coords_tensor[:self.max_length]
        # elif len(seq_tensor) < self.max_length:
        #     pad_length = self.max_length - len(seq_tensor)
        #     seq_tensor = F.pad(seq_tensor, (0, pad_length), value=0)
        #     coords_tensor = F.pad(coords_tensor, (0, 0, 0, pad_length), value=0)
                    # Pad or truncate sequences
        seq_tensor, coords_tensor = self.pad_sequence(seq_tensor, coords_tensor)
        assert seq_tensor.shape[0] == self.max_length, f"Sequence length mismatch: {seq_tensor.shape[0]} vs {self.max_length}"
        assert coords_tensor.shape[0] == self.max_length, f"Coordinates length mismatch: {coords_tensor.shape[0]} vs {self.max_length}"
        assert coords_tensor.shape[1] == 3, f"Coordinates dimension mismatch: {coords_tensor.shape[1]} vs 3"
        
        return seq_tensor, coords_tensor

def cycle(iter_dl):
    while True:
        for batch in iter_dl:
            yield batch


def protein_reconstruction_loss(seq_logits, coords_pred, seq_target, coords_target):
    """Calculate reconstruction loss for both sequence and structure"""
    # Sequence loss (cross entropy)
    seq_loss = F.cross_entropy(seq_logits.view(-1, 20), seq_target.view(-1))
    
    # Coordinate loss (MSE)
    coord_loss = F.mse_loss(coords_pred, coords_target)
    
    # RMSD loss for overall structure quality
    rmsd_loss = torch.sqrt(((coords_pred - coords_target) ** 2).sum(dim=-1)).mean()
    
    return seq_loss + coord_loss + 0.1 * rmsd_loss

def save_protein_structure(filename, sequence, coordinates):
    """Save generated protein structure in PDB format"""
    with open(filename, 'w') as f:
        for i, (aa, coord) in enumerate(zip(sequence, coordinates)):
            x, y, z = coord
            f.write(f"ATOM  {i+1:5d}  CA  {aa:3s} A{i+1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C  \n")
        f.write("END\n")

def collate_protein_batch(batch):
    """Custom collate function for protein data"""
    seq_tensors, coord_tensors = zip(*batch)
    return torch.stack(seq_tensors), torch.stack(coord_tensors)

dataset = ProteinDataset("/share/project/linwenjun/swissprot_pdb_v4", max_length = 512)

# Use custom collate function instead of model.create_dataloader
# dataloader = model.create_dataloader(dataset, batch_size = 2, shuffle = True)
dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,
    collate_fn=collate_protein_batch,
    num_workers=4,
    pin_memory=True
)
iter_dl = cycle(dataloader)

optimizer = AdamW(model.parameters(), lr = 2e-3)
for param_group in optimizer.param_groups:
    for param in param_group['params']:
        param.data = param.data.to(torch.float32)

# Initialize gradient scaler for mixed precision training
scaler = GradScaler()

# train loop
scheduler = CosineAnnealingLR(optimizer, T_max=100_000, eta_min=1e-6)

# Training loop
for step in range(1, 100_000 + 1):
    model.train()
    
    for _ in range(4):
        with autocast('cuda'):
            # Get batch
            seq_tensor, coords_tensor = next(iter_dl)
            seq_tensor, coords_tensor = seq_tensor.to(device), coords_tensor.to(device)
            
            # Forward pass
            # z, mu, logvar = model.modality_encoder(seq_tensor, coords_tensor)
            # seq_logits, coords_pred = model.modality_decoder(z)
            # recon_loss = protein_reconstruction_loss(seq_logits, coords_pred, seq_tensor, coords_tensor)
            # kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            
            # loss = recon_loss + 0.1 * kl_loss
            loss, (recon_loss, kl_loss) = model.forward_protein(
                    seq_tensor,
                    coords_tensor,
                    return_loss=True
                )
            # # Calculate losses
            loss = loss / 4
        
        scaler.scale(loss).backward()
    
    # Optimizer step
    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), 0.5)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    scheduler.step()
    
    ema_model.update()
    
    # Logging
    wandb.log({
        "loss": loss.item(),
        "recon_loss": recon_loss.item(),
        "kl_loss": kl_loss.item()
    }, step=step)
    print(f'{step}: loss={loss.item():.3f}, recon={recon_loss.item():.3f}, kl={kl_loss.item():.3f}')
    
    # Sampling
    if divisible_by(step, SAMPLE_EVERY):
        model.eval()
        with torch.no_grad():
            torch.cuda.empty_cache()
            ema_model.to(torch.float32)
            
            # Sample from the model
            # z = torch.randn(2, 512, 32).to(device)  # [batch, length, latent_dim]
            # seq_logits, coords = ema_model.modality_decoder(z)
            seq_logits, coords_tensor = ema_model.generate_modality_only(
                batch_size=1,
                modality_type=0,
                modality_steps=100,
                return_unprocessed_modalities=True
            )
            # print(seq_logits.shape, coords_tensor.shape) #torch.Size([1, 512, 20]) torch.Size([1, 512, 3])
            # Convert to amino acid sequence
            seq_pred = torch.argmax(seq_logits, dim=-1)
            print(seq_pred)
            sequences = [''.join([dataset.aa_vocab.get(i.item(), 'X') for i in seq]) for seq in seq_pred]
            
            # Save generated sequences and coordinates
            for i, (seq, coord) in enumerate(zip(sequences, coords_tensor)):
                filename = str(results_folder / f'{step}_sample_{i}.pdb')
                save_protein_structure(filename, seq, coord)
                print(f"Generated sequence {i}: {seq[:50]}...")
            
        torch.cuda.empty_cache()

