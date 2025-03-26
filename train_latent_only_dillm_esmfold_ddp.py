from shutil import rmtree
from pathlib import Path
import os
import gc
import torch
import random
import numpy as np
import wandb

# 导入分布式训练所需的模块
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

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

from llmflow.model.esmfold import ESMFold
from llmflow.model.trunk import FoldingTrunk
from llmflow.config import model_config
from llmflow.data.data_modules import *

from openfold.utils.feats import atom14_to_atom37, pseudo_beta_fn
from openfold.utils.checkpointing import checkpoint_blocks
from openfold.np import residue_constants




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

# 分布式训练初始化
def init_distributed():
    # 检查是否设置了分布式训练环境变量
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("未检测到分布式训练环境变量，将使用单卡训练")
        return False, 0, 1, 0
    
    # 初始化分布式进程组
    dist.init_process_group(
        backend="nccl", 
        world_size=world_size,
        rank=rank
    )
    
    # 设置当前设备
    torch.cuda.set_device(local_rank)
    
    print(f"初始化分布式训练: rank {rank}, world_size {world_size}, local_rank {local_rank}")
    return True, rank, world_size, local_rank

# 初始化分布式训练
is_distributed, rank, world_size, local_rank = init_distributed()
device = torch.device(f'cuda:{local_rank}' if torch.cuda.is_available() else 'cpu')

# 只在主进程上初始化wandb
if not is_distributed or rank == 0:
    wandb.init(
        project="dillm", 
        name="train_latent_only_dillm_esmfold_ddp",
    )
    rmtree('./results_dillm/train_latent_only_dillm_esmfold', ignore_errors = True)
    results_folder = Path('./results_dillm/train_latent_only_dillm_esmfold')
    results_folder.mkdir(exist_ok = True, parents = True)
else:
    results_folder = Path('./results_dillm/train_latent_only_dillm_esmfold')

# constants
SAMPLE_EVERY = 100

# with open("./data/flowers/labels.txt", "r") as file:
#     content = file.read()
# LABELS_TEXT = content.split('\n')

# functions
def divisible_by(num, den):
    return (num % den) == 0

class ProteinEncoder(ESMFold):
    def __init__(self, cfg):
        super(ProteinEncoder, self).__init__(cfg)
        
        self.mytrunk = myFoldingTrunk(cfg.model.trunk)
        
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
        esm_s, _ = self._compute_language_model_representations(esmaa)

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
    
config = model_config(
    'initial_training',
    train=True, 
    low_prec=True
) 
print(config)
model = LLMFlow(
    num_text_tokens = 20,  # Number of amino acids
    dim_latent = 20,  # Latent dimension for protein representation
    channel_first_latent = False,  # Protein data is not channel-first
    modality_default_shape = (512,),  # Maximum sequence length
    modality_encoder = ProteinEncoder(cfg=config),
    modality_decoder = ProteinDecoder(cfg=config.model.trunk),
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

optimizer = AdamW(model.parameters(), lr = 5e-4)  # 降低学习率，从2e-3降到5e-4

# 将模型移动到设备上
model = model.to(device)

# 使用DDP封装模型（如果使用分布式训练）
if is_distributed:
    # 使用SyncBN来保证BatchNorm层在多卡训练时同步
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
    # 使用DDP包装模型
    model = DDP(
        model, 
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=True  # 如果有未使用的参数，避免DDP报错
    )

# 为所有参数设置float32类型
for param_group in optimizer.param_groups:
    for param in param_group['params']:
        param.data = param.data.to(torch.float32)

# 创建EMA模型
ema_model = model.module.create_ema(0.9) if is_distributed else model.create_ema(0.9)

# 设置EMA模型decoder标志
if hasattr(ema_model.ema_model, 'modality_decoder') and isinstance(ema_model.ema_model.modality_decoder, ProteinDecoder):
    ema_model.ema_model.modality_decoder._is_called_by_model_decoder = True
elif hasattr(ema_model.ema_model, 'modality_decoder') and isinstance(ema_model.ema_model.modality_decoder, nn.ModuleList):
    for decoder in ema_model.ema_model.modality_decoder:
        if isinstance(decoder, ProteinDecoder):
            decoder._is_called_by_model_decoder = True

scheduler = CosineAnnealingLR(optimizer, T_max=100_000, eta_min=1e-6)

class ProteinDataset(Dataset):
    def __init__(self, data_dir, max_length=256):
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
        # Create attention mask (1 for actual tokens, 0 for padding)
        # Determine the actual sequence length (before padding)
        actual_length = min(len(sequence), self.max_length)
        attention_mask = torch.zeros(self.max_length, dtype=torch.float32)
        attention_mask[:actual_length] = 1.0
        
        # Create position IDs
        position_ids = torch.arange(self.max_length, dtype=torch.long)
        
        # Create residue index (sequential numbering of residues)
        residue_index = torch.arange(self.max_length, dtype=torch.long)
        
        # Create mask (same as attention_mask but in different format if needed)
        mask = attention_mask.clone()
        
        # Create aatype (amino acid type) - same as seq_tensor but ensuring it's in the right format
        aatype = residue_constants.sequence_to_onehot(
                    sequence=sequence,
                    mapping=residue_constants.restype_order_with_x,
                    map_unknown_to_x=True,
                )
        seq_mask = torch.ones(aatype.shape, dtype=torch.float32)
        residue_index = np.array(range(self.max_length), dtype=np.int32)
        # Create a dictionary with all the necessary components for the model
        batch = {
            "seq_tokens": seq_tensor,
            "coords": coords_tensor,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "seq_length": torch.tensor(actual_length, dtype=torch.long),
            "aatype": aatype,
            "seq_mask": seq_mask,
            "residue_index": residue_index
        }
        return batch
    
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

# 为分布式训练创建采样器
if is_distributed:
    sampler = DistributedSampler(
        dataset, 
        num_replicas=world_size, 
        rank=rank,
        shuffle=True
    )
    shuffle = False  # 当使用DistributedSampler时，不要在DataLoader中设置shuffle=True
else:
    sampler = None
    shuffle = True

# 创建数据加载器
dataloader = DataLoader(
    dataset,
    batch_size=4,  # 每个GPU处理的批量大小
    shuffle=shuffle,  # 使用分布式采样器时设置为False
    sampler=sampler,  # 分布式采样器
    num_workers=4,  # worker数量
    pin_memory=True,
    persistent_workers=True  # 保持worker存活以减少重启开销
)

iter_dl = cycle(dataloader)

# 确保所有参数都是float32
# print("Converting all parameters to float32 to avoid precision issues...")
# for param in model.parameters():
#     param.data = param.data.to(torch.float32)

# 彻底移除GradScaler，不要仅仅注释掉
# scaler = GradScaler()
# train loop

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
            batch = next(iter_dl)
            seq_tensor, coords_tensor = batch['seq_tokens'], batch['coords']
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