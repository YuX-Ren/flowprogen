from shutil import rmtree
from pathlib import Path
from contextlib import nullcontext

import torch
from torch import tensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW

from einops import rearrange, repeat, pack

import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image

from llmflow import LLMFlow, print_modality_sample
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
# 设置默认的torch dtype为float16，与模型一致
torch.set_default_dtype(torch.float16)
# 启用自动混合精度训练，以提高数值稳定性
from torch.amp import GradScaler, autocast
scaler = GradScaler('cuda')

# 导入wandb用于记录训练过程
import wandb
import os

def init_distributed():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return
    
    torch.distributed.init_process_group(backend="nccl")
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    return local_rank

local_rank = init_distributed()

# 只在rank 0进程初始化wandb
if local_rank == 0:
    # 设置wandb环境变量，避免可能的权限问题
    # os.environ["WANDB_SILENT"] = "true"
    
    # 初始化wandb项目
    wandb.init(
        project="dillm-training",
        name="mnist-training",
        config={
            "learning_rate": 1e-8,
            "weight_decay": 1e-5,
            "batch_size": 8,
            "model": "LLMFlow-Llama3.2-3B",
            "freeze_steps": 5000,
            "accumulation_steps": 4,
        }
    )

# 创建自定义的RandomFourierEmbed类，确保使用float16
class Float16RandomFourierEmbed(Module):
    def __init__(self, dim):
        super().__init__()
        assert (dim % 2) == 0, f'dimension must be divisible by 2, got {dim}'
        self.dim = dim
        # 确保权重是float16类型
        self.register_buffer('weights', torch.randn(dim // 2, dtype=torch.float16))

    def forward(self, times):
        # 确保输入是float16类型
        times = times.to(torch.float16)
        
        # 保存原始形状以便后续处理
        original_shape = times.shape
        
        # 将 times 重塑为 2D 张量，以便进行一致的处理
        if times.ndim == 1:
            times = rearrange(times, 'b -> b 1')
        elif times.ndim > 2:
            # 如果 times 是高维张量，将其展平为 2D
            batch_dims = times.shape[:-1]  # 保存除最后一维外的所有维度
            times = times.reshape(-1, times.shape[-1])
        
        # 使用einsum替代einx.multiply，确保类型一致
        freqs = torch.einsum('...i,j->...ij', times, self.weights) * 2 * torch.pi
        
        # 确保sin和cos操作保持float16类型
        sin_freqs = freqs.sin()
        cos_freqs = freqs.cos()
        
        # 确保所有张量具有相同的维度
        # times 形状: [batch, time_dim]
        # sin_freqs 和 cos_freqs 形状: [batch, time_dim, dim//2]
        # 需要将 times 扩展为 [batch, time_dim, 1] 以便与其他张量连接
        times_expanded = times.unsqueeze(-1)
        
        # 使用cat替代pack，确保类型一致
        # 在最后一个维度上连接
        fourier_embed = torch.cat([times_expanded, sin_freqs, cos_freqs], dim=-1)
        
        # 如果原始输入是高维张量，恢复原始形状
        if times.ndim > 2 and len(original_shape) > 2:
            new_shape = batch_dims + fourier_embed.shape[-1:]
            fourier_embed = fourier_embed.reshape(new_shape)
            
        return fourier_embed

# Memory-efficient EMA implementation
class MemoryEfficientEMA:
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow_params = {}
        self.backup_params = {}
        self.training = True
        
        # Initialize shadow parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = param.data.clone().detach()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow_params[name] = self.shadow_params[name] * self.decay + param.data * (1 - self.decay)
    
    def apply_shadow(self):
        # Backup current parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup_params[name] = param.data.clone()
                param.data.copy_(self.shadow_params[name])
    
    def restore(self):
        # Restore backed up parameters
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup_params[name])
    
    def generate_modality_only(self, batch_size=1):
        # 应用EMA权重
        self.apply_shadow()
        # 生成模态
        with torch.no_grad():
            # 检查model是否有module属性（DDP包装的模型）
            if hasattr(self.model, 'module'):
                result = self.model.module.generate_modality_only(batch_size=batch_size)
            else:
                # 直接调用model的generate_modality_only方法
                result = self.model.generate_modality_only(batch_size=batch_size)
        # 恢复原始权重
        self.restore()
        return result
    
    # 添加 eval 方法，模拟 torch.nn.Module 的行为
    def eval(self):
        self.training = False
        # 应用 EMA 权重
        self.apply_shadow()
        # 将模型设置为评估模式
        self.model.eval()
        return self
    
    # 添加 train 方法，模拟 torch.nn.Module 的行为
    def train(self, mode=True):
        self.training = mode
        # 如果从评估模式切换回训练模式，恢复原始权重
        if mode and not self.training:
            self.restore()
        # 将模型设置为训练模式
        self.model.train(mode)
        return self
    
    # 添加 __call__ 方法，使其行为类似于模型
    def __call__(self, *args, **kwargs):
        # 确保使用 EMA 权重
        was_training = self.training
        if was_training:
            self.eval()
        
        # 调用模型
        try:
            # 如果是 DDP 模型，调用 module 属性
            if hasattr(self.model, 'module'):
                return self.model.module(*args, **kwargs)
            else:
                return self.model(*args, **kwargs)
        finally:
            # 恢复原始状态
            if was_training:
                self.train()
    
    # 添加 return_only_pred_flows 方法，以兼容 LLMFlow 的期望
    def return_only_pred_flows(self, *args, **kwargs):
        # 确保使用 EMA 权重
        was_training = self.training
        if was_training:
            self.eval()
        
        try:
            # 调用模型的 return_only_pred_flows 方法
            if hasattr(self.model, 'module'):
                if hasattr(self.model.module, 'return_only_pred_flows'):
                    return self.model.module.return_only_pred_flows(*args, **kwargs)
                else:
                    # 如果模型没有这个方法，尝试直接调用模型
                    return self.model.module(*args, **kwargs, return_only_pred_flows=True)
            else:
                if hasattr(self.model, 'return_only_pred_flows'):
                    return self.model.return_only_pred_flows(*args, **kwargs)
                else:
                    # 如果模型没有这个方法，尝试直接调用模型
                    return self.model(*args, **kwargs, return_only_pred_flows=True)
        finally:
            # 恢复原始状态
            if was_training:
                self.train()

# 只在rank 0进程执行目录清理和创建
if local_rank == 0:
    try:
        rmtree('./results_llmflow/train_image_only_dillm', ignore_errors = True)
        results_folder = Path('./results_llmflow/train_image_only_dillm')
        results_folder.mkdir(exist_ok = True, parents = True)
        print(f"Results folder created at {results_folder.absolute()}")
    except Exception as e:
        print(f"Error creating results folder: {e}")
        # 尝试在当前目录创建
        results_folder = Path('./results_llmflow/train_image_only_dillm')
        results_folder.mkdir(exist_ok = True, parents = True)
else:
    results_folder = Path('./results_llmflow/train_image_only_dillm')

# functions

def divisible_by(num, den):
    return (num % den) == 0

# encoder / decoder

class Encoder(Module):
    def forward(self, x):
        x = rearrange(x, '... 1 (h p1) (w p2) -> ... h w (p1 p2)', p1 = 2, p2 = 2)
        return x * 2 - 1

class Decoder(Module):
    def forward(self, x):
        x = rearrange(x, '... h w (p1 p2) -> ... 1 (h p1) (w p2)', p1 = 2, p2 = 2, h = 14)
        return ((x + 1) * 0.5).clamp(min = 0., max = 1.)

# 修补LlamaTransformer类，确保time_mlp使用float16
from llmflow.llmflow import LlamaTransformer
def check_gradients(model, step):
    """检查模型梯度是否包含NaN或Inf值"""
    has_nan = False
    has_inf = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                has_nan = True
                # if local_rank == 0:
                    # print(f"Step {step}: NaN gradient in {name}")
            if torch.isinf(param.grad).any():
                has_inf = True
                # if local_rank == 0:
                #     print(f"Step {step}: Inf gradient in {name}")
    return has_nan or has_inf

# 添加梯度值缩放函数，用于处理异常大的梯度
def scale_gradients(model, max_norm=1.0):
    """将梯度值缩放到合理范围内"""
    for name, param in model.named_parameters():
        if param.grad is not None:
            # 将NaN和Inf替换为0
            param.grad.data.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
            
            # 如果梯度值过大，进行缩放
            if param.grad.data.abs().max() > max_norm:
                scale = max_norm / (param.grad.data.abs().max() + 1e-6)
                param.grad.data.mul_(scale)

# 添加特殊的梯度处理函数，专门针对model_to_latent_projs参数
def handle_model_to_latent_projs_gradients(model, max_norm=0.0001):
    """特别处理model_to_latent_projs的梯度，使用更严格的限制"""
    def process_module(module):
        if hasattr(module, 'model_to_latent_projs'):
            for i, proj in enumerate(module.model_to_latent_projs):
                # 处理Sequential
                if isinstance(proj, torch.nn.Sequential):
                    for layer in proj:
                        if isinstance(layer, torch.nn.Linear) and layer.weight.grad is not None:
                            # 更严格的梯度处理 - 使用裁剪而非缩放
                            layer.weight.grad.data.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                            # 直接裁剪梯度值
                            layer.weight.grad.data.clamp_(-max_norm, max_norm)
                            
                            if layer.bias is not None and layer.bias.grad is not None:
                                layer.bias.grad.data.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                                layer.bias.grad.data.clamp_(-max_norm, max_norm)
                
                # 处理Linear
                elif isinstance(proj, torch.nn.Linear) and proj.weight.grad is not None:
                    # 更严格的梯度处理 - 使用裁剪而非缩放
                    proj.weight.grad.data.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                    # 直接裁剪梯度值
                    proj.weight.grad.data.clamp_(-max_norm, max_norm)
                    
                    if proj.bias is not None and proj.bias.grad is not None:
                        proj.bias.grad.data.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
                        proj.bias.grad.data.clamp_(-max_norm, max_norm)
    
    # 处理主模型
    process_module(model)
    
    # 如果是DDP模型，处理内部模块
    if hasattr(model, 'module'):
        process_module(model.module)

# 保存原始的__init__方法
original_init = LlamaTransformer.__init__

# 定义新的__init__方法
def patched_init(self, dim, **kwargs):
    # 调用原始的__init__方法
    original_init(self, dim, **kwargs)
    
    # 重新创建time_mlp，确保使用float16
    from torch.nn import Linear, Sequential, SiLU
    
    # 使用自定义的Float16RandomFourierEmbed
    fourier_embed = Float16RandomFourierEmbed(dim)
    
    # 重新创建time_mlp，确保所有组件都是float16
    self.time_mlp = Sequential(
        fourier_embed,
        Linear(dim + 1, dim * 4, dtype=torch.float16),
        SiLU(),
        Linear(dim * 4, dim, dtype=torch.float16)
    )

    # 确保forward方法中的times转换为float16
    original_forward = self.forward
    
    def float16_forward(self, *args, **kwargs):
        # 检查参数并确保数据类型正确
        # 如果第一个参数是张量（即x），则转换为float16
        if args and isinstance(args[0], torch.Tensor):
            args = list(args)
            args[0] = args[0].to(torch.float16)
            args = tuple(args)
        
        # 检查kwargs中是否有times参数，如果有则转换为float16
        if 'times' in kwargs and kwargs['times'] is not None:
            kwargs['times'] = kwargs['times'].to(torch.float16)
        
        return original_forward(self, *args, **kwargs)
    
    self.forward = float16_forward

# 替换__init__方法
LlamaTransformer.__init__ = patched_init

# 添加特定的初始化函数，用于处理model_to_latent_projs的梯度问题
def initialize_model_to_latent_projs(model):
    """特别处理model_to_latent_projs的初始化，以避免梯度爆炸"""
    if hasattr(model, 'model_to_latent_projs'):
        for i, proj in enumerate(model.model_to_latent_projs):
            # 如果是Sequential，找到其中的Linear层
            if isinstance(proj, torch.nn.Sequential):
                for layer in proj:
                    if isinstance(layer, torch.nn.Linear):
                        # 使用极小的初始化值
                        torch.nn.init.normal_(layer.weight, mean=0.0, std=0.001)
                        if hasattr(layer, 'bias') and layer.bias is not None:
                            torch.nn.init.zeros_(layer.bias)
            # 如果直接是Linear层
            elif isinstance(proj, torch.nn.Linear):
                torch.nn.init.normal_(proj.weight, mean=0.0, std=0.001)
                if hasattr(proj, 'bias') and proj.bias is not None:
                    torch.nn.init.zeros_(proj.bias)
    
    # 递归处理子模块
    if hasattr(model, 'module'):
        initialize_model_to_latent_projs(model.module)

model = LLMFlow(
    num_text_tokens = 10,
    dim_latent = 4,
    channel_first_latent = False,
    modality_default_shape = (14, 14),
    modality_encoder = Encoder(),
    modality_decoder = Decoder(),
    add_pos_emb = True,
    modality_num_dim = 2,
    velocity_consistency_loss_weight = 0.1,
    reconstruction_loss_weight = 0.1,
    transformer={
            'use_llama': True,  # 指定使用 Llama
            'dim': 3072,
            'model_name_or_path': '/share/project/xiaohongwang/LLM_checkpoints/Llama3.2/Llama-3.2-3B-Instruct',
            'torch_dtype': torch.float16,
            'use_gradient_checkpointing': True
        },
).cuda()

# 初始化模型中的model_to_latent_projs，避免梯度爆炸
initialize_model_to_latent_projs(model)

# 冻结model_to_latent_projs参数，避免梯度爆炸
def freeze_model_to_latent_projs(model):
    """冻结model_to_latent_projs参数，避免梯度爆炸"""
    def process_module(module):
        if hasattr(module, 'model_to_latent_projs'):
            for i, proj in enumerate(module.model_to_latent_projs):
                # 处理Sequential
                if isinstance(proj, torch.nn.Sequential):
                    for layer in proj:
                        if isinstance(layer, torch.nn.Linear):
                            layer.weight.requires_grad = False
                            if layer.bias is not None:
                                layer.bias.requires_grad = False
                
                # 处理Linear
                elif isinstance(proj, torch.nn.Linear):
                    proj.weight.requires_grad = False
                    if proj.bias is not None:
                        proj.bias.requires_grad = False
    
    # 处理主模型
    process_module(model)
    
    # 如果是DDP模型，处理内部模块
    if hasattr(model, 'module'):
        process_module(model.module)

# 冻结前5000步训练，增加冻结时间
FREEZE_STEPS = 5000
freeze_model_to_latent_projs(model)

# 修改LLMfusion的forward方法，确保所有参数都参与计算
original_forward = model.forward

def forward_wrapper(*args, **kwargs):
    # 调用原始forward方法
    result = original_forward(*args, **kwargs)
    
    # 添加一个小的正则化项，确保所有参数都参与计算
    reg_loss = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            # 使用更明确的方式确保梯度流动
            reg_loss = reg_loss + 1e-10 * (param * param).sum()
    
    # 返回原始结果加上正则化项
    if isinstance(result, torch.Tensor):
        return result + reg_loss
    else:
        # 如果结果不是张量，可能是元组或其他结构，尝试处理
        return result

# 替换forward方法
model.forward = forward_wrapper

# Wrap model with DDP - 重新启用find_unused_parameters
model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

# Create EMA model
ema_model = MemoryEfficientEMA(model.module, decay=0.999)
# ema_model = model.create_ema()

class MnistDataset(Dataset):
    def __init__(self):
        self.mnist = torchvision.datasets.MNIST(
            './data',
            download = True
        )

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, idx):
        pil, labels = self.mnist[idx]
        digit_tensor = T.PILToTensor()(pil)
        # 返回float16类型的数据
        return (digit_tensor / 255).to(torch.float16)

def cycle(iter_dl):
    while True:
        for batch in iter_dl:
            yield batch

dataset = MnistDataset()
BATCH_SIZE = 8
sampler = DistributedSampler(dataset)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)
iter_dl = cycle(dataloader)

# 使用更小的学习率，提高训练稳定性
optimizer = AdamW(model.parameters(), lr = 1e-8, eps=1e-10, weight_decay=1e-5)

# 添加学习率调度器，包括预热和衰减
from torch.optim.lr_scheduler import LambdaLR

def get_lr_schedule_fn(warmup_steps=2000, decay_steps=100000, min_lr_ratio=0.1):
    def lr_lambda(step):
        # 预热阶段
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # 衰减阶段
        decay_ratio = min_lr_ratio + (1.0 - min_lr_ratio) * (
            0.5 * (1.0 + math.cos(math.pi * min(1.0, (step - warmup_steps) / decay_steps))))
        return max(0.0, decay_ratio)
    return lr_lambda

import math
lr_scheduler = LambdaLR(optimizer, lr_lambda=get_lr_schedule_fn(warmup_steps=2000, decay_steps=50000, min_lr_ratio=0.1))

# 解冻model_to_latent_projs参数
def unfreeze_model_to_latent_projs(model):
    """解冻model_to_latent_projs参数"""
    def process_module(module):
        if hasattr(module, 'model_to_latent_projs'):
            for i, proj in enumerate(module.model_to_latent_projs):
                # 处理Sequential
                if isinstance(proj, torch.nn.Sequential):
                    for layer in proj:
                        if isinstance(layer, torch.nn.Linear):
                            layer.weight.requires_grad = True
                            if layer.bias is not None:
                                layer.bias.requires_grad = True
                
                # 处理Linear
                elif isinstance(proj, torch.nn.Linear):
                    proj.weight.requires_grad = True
                    if proj.bias is not None:
                        proj.bias.requires_grad = True
    
    # 处理主模型
    process_module(model)
    
    # 如果是DDP模型，处理内部模块
    if hasattr(model, 'module'):
        process_module(model.module)

# 添加梯度累积功能，减少更新频率
ACCUMULATION_STEPS = 4

# 添加自动恢复机制
class TrainingStateManager:
    def __init__(self, model, optimizer, ema_model, lr_scheduler, save_dir='./checkpoints'):
        self.model = model
        self.optimizer = optimizer
        self.ema_model = ema_model
        self.lr_scheduler = lr_scheduler
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True, parents=True)
        
        # 训练状态
        self.best_loss = float('inf')
        self.nan_count = 0
        self.consecutive_nan_count = 0
        self.last_save_step = 0
        
    def update(self, step, loss):
        """更新训练状态，如果需要则保存检查点"""
        # 检查loss是否为NaN或Inf
        is_valid_loss = not (torch.isnan(loss).any() or torch.isinf(loss).any())
        
        if is_valid_loss:
            loss_value = loss.item()
            self.consecutive_nan_count = 0
            
            # 如果是新的最佳loss，保存模型
            if loss_value < self.best_loss:
                self.best_loss = loss_value
                self.save_checkpoint(step, 'best')
                
                # 记录最佳loss到wandb
                if local_rank == 0:
                    wandb.log({"best_loss": self.best_loss}, step=step)
            
            # 每1000步保存一次常规检查点
            if step - self.last_save_step >= 1000:
                self.save_checkpoint(step, 'latest')
                self.last_save_step = step
        else:
            self.nan_count += 1
            self.consecutive_nan_count += 1
            
            # 记录NaN计数到wandb
            if local_rank == 0:
                wandb.log({"nan_count": self.nan_count, 
                           "consecutive_nan_count": self.consecutive_nan_count}, step=step)
            
            # 如果连续出现5次NaN，尝试从最近的检查点恢复
            if self.consecutive_nan_count >= 5:
                if local_rank == 0:
                    print(f"Warning: {self.consecutive_nan_count} consecutive NaN losses detected. Attempting to recover from checkpoint.")
                    wandb.log({"checkpoint_recovery": 1}, step=step)
                self.load_checkpoint('best')
                self.consecutive_nan_count = 0
    
    def save_checkpoint(self, step, name):
        """保存检查点"""
        if local_rank != 0:
            return
        
        checkpoint = {
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
            'best_loss': self.best_loss,
            'nan_count': self.nan_count
        }
        
        # 保存EMA模型状态
        if hasattr(self.ema_model, 'shadow_params'):
            checkpoint['ema_shadow_params'] = self.ema_model.shadow_params
        
        torch.save(checkpoint, self.save_dir / f"{name}.pt")
        print(f"Checkpoint saved at step {step} with loss {self.best_loss:.6f}")
        
        # 记录检查点保存到wandb
        # wandb.log({"checkpoint_saved": 1, "checkpoint_name": name}, step=step)
    
    def load_checkpoint(self, name):
        """加载检查点"""
        checkpoint_path = self.save_dir / f"{name}.pt"
        if not checkpoint_path.exists():
            if local_rank == 0:
                print(f"No checkpoint found at {checkpoint_path}")
            return False
        
        # 在所有进程上同步加载检查点
        dist.barrier()
        
        # 显式设置 weights_only=False，因为我们需要加载完整的检查点（包括优化器状态等）
        # 注意：这里我们信任检查点文件，因为它是由我们自己的训练过程创建的
        checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{local_rank}', weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        self.best_loss = checkpoint['best_loss']
        self.nan_count = checkpoint['nan_count']
        
        # 恢复EMA模型状态
        if 'ema_shadow_params' in checkpoint and hasattr(self.ema_model, 'shadow_params'):
            self.ema_model.shadow_params = checkpoint['ema_shadow_params']
        
        if local_rank == 0:
            print(f"Loaded checkpoint from step {checkpoint['step']} with loss {self.best_loss:.6f}")
            # 记录检查点加载到wandb
            wandb.log({"checkpoint_loaded": 1, "loaded_step": checkpoint['step']}, step=checkpoint['step'])
        return True

# 创建训练状态管理器
state_manager = TrainingStateManager(model, optimizer, ema_model, lr_scheduler)

# 添加一个辅助函数来检查哪些参数没有梯度
def print_params_without_grad(model, step):
    if local_rank == 0:
        print(f"Step {step}: Checking parameters without gradients:")
        for name, param in model.named_parameters():
            if param.requires_grad and (param.grad is None or torch.all(param.grad == 0)):
                print(f"  - {name}: No gradient")

# 修改训练循环部分
for step in range(1, 10_000 + 1):
    try:
        # 在指定步数后解冻model_to_latent_projs参数
        if step == FREEZE_STEPS:
            if local_rank == 0:
                print(f"Step {step}: Unfreezing model_to_latent_projs parameters")
                wandb.log({"unfreeze_model_to_latent_projs": 1}, step=step)
            unfreeze_model_to_latent_projs(model)
            # 在解冻参数后同步所有进程
            torch.distributed.barrier()
        
        # 获取数据并移动到当前设备，确保是float16类型
        batch = next(iter_dl).cuda(local_rank)
        
        # 将张量转换为LLMfusion期望的格式（列表形式）
        # 每个批次的样本作为单独的列表项
        batch_list = []
        for i in range(batch.shape[0]):
            # 将每个样本作为单独的模态添加到列表中
            # (0, tensor) 表示模态类型为0
            batch_list.append([(0, batch[i].to(torch.float16))])
        
        # 使用no_sync上下文管理器优化梯度累积
        context = model.no_sync if step % ACCUMULATION_STEPS != 0 else nullcontext
        
        with context():
            # 前向传播 - 不使用autocast，因为模型已经使用float16
            loss = model(batch_list)
            
            # 检查loss是否为NaN，如果是则跳过此步骤
            if torch.isnan(loss).any() or torch.isinf(loss).any():
                if local_rank == 0:
                    print(f"Warning: Loss is NaN or Inf at step {step}, skipping update")
                    wandb.log({"nan_loss_detected": 1}, step=step)
                optimizer.zero_grad()
                state_manager.update(step, torch.tensor(float('nan')))
                continue
            
            # 梯度累积 - 缩放损失
            loss = loss / ACCUMULATION_STEPS
            
            # 使用scaler进行梯度缩放，防止梯度下溢或上溢
            scaler.scale(loss).backward()
        
        # 只在累积步骤结束时更新参数
        if step % ACCUMULATION_STEPS == 0:
            # 检查梯度是否包含NaN或Inf
            if check_gradients(model, step):
                if local_rank == 0:
                    print(f"{step}: Warning - NaN or Inf gradients detected, skipping update")
                    wandb.log({"nan_gradient_detected": 1}, step=step)
                optimizer.zero_grad()
                continue
            
            # 使用更小的梯度裁剪阈值
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.001)
            
            # 使用自定义的梯度缩放函数处理异常梯度
            scale_gradients(model, max_norm=0.001)
            
            # 特别处理model_to_latent_projs的梯度
            handle_model_to_latent_projs_gradients(model, max_norm=0.0001)  # 使用更小的阈值
            
            # 使用scaler更新参数
            scaler.step(optimizer)
            scaler.update()
            
            optimizer.zero_grad()
            
            # 更新学习率
            lr_scheduler.step()
            
            # 更新EMA模型
            ema_model.update()
            
            # 在更新参数后同步所有进程
            torch.distributed.barrier()
            
            # 每100步检查一次哪些参数没有梯度
            if step % 100 == 0:
                print_params_without_grad(model, step)
        
        # 更新训练状态
        state_manager.update(step, loss * ACCUMULATION_STEPS)  # 恢复原始损失值用于记录
        
        # 只在rank 0进程打印和保存
        if local_rank == 0: #and step % 10 == 0:  # 减少打印频率
            # 安全地打印loss值
            loss_value = loss.item() * ACCUMULATION_STEPS if not (torch.isnan(loss).any() or torch.isinf(loss).any()) else "NaN/Inf"
            current_lr = optimizer.param_groups[0]['lr']
            print(f'{step}: loss={loss_value}, lr={current_lr:.8f}')
            
            # 记录到wandb
            if isinstance(loss_value, (int, float)):
                wandb.log({
                    "loss": loss_value,
                    "learning_rate": current_lr,
                    "step": step
                }, step=step)

            print('=' * 20)
            if divisible_by(step, 2):
                print(f"divisible_by at step {step}")
                try:
                    # 确保结果目录存在
                    if local_rank == 0 and not results_folder.exists():
                        print(f"Results folder does not exist, creating at {results_folder.absolute()}")
                        results_folder.mkdir(exist_ok=True, parents=True)
                    
                    # 生成图像
                    image = ema_model.generate_modality_only(batch_size = 8)
                    
                    # 只在rank 0进程保存图像
                    if local_rank == 0:
                        # 构建保存路径
                        print(f"Saving image at step {step}")
                        save_path = results_folder / f'{step}.png'
                        
                        # 保存图像
                        save_image(
                            rearrange(image, '(gh gw) 1 h w -> 1 (gh h) (gw w)', gh = 8).detach().cpu(),
                            str(save_path)
                        )
                        
                        # 检查图像是否成功保存
                        if save_path.exists():
                            print(f"Image saved successfully at {save_path.absolute()}")
                        else:
                            print(f"Failed to save image at {save_path.absolute()}")
                    
                    # 将生成的图像记录到wandb
                    if local_rank == 0:
                        wandb.log({
                            "generated_images": wandb.Image(
                                rearrange(image, '(gh gw) 1 h w -> 1 (gh h) (gw w)', gh = 8).detach().cpu(),
                                caption=f"Step {step}"
                            )
                        }, step=step)
                except Exception as e:
                    if local_rank == 0:
                        print(f"Error saving image at step {step}: {e}")
                        print(traceback.format_exc())
                        # 记录错误到wandb
                        wandb.log({"image_save_error": str(e)}, step=step)
    except Exception as e:
        # 打印详细错误信息，帮助调试
        import traceback
        print(f"Error at step {step}: {e}")
        print(traceback.format_exc())
        if local_rank == 0:
            print("Trying to continue training...")
            # 记录错误到wandb
            wandb.log({"error": str(e)}, step=step)
        continue

# 在训练结束时关闭wandb
if local_rank == 0:
    wandb.finish()
