from shutil import rmtree
from pathlib import Path

import torch
from torch import tensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR
from torch.amp import GradScaler,autocast

from einops import rearrange

import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image

from dillm import DiLLM, print_modality_sample

# hf related

from datasets import load_dataset
from diffusers.models import AutoencoderKL
import torch.nn as nn
import numpy as np
import random
import os
import wandb  # 导入 wandb

# 设置默认的 torch dtype 为 float32，避免混合精度问题
# torch.set_default_dtype(torch.float32)

# 加载 VAE 模型，使用 float32 类型
vae = AutoencoderKL.from_pretrained("/share/project/xiaohongwang/LLM_checkpoints/stable-diffusion-v1-4", subfolder = "vae")
device = torch.device('cuda:0')
vae.to(device)

# 确保 VAE 模型使用 float32 类型
# def convert_module_to_f32(l):
#     if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
#         l.weight.data = l.weight.data.float()
#         if l.bias is not None:
#             l.bias.data = l.bias.data.float()

# vae.apply(convert_module_to_f32)

class Encoder(Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, image):
        # 确保输入是 float32 类型
        # image = image.to(torch.float32)
        
        with torch.no_grad():
            latent = self.vae.encode(image * 2 - 1)

        # 返回 float32 类型的结果，与模型一致
        return 0.18215 * latent.latent_dist.sample()

class Decoder(Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        # 确保输入是 float32 类型
        # latents = latents.to(torch.float32)
        
        latents = (1 / 0.18215) * latents

        with torch.no_grad():
            image = self.vae.decode(latents).sample

        # 返回 float32 类型的结果，与模型一致
        return (image / 2 + 0.5).clamp(0, 1)

# results folder

rmtree('./results_dillm/train_latent_only_dillm', ignore_errors = True)
results_folder = Path('./results_dillm/train_latent_only_dillm')
results_folder.mkdir(exist_ok = True, parents = True)

# constants

SAMPLE_EVERY = 100
# functions

def divisible_by(num, den):
    return (num % den) == 0

# encoder / decoder

# 在创建 DiLLM 模型之前，将默认数据类型设置回 float16
# torch.set_default_dtype(torch.float32)

# 添加梯度缩放器，用于混合精度训练
scaler = GradScaler()

# 设置随机种子，提高实验可重复性
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

set_seed()

# 初始化 wandb
wandb.init(
    project="dillm",  # 项目名称
    name="train_latent_only_dillm",      # 实验名称
    config={
        "learning_rate": 1e-6,
        "weight_decay": 1e-7,
        "optimizer": "AdamW",
        "scheduler": "OneCycleLR",
        "max_lr": 5e-5,
        "batch_size": 4,
        "model": "DiLLM-Llama-1B",
        "dataset": "oxford-flowers",
        "image_size": 256,
        "velocity_consistency_loss_weight": 0.01,
        "reconstruction_loss_weight": 0.01,
        "gradient_clip_threshold": 0.0001,
        "ema_decay": 0.999,
    }
)

# 创建一个 float32 版本的模型，用于训练
# torch.set_default_dtype(torch.float32)

model = DiLLM(
    num_text_tokens = 10,
    dim_latent = 4,
    channel_first_latent = True,
    modality_default_shape = (32, 32),
    modality_encoder = Encoder(vae),
    modality_decoder = Decoder(vae),
    add_pos_emb = True,
    modality_num_dim = 2,
    velocity_consistency_loss_weight = 0.1,  # 降低权重，提高稳定性
    reconstruction_loss_weight = 0.1,        # 降低权重，提高稳定性
    transformer={
        'use_llama': True,  # 指定使用 Llama
        'dim': 2048,
        'model_name_or_path': '/share/project/xiaohongwang/LLM_checkpoints/Llama3.2/Llama-3.2-1B-Instruct',
        'use_gradient_checkpointing': True
    },
).to(device)

# 初始化模型参数，提高训练稳定性
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight, gain=0.01)  # 使用较小的增益值
        if m.bias is not None:
            nn.init.zeros_(m.bias)

# 只初始化非预训练的部分
for name, module in model.named_children():
    if name != 'transformer':  # 不初始化预训练的 transformer
        module.apply(init_weights)

# 创建 EMA 模型
ema_model = model.create_ema(0.9)  # 使用更大的 EMA 衰减率，提高稳定性

class FlowersDataset(Dataset):
    def __init__(self, image_size):
        self.ds = load_dataset("nelorth/oxford-flowers")['train']

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.PILToTensor()
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        pil = self.ds[idx]['image']
        tensor = self.transform(pil)
        return tensor / 255.
        # return (tensor / 255.).to(torch.float32)

def cycle(iter_dl):
    while True:
        for batch in iter_dl:
            yield batch

dataset = FlowersDataset(256)

dataloader = DataLoader(dataset, batch_size = 4, shuffle = True)

iter_dl = cycle(dataloader)

# 使用 AdamW 优化器，并设置较小的学习率和权重衰减
optimizer = Adam(model.parameters(), lr = 2e-5)
# optimizer = AdamW(model.parameters(), lr = 1e-3, weight_decay=1e-2, eps=1e-8)
for param_group in optimizer.param_groups:
    for param in param_group['params']:
        param.data = param.data.to(torch.float32)
# for name, param in model.named_parameters():
#     print(f"{name}: {param.dtype}")

for name, param in model.named_parameters():
    print(f"{name}: {param.requires_grad}")


first_param = next(iter(optimizer.param_groups[0]['params']))
print(f"Optimizer first param dtype: {first_param.dtype}")
# 添加学习率调度器，包括预热阶段
# scheduler = OneCycleLR(
#     optimizer, 
#     max_lr=5e-5,  # 最大学习率
#     total_steps=100000,
#     pct_start=0.1,  # 预热阶段占总步数的比例
#     div_factor=25.0,  # 初始学习率 = max_lr / div_factor
#     final_div_factor=10000.0,  # 最终学习率 = max_lr / final_div_factor
#     anneal_strategy='cos'
# )
scheduler = CosineAnnealingLR(optimizer, T_max=100000, eta_min=1e-6)
# 添加梯度检查函数
def check_gradients(model):
    """检查模型梯度是否包含NaN或Inf值"""
    has_nan = False
    has_inf = False
    for name, param in model.named_parameters():
        if param.grad is not None:
            if torch.isnan(param.grad).any():
                has_nan = True
                print(f"NaN gradient in {name}")
                # 将 NaN 梯度替换为 0
                param.grad[torch.isnan(param.grad)] = 0.0
            if torch.isinf(param.grad).any():
                has_inf = True
                print(f"Inf gradient in {name}")
                # 将 Inf 梯度替换为 0
                param.grad[torch.isinf(param.grad)] = 0.0
    return has_nan or has_inf

# 添加检查点保存函数
def save_checkpoint(model, optimizer, scheduler, step, loss, path='./checkpoints'):
    import os
    os.makedirs(path, exist_ok=True)
    
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'step': step,
        'loss': loss
    }
    
    checkpoint_path = f'{path}/checkpoint_{step}.pt'
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at step {step}")

# train loop
accumulated_loss = 0
valid_steps = 0
best_loss = float('inf')
patience = 0
max_patience = 5  # 早停的耐心值

# 创建一个字典来跟踪训练指标
metrics = {
    "train_loss": [],
    "learning_rate": [],
    "nan_grad_count": 0,
    "inf_grad_count": 0,
    "best_loss": float('inf'),
}

# for step in range(1, 100_000 + 1):
#     model.train()
    
#     try:
#         for _ in range(4):  # 进一步减少累积步数，提高稳定性
#             # 获取数据并确保是 float32 类型（与模型一致）
#             batch = next(iter_dl)
#             batch = batch.to(torch.float32)  # 转换为 float32
#             batch = batch.to(device)  # 确保数据在正确的设备上
            
#             # 前向传播
#             with autocast('cuda', enabled=True):  # 启用自动混合精度
#                 loss = model.forward_modality(batch)
                
#                 # 检查损失是否为 NaN
#                 if torch.isnan(loss).any() or torch.isinf(loss).any():
#                     print(f"Warning: Loss is NaN or Inf at step {step}, skipping batch")
#                     # 记录 NaN/Inf 损失
#                     wandb.log({"nan_loss_detected": 1}, step=step)
#                     continue
                
#                 # 反向传播
#                 loss = loss / 4  # 使用正确的累积步数
            
#             # 使用梯度缩放器进行反向传播
#             scaler.scale(loss).backward()
            
#             # 累积有效的损失
#             accumulated_loss += loss.item()
#             valid_steps += 1

#         # 检查梯度是否包含 NaN 或 Inf
#         has_bad_grads = check_gradients(model)
#         if has_bad_grads:
#             print(f"Warning: NaN or Inf gradients detected at step {step}, some gradients were zeroed")
#             # 记录 NaN/Inf 梯度
#             wandb.log({"nan_grad_detected": 1}, step=step)
#             metrics["nan_grad_count"] += 1
        
#         # 梯度裁剪 - 使用更小的阈值
#         scaler.unscale_(optimizer)
#         torch.nn.utils.clip_grad_norm_(model.parameters(), 0.0001)  # 使用非常小的阈值
        
#         # 更新参数
#         scaler.step(optimizer)
#         scaler.update()
#         optimizer.zero_grad()
        
#         # 更新学习率
#         scheduler.step()
        
#         # 更新 EMA 模型
#         ema_model.update()
        
#         # 打印损失
#         if valid_steps > 0:
#             avg_loss = accumulated_loss / valid_steps
#             current_lr = scheduler.get_last_lr()[0]
#             print(f'{step}: {avg_loss:.6f} (lr: {current_lr:.8f})')
            
#             # 记录指标到 wandb
#             wandb.log({
#                 "train_loss": avg_loss,
#                 "learning_rate": current_lr,
#                 "step": step,
#                 "nan_grad_count": metrics["nan_grad_count"],
#                 "inf_grad_count": metrics["inf_grad_count"],
#             }, step=step)
            
#             # 更新指标字典
#             metrics["train_loss"].append(avg_loss)
#             metrics["learning_rate"].append(current_lr)
            
#             # 检查是否需要保存检查点
#             if avg_loss < best_loss:
#                 best_loss = avg_loss
#                 metrics["best_loss"] = best_loss
#                 wandb.log({"best_loss": best_loss}, step=step)
#                 patience = 0
#                 if step % 100 == 0:  # 每 100 步保存一次最佳检查点
#                     save_checkpoint(model, optimizer, scheduler, step, avg_loss)
#             else:
#                 patience += 1
                
#             accumulated_loss = 0
#             valid_steps = 0
#         else:
#             print(f'{step}: No valid steps')
#             wandb.log({"no_valid_steps": 1}, step=step)
            
#         # 早停检查
#         # if patience >= max_patience and step > 1000:
#         #     print(f"Early stopping triggered at step {step}")
#         #     wandb.log({"early_stopping": 1}, step=step)
#         #     # 保存最终检查点
#         #     save_checkpoint(model, optimizer, scheduler, step, avg_loss, path='./checkpoints_final')
#         #     break

#         # 生成样本
#         if divisible_by(step, SAMPLE_EVERY):
#             # 确保在评估模式下生成
#             model.eval()
#             with torch.no_grad():
#                 try:
#                     # 使用原始模型而不是 EMA 模型生成图像
#                     ema_model = ema_model.to(torch.float32)
#                     image = ema_model.generate_modality_only(batch_size = 4)  # 减少批量大小
                    
#                     # 保存图像
#                     image_path = str(results_folder / f'{step}.png')
#                     save_image(
#                         rearrange(image, '(gh gw) c h w -> c (gh h) (gw w)', gh = 2).detach().cpu(),
#                         image_path
#                     )
#                     print(f"Successfully saved image at step {step}")
                    
#                     # 将图像记录到 wandb
#                     wandb.log({
#                         "generated_images": wandb.Image(
#                             image_path,
#                             caption=f"Step {step}"
#                         )
#                     }, step=step)
#                 except Exception as e:
#                     print(f"Error generating image at step {step}: {e}")
#                     wandb.log({"image_generation_error": 1}, step=step)
#                     import traceback
#                     traceback.print_exc()
#             # 恢复训练模式
#             model.train()
    
#     except Exception as e:
#         print(f"Error at step {step}: {e}")
#         wandb.log({"training_error": 1}, step=step)
#         import traceback
#         traceback.print_exc()
#         continue

# # 训练结束，关闭 wandb
# wandb.finish()
for step in range(1, 100_000 + 1):
    model.train()
    for _ in range(4):
        loss = model.forward_modality(next(iter_dl))
        (loss / 4).backward()
    has_bad_grads = check_gradients(model)
    if has_bad_grads:
        print(f"Warning: NaN or Inf gradients detected at step {step}, some gradients were zeroed")
    
    clip_grad_norm_(model.parameters(), 0.5)

    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

    ema_model.update()

    print(f'{step}: {loss.item():.3f}')

    if divisible_by(step, SAMPLE_EVERY):
        ema_model.to(torch.float32)
        image = ema_model.generate_modality_only(batch_size = 4)

        save_image(
            rearrange(image, '(gh gw) c h w -> c (gh h) (gw w)', gh = 2).detach().cpu(),
            str(results_folder / f'{step}.png')
        )
        print(f"Successfully saved image at step {step}")