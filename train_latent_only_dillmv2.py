from shutil import rmtree
from pathlib import Path

import torch
from torch import tensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, AdamW
from torch.amp import GradScaler,autocast
from einops import rearrange

import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image

from dillm import LLMfusion, print_modality_sample

# hf related

from datasets import load_dataset
from diffusers.models import AutoencoderKL

vae = AutoencoderKL.from_pretrained("/share/project/xiaohongwang/LLM_checkpoints/stable-diffusion-v1-4", subfolder = "vae")
device = torch.device('cuda:0')
vae.to(device)

class Encoder(Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, image):
        with torch.no_grad():
            latent = self.vae.encode(image * 2 - 1)

        return 0.18215 * latent.latent_dist.sample()

class Decoder(Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, latents):
        latents = (1 / 0.18215) * latents

        with torch.no_grad():
            image = self.vae.decode(latents).sample

        return (image / 2 + 0.5).clamp(0, 1)

# results folder

rmtree('./results/train_latent_only_dillmv2', ignore_errors = True)
results_folder = Path('./results/train_latent_only_dillmv2')
results_folder.mkdir(exist_ok = True, parents = True)

# constants

SAMPLE_EVERY = 50

# functions

def divisible_by(num, den):
    return (num % den) == 0

# encoder / decoder

model = LLMfusion(
    num_text_tokens = 10,
    dim_latent = 4,
    channel_first_latent = True,
    modality_default_shape = (32, 32),
    modality_encoder = Encoder(vae),
    modality_decoder = Decoder(vae),
    add_pos_emb = True,
    modality_num_dim = 2,
    velocity_consistency_loss_weight = 0.1,
    reconstruction_loss_weight = 0.1,
    fallback_to_default_shape_if_invalid = True,
    transformer={
        'use_llama': True,
        'dim': 2048,  # Keep 2048 as required
        'model_name_or_path': '/share/project/xiaohongwang/LLM_checkpoints/Llama3.2/Llama-3.2-1B-Instruct',
        'torch_dtype': torch.float16,
        'use_gradient_checkpointing': True
    },
).to(device)

ema_model = model.create_ema(0.9)

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

def cycle(iter_dl):
    while True:
        for batch in iter_dl:
            yield batch

dataset = FlowersDataset(256)

dataloader = DataLoader(dataset, batch_size = 4, shuffle = True)

iter_dl = cycle(dataloader)

optimizer = AdamW(model.parameters(), lr = 8e-4, weight_decay=1e-2, eps=1e-8)
for param_group in optimizer.param_groups:
    for param in param_group['params']:
        param.data = param.data.to(torch.float32)
scaler = GradScaler()
# train loop

for step in range(1, 100_000 + 1):

    for _ in range(4):
        with autocast('cuda', dtype=torch.float16):
            loss = model.forward_modality(next(iter_dl))
            loss = loss / 4
        
        scaler.scale(loss).backward()
    # Clear VAE cache to save memory
    if hasattr(vae, 'clear_cache'):
        vae.clear_cache()
    
    # Clear CUDA cache more frequently
    if step % 25 == 0:  # Even more frequent cache clearing
        torch.cuda.empty_cache()

    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

    scaler.step(optimizer)
    scaler.update()

    ema_model.update()

    print(f'{step}: {loss.item():.3f}')

    if divisible_by(step, SAMPLE_EVERY):
        model.eval()
        with torch.no_grad():
            torch.cuda.empty_cache()
            ema_model.to(torch.float32)
            image = ema_model.generate_modality_only(batch_size = 4)

        save_image(
            rearrange(image, '(gh gw) c h w -> c (gh h) (gw w)', gh = 2).detach().cpu(),
            str(results_folder / f'{step}.png')
        )
