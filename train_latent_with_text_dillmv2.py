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

import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image

from llmflow import LLMFlow, print_modality_sample

import json
import wandb
from PIL import Image
# hf related
from datasets import load_dataset
from diffusers.models import AutoencoderKL


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
    name="train_latent_with_text_dillm",
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

class Encoder(Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae

    def forward(self, image):
        with torch.no_grad():
            # Convert to float16 for VAE
            # image = image.half()
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

rmtree('./results_llmflow/train_latent_with_text_dillmv2-cc3m', ignore_errors = True)
results_folder = Path('./results_llmflow/train_latent_with_text_dillmv2-cc3m')
results_folder.mkdir(exist_ok = True, parents = True)

# constants

SAMPLE_EVERY = 100

with open("./data/flowers/labels.txt", "r") as file:
    content = file.read()

LABELS_TEXT = content.split('\n')

# functions

def divisible_by(num, den):
    return (num % den) == 0

def decode_token(token):
    return str(chr(max(32, token)))

def decode_tokens(tokens: Tensor) -> str:
    return "".join(list(map(decode_token, tokens.tolist())))

def encode_tokens(str: str) -> Tensor:
    return tensor([*bytes(str, 'UTF-8')])

# Custom collate function to handle different text tensor sizes
# def custom_collate_fn(batch):
#     texts, images = zip(*batch)
    
#     # Find the maximum length of text tensors
#     max_len = max(len(text) for text in texts)
    
#     # Pad all text tensors to the same length
#     padded_texts = []
#     for text in texts:
#         if len(text) < max_len:
#             # Create a padded tensor
#             padded = torch.zeros(max_len, dtype=text.dtype)
#             padded[:len(text)] = text
#             padded_texts.append(padded)
#         else:
#             padded_texts.append(text)
    
#     # Stack the padded texts and images
#     text_batch = torch.stack(padded_texts)
#     image_batch = torch.stack(images)
    
#     return text_batch, image_batch

# encoder / decoder

# Create a tuple of encoders and decoders
# The first element (index 0) is for images, the second (index 1) is for text
# image_encoder = Encoder(vae)
# text_encoder = TextIdentityEncoder()
# modality_encoders = (image_encoder, text_encoder)

# image_decoder = Decoder(vae)
# text_decoder = TextIdentityDecoder()
# modality_decoders = (image_decoder, text_decoder)

model = LLMFlow(
    num_text_tokens = 256,
    dim_latent = 4,
    channel_first_latent = True,
    modality_default_shape = (8, 8),
    modality_encoder = Encoder(vae),
    modality_decoder = Decoder(vae),
    pre_post_transformer_enc_dec = (
        nn.Conv2d(4, 2048, 3, 2, 1),  # Keep 2048 for LLM compatibility
        nn.ConvTranspose2d(2048, 4, 3, 2, 1, output_padding = 1),
    ),
    add_pos_emb = False,
    modality_num_dim = 2,
    fallback_to_default_shape_if_invalid = True,
    reconstruction_loss_weight = 0.1,
    transformer={
        'use_llama': True,
        'dim': 2048,  # Keep 2048 as required
        'model_name_or_path': '/share/project/xiaohongwang/LLM_checkpoints/Llama3.2/Llama-3.2-1B-Instruct',
        'torch_dtype': torch.float16,
        'use_gradient_checkpointing': True
    },
).to(device)

ema_model = model.create_ema(0.9)

class CC3MDataset(Dataset):
    def __init__(self, image_size):
        # self.ds = load_dataset("imagefolder", data_dir = "/share/project/public_datasets/public_natural_images/CC3M/conceptual-captions/val_image")['train']
        with open("/share/project/public_datasets/public_natural_images/CC3M/conceptual-captions/val_label.json", 'r') as f:
            self.content = [json.loads(line) for line in f]

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.PILToTensor(),
            T.Lambda(lambda t: t / 255.)
        ])

    def __len__(self):
        return len(self.content)

    def __getitem__(self, idx):
        content_info = self.content[idx]
        image_path = os.path.join("/share/project/public_datasets/public_natural_images/CC3M/conceptual-captions/val_image", os.path.basename(content_info['image'].split("@/")[-1]))  # 获取文件名
        pil = Image.open(image_path).convert("RGB")
        labels_text = " ".join(content_info['caption'])
        tensor = self.transform(pil)
        return encode_tokens(labels_text), tensor
    

class FlowersDataset(Dataset):
    def __init__(self, image_size):
        self.ds = load_dataset("nelorth/oxford-flowers")['train']

        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.PILToTensor(),
            T.Lambda(lambda t: t / 255.)
        ])

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        sample = self.ds[idx]
        pil = sample['image']

        labels_int = sample['label']
        labels_text = LABELS_TEXT[labels_int]

        tensor = self.transform(pil)
        return encode_tokens(labels_text), tensor

def cycle(iter_dl):
    while True:
        for batch in iter_dl:
            yield batch

# dataset = FlowersDataset(512)
dataset = CC3MDataset(512)

# Use custom collate function instead of model.create_dataloader
dataloader = model.create_dataloader(dataset, batch_size = 2, shuffle = True)

iter_dl = cycle(dataloader)

optimizer = AdamW(model.parameters(), lr = 2e-4)
for param_group in optimizer.param_groups:
    for param in param_group['params']:
        param.data = param.data.to(torch.float32)

# Initialize gradient scaler for mixed precision training
scaler = GradScaler()


# train loop
scheduler = CosineAnnealingLR(optimizer, T_max=100_000, eta_min=1e-6)

for step in range(1, 10_000 + 1):
    model.train()
    
    for _ in range(4):
        # Use mixed precision training with float16
        with autocast('cuda'):
            loss = model.forward(next(iter_dl))
            loss = loss / 4
        
        # Scale loss and backpropagate
        scaler.scale(loss).backward()
        
    # Clear VAE cache to save memory
    if hasattr(vae, 'clear_cache'):
        vae.clear_cache()
    
    # Clear CUDA cache more frequently
    if step % 25 == 0:  # Even more frequent cache clearing
        torch.cuda.empty_cache()

    # Clip gradients
    scaler.unscale_(optimizer)
    clip_grad_norm_(model.parameters(), 0.5)
    
    # Optimizer step with scaler
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad()
    scheduler.step()

    ema_model.update()

    wandb.log({"loss": loss.item()}, step=step)
    print(f'{step}: {loss.item():.3f}')

    if divisible_by(step, SAMPLE_EVERY):
        # Move to eval mode for sampling
        model.eval()
        with torch.no_grad():
            # Clear memory before sampling
            torch.cuda.empty_cache()
            # for name, param in ema_model.named_parameters():
            #     print(f"{name}: {param.dtype}")
            # Use smaller batch size for sampling
            ema_model.to(torch.float32)
            sample = ema_model.sample()
            print_modality_sample(sample)

            if len(sample) < 3:
                continue

            text_tensor, maybe_image, *_ = sample

            if not isinstance(maybe_image, tuple):
                continue

            _, image = maybe_image
            print(image.shape)
            text_tensor = text_tensor[text_tensor < 256]

            text = decode_tokens(text_tensor)
            print(f'{step}: {text}')
            filename = str(results_folder / f'{step}.png')

            save_image(
                image.detach().cpu(),
                filename
            )
            print(f"Saved image at step {step}")
            # wandb.log({"image": wandb.Image(image.detach().cpu(), caption=text)}, step=step)
            
            # Clear memory after saving
            del image
            torch.cuda.empty_cache()
        
        # Clear memory after sampling
        torch.cuda.empty_cache()
