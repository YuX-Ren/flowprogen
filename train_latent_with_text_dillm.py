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
from torch.amp import autocast, GradScaler

from einops import rearrange

import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image

from dillm import LLMfusion, print_modality_sample

# hf related

from datasets import load_dataset
from diffusers.models import AutoencoderKL

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
            image = image.half()
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

rmtree('./results/train_latent_with_text_dillm', ignore_errors = True)
results_folder = Path('./results/train_latent_with_text_dillm')
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
def custom_collate_fn(batch):
    texts, images = zip(*batch)
    
    # Find the maximum length of text tensors
    max_len = max(len(text) for text in texts)
    
    # Pad all text tensors to the same length
    padded_texts = []
    for text in texts:
        if len(text) < max_len:
            # Create a padded tensor
            padded = torch.zeros(max_len, dtype=text.dtype)
            padded[:len(text)] = text
            padded_texts.append(padded)
        else:
            padded_texts.append(text)
    
    # Stack the padded texts and images
    text_batch = torch.stack(padded_texts)
    image_batch = torch.stack(images)
    
    return text_batch, image_batch

# encoder / decoder

# Create a tuple of encoders and decoders
# The first element (index 0) is for images, the second (index 1) is for text
# image_encoder = Encoder(vae)
# text_encoder = TextIdentityEncoder()
# modality_encoders = (image_encoder, text_encoder)

# image_decoder = Decoder(vae)
# text_decoder = TextIdentityDecoder()
# modality_decoders = (image_decoder, text_decoder)

model = LLMfusion(
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

dataset = FlowersDataset(512)

# Use custom collate function instead of model.create_dataloader
dataloader = model.create_dataloader(dataset, batch_size = 1, shuffle = True)

iter_dl = cycle(dataloader)

# optimizer = Adam(model.parameters(), lr = 8e-4)
optimizer = AdamW(model.parameters(), lr = 1e-6, weight_decay=1e-7, eps=1e-8)
for param_group in optimizer.param_groups:
    for param in param_group['params']:
        param.data = param.data.to(torch.float32)

# Initialize gradient scaler for mixed precision training
scaler = GradScaler()

# train loop

for step in range(1, 100_000 + 1):
    optimizer.zero_grad()  # Zero gradients at the start of each step
    
    for _ in range(4):
        # Use mixed precision training with float16
        with autocast('cuda', dtype=torch.float16):
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
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
    
    # Optimizer step with scaler
    scaler.step(optimizer)
    scaler.update()

    ema_model.update()

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
            text_tensor = text_tensor[text_tensor < 256]

            text = decode_tokens(text_tensor)
            filename = str(results_folder / f'{step}.{text}.png')

            save_image(
                image.detach().cpu(),
                filename
            )
            
            # Clear memory after saving
            del image
            torch.cuda.empty_cache()
        
        # Clear memory after sampling
        torch.cuda.empty_cache()
        model.train()
