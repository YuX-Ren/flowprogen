from shutil import rmtree
from pathlib import Path

import torch
from torch import tensor
from torch.nn import Module
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from einops import rearrange

import torchvision
import torchvision.transforms as T
from torchvision.utils import save_image

from dillm import DiLLM, print_modality_sample

rmtree('./results_dillm/train_image_only_dillmv2', ignore_errors = True)
results_folder = Path('./results_dillm/train_image_only_dillmv2')
results_folder.mkdir(exist_ok = True, parents = True)
device = torch.device('cuda:0')

# functions
SAMPLE_EVERY = 100
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
model = DiLLM(
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
        'dim': 2048,
        'model_name_or_path': '/share/project/xiaohongwang/LLM_checkpoints/Llama3.2/Llama-3.2-1B-Instruct',
        'use_gradient_checkpointing': True
    },
).to(device)

ema_model = model.create_ema()

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
        # return (digit_tensor / 255).float()
        return digit_tensor / 255.

def cycle(iter_dl):
    while True:
        for batch in iter_dl:
            yield batch

dataset = MnistDataset()

dataloader = DataLoader(dataset, batch_size = 32, shuffle = True)
iter_dl = cycle(dataloader)

optimizer = Adam(model.parameters(), lr = 8e-4)
for param_group in optimizer.param_groups:
    for param in param_group['params']:
        param.data = param.data.to(torch.float32)
# train loop
# for name, param in model.named_parameters():
#     print(f"{name}: {param.requires_grad}")  #rotary_emb.freqs: False

scheduler = CosineAnnealingLR(optimizer, T_max=100_000, eta_min=1e-6)

for step in range(1, 100_000 + 1):
    model.train()
    ema_model.to(torch.float32)
    loss = model(next(iter_dl), velocity_consistency_ema_model = ema_model)
    loss.backward()

    clip_grad_norm_(model.parameters(), 0.5)

    optimizer.step()
    optimizer.zero_grad()
    scheduler.step()

    ema_model.update()

    print(f'{step}: {loss.item():.3f}')

    if divisible_by(step, SAMPLE_EVERY):
        image = ema_model.generate_modality_only(batch_size = 64)

        save_image(
            rearrange(image, '(gh gw) 1 h w -> 1 (gh h) (gw w)', gh = 8).detach().cpu(),
            str(results_folder / f'{step}.png')
        )
