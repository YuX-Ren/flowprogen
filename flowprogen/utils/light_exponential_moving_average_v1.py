import torch
import torch.nn as nn
from copy import deepcopy
from typing import Callable
from openfold.utils.tensor_utils import tensor_tree_map

def exists(val):
    return val is not None

class LightExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float = 0.9999,
                 forward_method_names: tuple[str, ...] = ('forward_modality', 'forward', 'sample', 'generate_text_only','generate_modality_only')):
        self.model = model
        self.ema_transformer = deepcopy(self.model.transformer)
        self.decay = decay
        self.device = next(model.parameters()).device
        self.shadow_params = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
            if param.requires_grad and param.dtype in (torch.float32, torch.float16, torch.bfloat16)
        }

        for p in self.shadow_params.values():
            p.requires_grad = False
        self.forward_method_names = forward_method_names
        for forward_method_name in self.forward_method_names:
            fn = getattr(model, forward_method_name)
            setattr(self, forward_method_name, fn)

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if name not in self.shadow_params:
                continue
            shadow = self.shadow_params[name]
            shadow.data = self.decay * shadow.data + (1. - self.decay) * param.data

    def apply_to(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name in self.shadow_params:
                param.data.copy_(self.shadow_params[name])

    def to(self, device):
        for p in self.shadow_params.values():
            p.data = p.data.to(device)
