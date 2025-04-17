import torch
import torch.nn as nn
from copy import deepcopy
from typing import Callable
from openfold.utils.tensor_utils import tensor_tree_map

def exists(val):
    return val is not None

class LightExponentialMovingAverage:
    def __init__(self, model: nn.Module, decay: float = 0.9999,
                ignore_names: set[str] = set(),
                ignore_startswith_names: set[str] = set(),
                ):
        self.model = model
        self.ema_transformer = deepcopy(model.transformer)
        self.decay = decay
        self.device = next(model.parameters()).device

        self.shadow_params = {
            name: param.clone().detach().to(self.device)
            for name, param in model.named_parameters() 
            if param.requires_grad
        }

        for p in self.shadow_params.values():
            p.requires_grad = False
        
        self.ignore_names = ignore_names
        self.ignore_startswith_names = ignore_startswith_names

    def get_params_iter(self, model):
        for name, param in model.named_parameters():
            if name in self.ignore_names:
                continue
            if any(name.startswith(prefix) for prefix in self.ignore_startswith_names):
                continue
            yield name, param

    @torch.no_grad()
    def update(self):
        for name, param in self.get_params_iter(self.model):
            if name not in self.shadow_params:
                continue
            self.shadow_params[name].data = self.decay * self.shadow_params[name].data + (1. - self.decay) * param.data.to(self.device)

    def apply_to(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name in self.ignore_names or any(name.startswith(prefix) for prefix in self.ignore_startswith_names):
                continue
            if name in self.shadow_params:
                param.data.copy_(self.shadow_params[name].to(self.device))

    def to(self, device):
        self.shadow_params = tensor_tree_map(lambda t: t.to(device), self.shadow_params)
        self.ema_transformer = self.ema_transformer.to(device)
        self.device = device
