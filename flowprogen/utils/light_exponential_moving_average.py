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
        clone_param = lambda t: t.clone().detach()
        self.params = tensor_tree_map(clone_param, self.ema_transformer.state_dict())
        self.decay = decay
        self.device = next(model.parameters()).device

        for name, para in self.ema_transformer.named_parameters():
            para.requires_grad = False
            para.data = para.data.to(self.device)
        
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
        for (_, ma_params), (_, current_params) in zip(self.get_params_iter(self.ema_transformer), self.get_params_iter(self.model)):
            ma_params.data = self.decay * ma_params.data + (1. - self.decay) * current_params.data.to(self.device)

    def apply_to(self, model: nn.Module):
        for name, param in model.named_parameters():
            if name in self.ema_transformer.state_dict():
                param.data.copy_(self.ema_transformer.state_dict()[name].to(self.device))

    def to(self, device):
        self.params = tensor_tree_map(lambda t: t.to(device), self.params)
        self.device = device
        self.ema_transformer = self.ema_transformer.to(device)
