from collections import OrderedDict
from typing import Callable
import torch
import torch.nn as nn
from copy import deepcopy
from openfold.utils.tensor_utils import tensor_tree_map

def exists(val):
    return val is not None

class newExponentialMovingAverage(nn.Module):
    """
    Maintains moving averages of parameters with exponential decay

    At each step, the stored copy `copy` of each parameter `param` is
    updated as follows:

        `copy = decay * copy + (1 - decay) * param`

    where `decay` is an attribute of the ExponentialMovingAverage object.
    """

    def __init__(self, model: nn.Module, 
                 ema_model: nn.Module | Callable[[], nn.Module] | None = None,
                 decay: float = 0.9999, 
                 forward_method_names: tuple[str, ...] = ('forward_modality', 'forward', 'sample', 'generate_text_only','generate_modality_only')):
        """
        Args:
            model:
                A torch.nn.Module whose parameters are to be tracked
            decay:
                A value (usually close to 1.) by which updates are
                weighted as part of the above formula
        """
        super(newExponentialMovingAverage, self).__init__()

        clone_param = lambda t: t.clone().detach()
        self.params = tensor_tree_map(clone_param, model.state_dict())
        self.decay = decay
        self.device = next(model.parameters()).device
        self.forward_method_names = forward_method_names
        self.init_ema(ema_model, model)
    
    def init_ema(
        self,
        ema_model: nn.Module | None = None,
        model: nn.Module | None = None
    ):
        self.ema_model = ema_model

        if not exists(self.ema_model):
            try:
                self.ema_model = deepcopy(model)
            except Exception as e:
                print(f'Error: While trying to deepcopy model: {e}')
                print('Your model was not copyable. Please make sure you are not using any LazyLinear')
                exit()

        for p in self.ema_model.parameters():
            p.detach_()

        # forwarding methods

        for forward_method_name in self.forward_method_names:
            fn = getattr(self.ema_model, forward_method_name)
            setattr(self, forward_method_name, fn)

        # parameter and buffer names

        self.parameter_names = {name for name, param in self.ema_model.named_parameters() if torch.is_floating_point(param) or torch.is_complex(param)}
        self.buffer_names = {name for name, buffer in self.ema_model.named_buffers() if torch.is_floating_point(buffer) or torch.is_complex(buffer)}

    def to(self, device):
        self.params = tensor_tree_map(lambda t: t.to(device), self.params)
        self.device = device

    def _update_state_dict_(self, update, state_dict):
        with torch.no_grad():
            for k, v in update.items():
                stored = state_dict[k]
                if 'transformer' not in k:
                    continue
                if not isinstance(v, torch.Tensor):
                    self._update_state_dict_(v, stored)
                else:
                    if stored.device != v.device:
                        v = v.to(stored.device)
                    diff = stored - v
                    diff *= 1 - self.decay
                    stored -= diff

    def update(self, model: torch.nn.Module) -> None:
        """
        Updates the stored parameters using the state dict of the provided
        module. The module should have the same structure as that used to
        initialize the ExponentialMovingAverage object.
        """
        self._update_state_dict_(model.state_dict(), self.params)

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        for k in state_dict["params"].keys():
            self.params[k] = state_dict["params"][k].clone()
        self.decay = state_dict["decay"]

    def state_dict(self) -> OrderedDict:
        return OrderedDict(
            {
                "params": self.params,
                "decay": self.decay,
            }
        )