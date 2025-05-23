# type: ignore

from __future__ import annotations

"""
global ein notation

b - batch
t - one modality type
m - separate modality instance
n - sequence
d - dimension
l - logits (text)
i, j - sequence (row, col)
p - positions
s - residual streams
"""

import os
import math
from collections import defaultdict

from random import randrange
from itertools import count
from functools import partial, wraps, cache
from typing import NamedTuple, Callable, Literal, Dict

import torch
import torch.nn.functional as F
from torch import nn, Tensor, tensor, is_tensor, cat, stack
from torch.nn import Module, ModuleList, Linear

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.utils._pytree import tree_map, tree_flatten, tree_unflatten

from torchdiffeq import odeint

import einx
from einops.layers.torch import Rearrange
from einops import rearrange, repeat, reduce, einsum, pack, unpack

from ema_pytorch import EMA

from axial_positional_embedding import ContinuousAxialPositionalEmbedding

from rotary_embedding_torch import RotaryEmbedding, apply_rotary_emb

from hyper_connections import HyperConnections

from tqdm import tqdm
from loguru import logger

# 添加 transformers 库导入
from transformers import AutoModelForCausalLM, AutoConfig

pad_sequence = partial(pad_sequence, batch_first = True)

# tensor typing

from typing import Annotated, Union
import jaxtyping
from jaxtyping import jaxtyped
from jaxtyping import Int, Float 
from beartype import beartype
from beartype.door import is_bearable

class TorchTyping:
    def __init__(self, abstract_dtype):
        self.abstract_dtype = abstract_dtype

    def __getitem__(self, shapes: str):
        return self.abstract_dtype[Tensor, shapes]

Float = TorchTyping(jaxtyping.Float)
Int   = TorchTyping(jaxtyping.Int)
Bool  = TorchTyping(jaxtyping.Bool)

# maybe flex attention

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask

    if torch.cuda.is_available():
        flex_attention = torch.compile(flex_attention)

except ImportError:
    flex_attention = None

# types

Scalar = Float['']

ModalitySample = list[Int[''] | Int['_'] | Float['...'] | tuple[int, Float['...']]]
# ModalitySample = list[
#     Union[
#         Int[''],
#         Int['_'],
#         Float['...'],
#         tuple[int, Float['...']]
#     ]
# ]

ModalityTokenTransform = str | Callable | None

RawModalityPositions = list[list[tuple[int, int, int]]]

GetPredFlows = dict[int, list[Callable[[Tensor], Tensor]]]

class LossBreakdown(NamedTuple):
    total: Scalar
    text: Scalar
    flow: list[Scalar]
    velocity: list[Scalar] | None = None
    recon: list[Scalar] | None = None

class ModalityInfo(NamedTuple):
    encoder: Module | None
    decoder: Module | None
    latent_to_model: Module
    model_to_latent: Module
    add_pos_emb: bool
    pos_emb_mlp: Module | None
    num_dim: int | None
    dim_latent: int
    default_shape: tuple[int, ...]
    som_id: int
    eom_id: int
    to_shape_fn: Callable | None
    channel_first_latent: bool
    modality_type: int

# helper functions

def exists(v):
    return v is not None

def default(v, d):
    return v if exists(v) else d

def identity(t):
    return t

def always(val):
    def inner(*args, **kwargs):
        return val
    return inner

def first(it):
    return it[0]

def join(arr, delimiter = ''):
    return delimiter.join(arr)

def divisible_by(num, den):
    return (num % den) == 0

def cast_tuple(t, length = 1):
    return t if isinstance(t, tuple) else ((t,) * length)

def tree_map_tensor(sample, fn: Callable):
    return tree_map(lambda t: t if not is_tensor(t) else fn(t), sample)

def add_temp_batch_dim(fn: Callable):
    @wraps(fn)
    def inner(t: Tensor, *args, **kwargs) -> Tensor:
        t = rearrange(t, '... -> 1 ...')
        out = fn(t, *args, **kwargs)
        out = rearrange(out, '1 ... -> ...')
        return out
    return inner

def pack_with_inverse(t, pattern):
    packed, packed_shape = pack(t, pattern)

    def inverse(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        return unpack(out, packed_shape, inv_pattern)

    return packed, inverse

def pack_one_with_inverse(t, pattern):
    packed, packed_shape = pack([t], pattern)

    def inverse(out, inv_pattern = None):
        inv_pattern = default(inv_pattern, pattern)
        return unpack(out, packed_shape, inv_pattern)[0]

    return packed, inverse

def eval_decorator(fn):
    def inner(self, *args, **kwargs):
        was_training = self.training
        self.eval()
        out = fn(self, *args, **kwargs)
        self.train(was_training)
        return out
    return inner

# maybe typecheck

typecheck = jaxtyped(typechecker = beartype) if os.environ.get('TYPECHECK', '').lower() in ('1', 'true') else identity

# default function for constituting modality shape from string

def default_to_modality_shape_fn(maybe_shape_str) -> tuple[int, ...]:
    return tuple([*map(int, maybe_shape_str.split(','))])

# default function for translating modality length to times (noise level, where 0 is highest noise)

def random_modality_length_to_time_fn(num_modalities: Int['b']) -> Float['b m']:
    batch = num_modalities.shape[0]
    device = num_modalities.device
    total_modalities = modality_length.amax().item()
    return torch.rand((batch, total_modalities), device = device)

def default_modality_length_to_time_fn(num_modalities: Int['b']) -> Float['b m']:
    batch, device = num_modalities.shape[0], num_modalities.device
    total_modalities = num_modalities.amax().item()

    if total_modalities == 0:
        return torch.empty((batch, 0), device = device, dtype = torch.float)

    rand_num_modalities = torch.floor(torch.rand_like(num_modalities.float()) * num_modalities)
    seq = torch.arange(total_modalities, device = device)

    prev_decoded_modality = einx.less('m, b -> b m', seq, rand_num_modalities)
    curr_modality_rand_time = torch.rand_like(num_modalities.float())

    # in paper, they fix previous decoded modalities to 500 / 1000 steps for discrete ddpm, here using flow matching with times 0 - 1 so corresponds to 0.5
    return einx.where('b m, , b -> b m', prev_decoded_modality, 0.5, curr_modality_rand_time)

# pretty print

def concat_contiguous_text(
    modality_sample: ModalitySample
) -> ModalitySample:
    """ within a modality sample, any two tensors of type int / long will be concatted together if next to each other, so all text is followed by a modality, and all modality followed by text """

    output = []

    for modality in modality_sample:
        if (
            len(output) > 0 and
            is_tensor(output[-1]) and is_tensor(modality) and
            output[-1].dtype == modality.dtype and
            modality.dtype in (torch.int, torch.long)
        ):
            packed_text, _ = pack((output[-1], modality), '*')
            output[-1] = packed_text

        else:
            output.append(modality)

    return output

def print_modality_sample(
    modality_sample: ModalitySample
):
    output = []

    for sample in modality_sample:
        if isinstance(sample, tuple):
            modality_type, sample = sample
            output.append((f'modality:{modality_type}', sample.shape))
        elif sample.dtype in (torch.int, torch.long):
            output.append(('text', sample.shape))
        else:
            output.append(('modality', sample.shape))

    logger.info(output)

# character based tokenizer

def char_tokenize(
    text: str,
    device = None,
    offset = 0
) -> Tensor:
    return tensor([*map(ord, text)], device = device) + offset

def decode_chars(
    t: Tensor,
    offset = 0,
) -> str:
    byte_list = (t - offset).clamp(min = 0, max = 127).tolist()
    return ''.join([*map(chr, byte_list)])

def get_tokens_since_rightmost_id(
    t: Tensor,
    rightmost_id: int
) -> Tensor:
    """
    ex. [9] [2] [8] [4] [7]
    2 would return [8] [4] [7]
    """

    mask = t == rightmost_id

    if not mask.any():
        return t[0:0] # return empty tensor if no id found

    reverse_cumsum = mask.flip(dims = (0,)).cumsum(dim = 0).flip(dims = (0,))
    after_right_mask = reverse_cumsum == 0
    return t[after_right_mask]

# tensor helpers

def l2norm(t):
    return F.normalize(t, dim = -1)

def softclamp(t, value = 50.):
    return (t / value).tanh() * value

def max_neg_value(t):
    return -torch.finfo(t.dtype).max

def append_dims(t, ndims):
    return t.reshape(*t.shape, *((1,) * ndims))

def is_empty(t):
    return t.numel() == 0

def log(t, eps = 1e-20):
    return torch.log(t.clamp(min = eps))

def gumbel_noise(t):
    noise = torch.rand_like(t)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1, keepdim = True):
    noise = gumbel_noise(t) * int(temperature > 0)
    return (t / temperature + noise).argmax(dim = dim, keepdim = keepdim)

# dataloader related

def collate_fn(data):
    return [*map(list, data)]

@typecheck
def create_dataloader(dataset: Dataset, **kwargs) -> DataLoader:
    return DataLoader(dataset, collate_fn = collate_fn, **kwargs)

# flex attention mask construction
# https://pytorch.org/blog/flexattention/

def causal(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

def modality(offset, length):

    def mask_fn(b, h, q_idx, kv_idx):
        return (q_idx >= offset) & (kv_idx < (offset + length))

    return mask_fn

def transfusion_attn_mask(modalities: Int['b m 3']):
    modalities = modalities.long()

    def mask_mod(b, h, q_idx, kv_idx):
        mask = causal(b, h, q_idx, kv_idx)

        modality_batch = modalities[b]

        for _, offset, length in modality_batch:
            mask = mask | modality(offset, length)(b, h, q_idx, kv_idx)

        return mask

    return mask_mod

def softcap_score_mod(softcap):
    def inner(score, b, h, q_idx, kv_idx):
        score = score / softcap
        score = torch.tanh(score)
        score = score * softcap
        return score
    return inner

# converting a raw list of modality offsets and lengths to tensor

@typecheck
def modality_positions_to_tensor(
    modalities: RawModalityPositions,
    pad_value = 0,
    device = None
) -> Int['b m 2'] | Int['b m 3']:

    modalities: list[Tensor] = [tensor(modality, device = device) for modality in modalities]
    modalities = pad_sequence(modalities, padding_value = pad_value)

    if modalities.ndim == 2:
        modalities = modalities.reshape(*modalities.shape, 3)

    return modalities.long()

# sanitizing modalities tensor, making sure it is ordered

@typecheck
def order_modality_positions_by_seq_offset(
    modalities: Int['b m 3']
) -> tuple[Int['b m 3'], Int['b m']]:

    modality_type, offsets, lengths = modalities.unbind(dim = -1)

    no_modality_mask = lengths <= 0 # there may be uneven number of modalities per batch sample
    offsets_to_sort = offsets.masked_fill(no_modality_mask, 1e10)
    _, sorted_indices = offsets_to_sort.sort(dim = -1)

    # sort by ascending offset

    modalities = einx.get_at('b [mi] ..., b mo -> b mo ...', modalities, sorted_indices)
    return modalities, sorted_indices

# deriving relative positions from modality positions
# ex. given a sequence of 10 with an image at offset 3 with length 4 - [t] [t] [t] [i] [i] [i] [i] [t] [t] [t]
# relative positions for rotary will be [0] [1] [2] [3] [3] [3] [3] [4] [5] [6]
# rationale is that each modality will need the same position so there is no distance when conducting bidirectional attention, but should still have a relative distance to other text tokens and modalities

def derive_rotary_positions_from_modality_positions(
    seq_len: int,
    modalities: Int['b m 3']
) -> Int['b n']:

    device = modalities.device

    modality_mask = modality_positions_to_is_modality_mask(seq_len, modalities, offset = torch.tensor([1, -1]))
    is_any_modality = reduce(modality_mask, 'b t m n -> b n', 'any')

    return torch.arange(seq_len, device = device) - is_any_modality.cumsum(dim = -1)

# modality tokens are given as list of tensors, can be then be embedded into the modality tokens for attending alongside text tokens

@typecheck
def embed_modality_tokens(
    seq_len: int,
    dim: int,
    modality_tokens: list[list[Float['...']]],
    modalities: Int['b m 3'],
    modality_id: int,
    channel_first: bool
) -> Float['b n d']:

    batch, device = modalities.shape[0], modalities.device

    shape = (batch, seq_len, dim) if not channel_first else (batch, dim, seq_len)
    output = torch.zeros(shape, device = device)

    for batch_ind, (one_modality, one_modality_token) in enumerate(zip(modalities, modality_tokens)):
        for (modality_type, offset, length), batch_modality_token in zip(one_modality, one_modality_token):

            if modality_id != modality_type or length <= 0:
                continue

            modality_shape = batch_modality_token.shape

            if channel_first:
                mod_dim, *mod_axial_shape = modality_shape
                batch_modality_token = rearrange(batch_modality_token, 'd ... -> d (...)')
            else:
                *mod_axial_shape, mod_dim = modality_shape
                batch_modality_token = rearrange(batch_modality_token, '... d -> (...) d')

            mod_length = math.prod(mod_axial_shape)

            assert length == mod_length, f'received a modality of shape {modality_shape} but sequence length in modalities info is {length}'
            assert dim == mod_dim, f'received modality [{modality_id}] with shape {modality_shape} but expected dimension of {dim}'

            if channel_first:
                output[batch_ind, :, offset:(offset + length)] = batch_modality_token
            else:
                output[batch_ind, offset:(offset + length), :] = batch_modality_token

    return output

# functions for managing modality token mask

@typecheck
def modality_positions_to_is_modality_mask(
    seq_len: int,
    modalities: Int['b m 3'],
    offset: Int['2'] | None = None,
    device = None,
    num_modalities = 1
) -> Bool['b t m n']:

    device = modalities.device

    if exists(offset):
        offset = F.pad(offset, (1, 0))
        modalities = modalities + offset.to(modalities)

    seq = torch.arange(seq_len, device = device)
    type_seq = torch.arange(num_modalities, device = device)

    modality_types = modalities[..., 0]

    left, right = modalities[..., 1:].cumsum(dim = -1).unbind(dim = -1)

    is_instance_for_type = einx.equal('b m, t -> b t m', modality_types, type_seq)

    is_modality_along_seq = (
        einx.greater_equal('i, b m -> b m i', seq, left) &
        einx.less('j, b m -> b m j', seq, right)
    )

    return einx.logical_and('b t m, b m n -> b t m n', is_instance_for_type, is_modality_along_seq)

@typecheck
def naive_attn_mask(
    seq_len: int,
    modalities: Int['b m 3'],
    device = None
) -> Bool['b i j']:

    _, offsets, length = modalities.unbind(dim = -1)

    seq = torch.arange(seq_len, device = device)

    is_causal = einx.greater_equal('i, j -> i j', seq, seq)

    is_modality = (
        einx.greater_equal('i, b m -> b m i 1', seq, offsets) &
        einx.less('j, b m -> b m 1 j', seq, offsets + length)
    )

    return is_causal | is_modality.any(dim = 1)

# unet encoder related function

def stack_same_shape_tensors_with_inverse(tensors: list[Tensor]):

    shape_tensors_dict = defaultdict(list)
    shape_batch_dict = defaultdict(int) # also store a shape -> num tensors dictionary to validate inverse function input
    inverse_index_list = []

    for tensor in tensors:
        shape = tuple(tensor.shape)
        batch_el = shape_batch_dict[shape]

        shape_tensors_dict[shape].append(tensor)
        shape_batch_dict[shape] += 1

        inverse_index_list.append((shape, batch_el))

    # stack all the tensors with same shapes to have a batch dimension

    shape_tensors_dict = {shape: torch.stack(same_shape_tensors) for shape, same_shape_tensors in shape_tensors_dict.items()}

    # inverse function

    def inverse(
        indexed_tensors: dict[tuple[int, ...], Tensor]
    ) -> list[Tensor]:

        out_shape_batch_dict = {shape: len(tensors) for shape, tensors in indexed_tensors.items()}

        assert out_shape_batch_dict == shape_batch_dict

        inversed = []

        for shape, batch_el in inverse_index_list:
            tensor = indexed_tensors[shape][batch_el]
            inversed.append(tensor)

        return inversed

    return shape_tensors_dict, inverse

def filter_with_inverse(cond, inp):

    indices = set()
    filtered = []

    for ind, el in enumerate(inp):
        if cond(el):
            indices.add(ind)
            filtered.append(el)

    def inverse(inverse_inp):
        assert len(inverse_inp) == len(filtered)

        output = []
        inverse_inp_index = 0

        for ind, el in enumerate(inp):
            if ind not in indices:
                output.append(el)
                continue

            inverse_inp_el = inverse_inp[inverse_inp_index]
            output.append(inverse_inp_el)
            inverse_inp_index += 1

        return output

    return filtered, inverse

def apply_fn_modality_type(
    fn: Callable,
    modalities: ModalitySample | list[ModalitySample],
    modality_type = 0,
    return_untransformed = False
) -> ModalitySample | list[ModalitySample]:

    modalities, tree_spec = tree_flatten(modalities, is_leaf = lambda el: isinstance(el, tuple))

    # standardize tuples to (<modality_type>, <modality_tensor>)

    modalities = [(0, m) if (is_tensor(m) and m.dtype == torch.float) else m for m in modalities]

    # filter for specific modality type to transform

    modalities, inverse_filter = filter_with_inverse(lambda el: isinstance(el, tuple) and el[0] == modality_type, modalities)

    # remove the type

    modalities = [m for _, m in modalities]

    # batch process

    stacked_modalities, inverse_stack = stack_same_shape_tensors_with_inverse(modalities)

    out = {shape: fn(batched_modalities) for shape, batched_modalities in stacked_modalities.items()}

    out = inverse_stack(out)

    # add back the type

    if return_untransformed:
        out = [(modality_type, transformed_m, prev_m) for transformed_m, prev_m in zip(out, modalities)]
    else:
        out = [(modality_type, transformed_m) for transformed_m in out]

    # replace transformed modalities and untree flatten

    out = inverse_filter(out)

    return tree_unflatten(out, tree_spec)

# sampling related functions

# min_p for text
# https://arxiv.org/abs/2407.01082

def min_p_filter(logits, min_p = 0.1):
    probs = logits.softmax(dim = -1)
    max_probs = probs.amax(dim = -1, keepdim = True)
    limit = min_p * max_probs
    return torch.where(probs < limit, float('-inf'), logits)

# random fourier embedding

class RandomFourierEmbed(Module):
    def __init__(self, dim):
        super().__init__()
        assert divisible_by(dim, 2)
        self.dim = dim
        self.register_buffer('weights', torch.randn(dim // 2))

    @typecheck
    def forward(
        self,
        times: Float['b n'] | Float['b']
    ) -> Float['b n {self.dim+1}']:

        if times.ndim == 1:
            times = rearrange(times, 'b -> b 1')

        freqs = einx.multiply('... i, j -> ... i j', times, self.weights) * 2 * torch.pi
        fourier_embed, _ = pack((times, freqs.sin(), freqs.cos()), 'b n *')
        return fourier_embed

# adaptive layernorm and ada-ln zero rolled into one wrapper
# from DiT paper and sota for time conditioning for now

class AdaptiveWrapper(Module):
    @beartype
    def __init__(
        self,
        fn: Module,
        dim,
        dim_cond,
        ada_ln_zero_init_bias = -2
    ):
        super().__init__()
        self.fn = fn
        self.dim = dim
        self.dim_cond = dim_cond

        self.layernorm = nn.LayerNorm(dim, elementwise_affine = False)

        # text will be subjected to normal layernorm bias
        # and for output will use layerscale

        self.layernorm_gamma = nn.Parameter(torch.zeros(dim))
        self.layerscale = nn.Parameter(torch.zeros(dim))

        # modalities will get the adaptive layernorm + ada-ln zero

        self.to_film = Linear(dim_cond, dim * 2)
        self.to_ada_ln_zero = Linear(dim_cond, dim)

        nn.init.zeros_(self.to_film.weight)
        nn.init.zeros_(self.to_ada_ln_zero.weight)
        nn.init.constant_(self.to_ada_ln_zero.bias, ada_ln_zero_init_bias)

    @typecheck
    def forward_text(
        self,
        x: Float['b n {self.dim}'],
        **kwargs
    ):
        x = self.layernorm(x)

        x = x * (self.layernorm_gamma + 1.)

        out = self.fn(x, **kwargs)

        (out, *rest), tree_spec = tree_flatten(out)

        out = out * (self.layerscale + 1.)

        out = tree_unflatten((out, *rest), tree_spec)

        return out

    @typecheck
    def forward_modality(
        self,
        x: Float['b n {self.dim}'],
        cond: Float['... {self.dim_cond}'],
        **kwargs
    ):
        x = self.layernorm(x)

        gamma, beta = self.to_film(cond).chunk(2, dim = -1)

        modality_tokens = x * (gamma + 1.) + beta

        # attention or feedforwards

        out = self.fn(modality_tokens, **kwargs)

        (out, *rest), tree_spec = tree_flatten(out)

        # take care of conditioning output separately for text vs modality

        modalities_out = out * self.to_ada_ln_zero(cond).sigmoid()

        # take care of function returning cache or value residual

        modalities_out = tree_unflatten((modalities_out, *rest), tree_spec)

        return modalities_out

    @typecheck
    def forward(
        self,
        x: Float['b n {self.dim}'],
        cond: Float['... {self.dim_cond}'] | None = None,
        is_any_modality: bool | Bool['b n'] | None = None,
        modality_only = False,
        **kwargs
    ):
        if exists(cond) and cond.ndim == 2:
            cond = rearrange(cond, 'b d -> b 1 d')

        if modality_only:
            return self.forward_modality(x, cond = cond, **kwargs)

        assert not (exists(cond) ^ exists(is_any_modality))

        has_modality = exists(is_any_modality)

        if not has_modality:
            return self.forward_text(x, **kwargs)

        if isinstance(is_any_modality, bool):
            is_any_modality = torch.full((x.shape[:-1]), is_any_modality, device = x.device, dtype = torch.bool)

        is_any_modality = rearrange(is_any_modality, '... -> ... 1')

        x = self.layernorm(x)

        gamma, beta = self.to_film(cond).chunk(2, dim = -1)

        text_tokens = x * (self.layernorm_gamma + 1.)

        modality_tokens = x * (gamma + 1.) + beta

        x = torch.where(is_any_modality, modality_tokens, text_tokens)

        # attention or feedforwards

        out = self.fn(x, **kwargs)

        (out, *rest), tree_spec = tree_flatten(out)

        # take care of conditioning output separately for text vs modality

        text_out = out * (self.layerscale + 1.)

        modalities_out = out * self.to_ada_ln_zero(cond).sigmoid()

        conditioned_out = torch.where(is_any_modality, modalities_out, text_out)

        # take care of function returning cache or value residual

        conditioned_out = tree_unflatten((conditioned_out, *rest), tree_spec)

        return conditioned_out

# attention

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        return l2norm(x) * self.scale * (self.gamma + 1.) # use unit offset from Ohad Rubin

class GEGLU(Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return F.gelu(gates) * x

def FeedForward(
    dim,
    expansion_factor = 4.,
    dropout = 0.
):
    dim_inner = int(dim * expansion_factor * 2 / 3)
    return nn.Sequential(
        Linear(dim, dim_inner * 2),
        GEGLU(),
        nn.Dropout(dropout),
        Linear(dim_inner, dim)
    )

class Attention(Module):
    def __init__(
        self,
        dim,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        softcap_value = 50.,
        use_flex_attn = False,
        gate_values = True,
        laser = False,
        laser_softclamp_value = 15.,
        learned_value_residual_mix = False
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        dim_inner = dim_head * heads

        assert not (use_flex_attn and not exists(flex_attention)), 'flex attention is only available on torch 2.5.0 (nightly) onwards'
        self.use_flex_attn = use_flex_attn

        self.to_qkv = nn.Sequential(
            Linear(dim, dim_inner * 3, bias = False),
            Rearrange('b n (qkv h d) -> qkv b h n d', qkv = 3, h = heads)
        )

        self.to_learned_value_residual = nn.Sequential(
            nn.Linear(dim, heads),
            nn.Sigmoid(),
            Rearrange('b n h -> b h n 1') # add head dimension
        ) if learned_value_residual_mix else always(0.5)

        self.to_gates = nn.Sequential(
            nn.Linear(dim, heads, bias = False),
            Rearrange('b n h -> b h n 1', h = heads)
        ) if gate_values else None

        self.softcap_value = softcap_value

        self.laser = laser
        self.laser_softclamp_value = laser_softclamp_value

        self.dropout = nn.Dropout(dropout)

        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            Linear(dim_inner, dim, bias = False)
        )

    def forward(
        self,
        x,
        attn_mask: Tensor | None = None, # for manual masking
        rotary_emb: Tensor | None = None,
        cache: Tensor | None = None,
        causal = False,
        block_mask = None, # only passed in for flex attention
        return_kv_cache = False,
        return_values = False,
        value_residual: Tensor | None = None
    ):
        device, input_is_cuda, is_decoding_with_cache = x.device, x.is_cuda, exists(cache)

        should_use_flex_attn = self.use_flex_attn and input_is_cuda

        # handle maybe mask
        # if receiving kv cache, assume decoding and turn off all masking

        if is_decoding_with_cache:
            block_mask = attn_mask = None

        assert not (exists(block_mask) and exists(attn_mask))
        assert not (not self.use_flex_attn and exists(block_mask)), 'you cannot pass in the `block_mask` if `use_flex_attn` was not set to be `True`'

        # project to queries, keys, values

        q, k, v = self.to_qkv(x)

        # value residual

        orig_v = v

        if exists(value_residual):
            mix = self.to_learned_value_residual(x)
            v = v * mix + value_residual * (1. - mix)

        # handle cache being passed in

        if exists(cache):
            cached_k, cached_v = cache
            k = cat((cached_k, k), dim = -2)
            v = cat((cached_v, v), dim = -2)

        # maybe kv cache

        if return_kv_cache:
            kv_cache = stack((k, v))

        # rotary embeddings

        if exists(rotary_emb):
            q, k = tuple(apply_rotary_emb(rotary_emb, t, freqs_seq_dim = -2) for t in (q, k))

        # laser attention

        if self.laser:
            v = softclamp(v, self.laser_softclamp_value)
            v = v.exp()

        # whether to use flex attention or not

        if should_use_flex_attn:
            assert not causal, 'causal mask should be constructed in transformer'

            flex_attn_kwargs = dict(block_mask = block_mask)

            if self.softcap_value > 0.:
                flex_attn_kwargs.update(score_mod = softcap_score_mod(self.softcap_value))

            out = flex_attention(q, k, v, **flex_attn_kwargs)

        else:
            q = q * self.scale
            sim = einsum(q, k, 'b h i d, b h j d -> b h i j')

            sim = softclamp(sim, self.softcap_value)

            mask_value = max_neg_value(sim)

            if causal:
                i, j = sim.shape[-2:]
                causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
                sim = sim.masked_fill(causal_mask, mask_value)

            if exists(attn_mask):
                sim = einx.where('b i j, b h i j, -> b h i j', attn_mask, sim, mask_value)

            attn = sim.softmax(dim = -1)

            attn = self.dropout(attn)

            out = einsum(attn, v, 'b h i j, b h j d -> b h i d')

        # laser attention

        if self.laser:
            out = log(out)

        # maybe gate values

        if exists(self.to_gates):
            out = out * self.to_gates(x).sigmoid()

        # combine heads and out

        out = self.to_out(out)

        if return_values:
            out = (out, orig_v)

        if not return_kv_cache:
            return out

        return out, kv_cache

class Transformer(Module):
    @beartype
    def __init__(
        self,
        dim,
        *,
        depth,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        ff_expansion_factor = 4,
        attn_kwargs: dict = dict(),
        ff_kwargs: dict = dict(),
        attn_laser = False,
        unet_skips = True,
        use_flex_attn = False,
        num_residual_streams = 4
    ):
        super().__init__()
        self.dim = dim
        self.dim_head = dim_head
        self.depth = depth
        self.unet_skips = unet_skips

        # 创建新的RandomFourierEmbed实例，确保weights是float16
        fourier_embed = RandomFourierEmbed(dim)
        fourier_embed.weights.data = fourier_embed.weights.data.to(torch.float16)
        
        # 重新创建time_mlp
        self.time_mlp = nn.Sequential(
            fourier_embed,
            Linear(dim + 1, dim * 4).to(torch.float16),
            nn.SiLU(),
            Linear(dim * 4, dim).to(torch.float16)
        )

        self.layers = ModuleList([])

        for _ in range(depth):
            self.layers.append(ModuleList([
                AdaptiveWrapper(
                    Attention(
                        dim = dim,
                        dim_head = dim_head,
                        heads = heads,
                        dropout = dropout,
                        use_flex_attn = use_flex_attn,
                        laser = attn_laser,
                        **attn_kwargs
                    ),
                    dim = dim,
                    dim_cond = dim
                ),
                AdaptiveWrapper(
                    FeedForward(
                        dim = dim,
                        expansion_factor = ff_expansion_factor,
                        dropout = dropout,
                        **ff_kwargs
                    ),
                    dim = dim,
                    dim_cond = dim
                )
            ]))

        # residual streams

        self.num_residual_streams = num_residual_streams

        if num_residual_streams > 1:
            self.hyper_connections = HyperConnections(
                dim = dim,
                depth = depth,
                num_streams = num_residual_streams
            )

    @typecheck
    def forward(
        self,
        x,
        times: Scalar | Float['b'] | Float['b n'] | None = None,
        attn_mask: Bool['b i j'] | None = None,
        modality_positions: RawModalityPositions | Int['b m 3'] | None = None,
        is_any_modality: bool | Bool['b n'] | None = None,
        rotary_emb: Tensor | None = None,
        cache: Tensor | None = None,
        decode_length: int | None = None,
        modality_only = False,
        causal_mask = False,
        return_kv_cache = False
    ):
        device, batch, seq_len, is_decoding_with_cache = x.device, *x.shape[:2], exists(cache)

        # handle times

        if exists(times):
            if times.ndim == 0:
                times = repeat(times, ' -> b', b = batch)

            if times.ndim == 1:
                times = repeat(times, 'b -> b n', n = seq_len)

            time_embed = self.time_mlp(times)
        else:
            time_embed = None

        # handle cache

        if is_decoding_with_cache:
            assert exists(decode_length)
            assert not exists(attn_mask)

            if decode_length > 1:
                assert not causal_mask, 'cannot use causal mask when decoding multiple tokens'

            x = x[:, -decode_length:]

        # handle attn mask

        if causal_mask:
            i, j = seq_len, seq_len

            if is_decoding_with_cache:
                i = decode_length
                j = decode_length + cache.shape[-2]

            causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
            attn_mask = default(attn_mask, torch.zeros((batch, i, j), dtype = torch.bool, device = device))
            attn_mask = attn_mask | repeat(causal_mask, 'i j -> b i j', b = batch)

        # handle modality positions

        if exists(modality_positions) and not is_tensor(modality_positions):
            modality_positions = modality_positions_to_tensor(modality_positions, device = device)

        # handle flex attention block mask

        block_mask = None

        if exists(modality_positions) and exists(flex_attention):
            block_mask = create_block_mask(transfusion_attn_mask(modality_positions))

        # handle cache

        if is_decoding_with_cache:
            cache = cache.unbind(dim = 0)
            cache = list(map(lambda t: t.unbind(dim = 0), cache))
            cache = list(zip(*cache))

        # handle residual streams

        if self.num_residual_streams > 1:
            x = repeat(x, 'b n d -> s b n d', s = self.num_residual_streams)

        # go through layers

        hiddens = []
        kv_cache = []

        for ind, (attn, ff) in enumerate(self.layers):
            layer_cache = None if not is_decoding_with_cache else cache[ind]

            if self.num_residual_streams > 1:
                x = self.hyper_connections.route_layer(x, ind)

            if self.unet_skips:
                hiddens.append(x)

            attn_kwargs = dict(
                attn_mask = attn_mask,
                rotary_emb = rotary_emb,
                block_mask = block_mask,
                causal = causal_mask and not exists(attn_mask)
            )

            if exists(layer_cache):
                attn_kwargs.update(cache = layer_cache)

            if return_kv_cache:
                attn_kwargs.update(return_kv_cache = True)

            # attention layer

            if return_kv_cache:
                x, layer_kv_cache = attn(
                    x,
                    cond = time_embed,
                    is_any_modality = is_any_modality,
                    modality_only = modality_only,
                    **attn_kwargs
                )

                kv_cache.append(layer_kv_cache)
            else:
                x = attn(
                    x,
                    cond = time_embed,
                    is_any_modality = is_any_modality,
                    modality_only = modality_only,
                    **attn_kwargs
                )

            # feedforward

            x = ff(
                x,
                cond = time_embed,
                is_any_modality = is_any_modality,
                modality_only = modality_only
            )

            # maybe unet skip connections

            if self.unet_skips and len(hiddens) > 1:
                x = x + hiddens.pop(0)

        if self.num_residual_streams > 1:
            x = self.hyper_connections.combine_final(x)

        if not return_kv_cache:
            return x

        kv_cache = torch.stack(list(map(torch.stack, zip(*kv_cache))))
        return x, kv_cache

# 添加 LlamaTransformer 类
class LlamaTransformer(Module):
    @beartype
    def __init__(
        self,
        dim,
        *,
        model_name_or_path=None,  # Llama 模型名称或路径
        config=None,    
        use_gradient_checkpointing=True,          # 自定义 Llama 配置
        dim_head=64,              # 添加 dim_head 参数
        **kwargs                  # 兼容原有参数
    ):
        super().__init__()
        
        if config is None:
            if model_name_or_path is not None:
                self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                            torch_dtype=torch.float16,  # 或 torch.bfloat16
                                                            # device_map={"": 0},  # 自动分配到可用设备
                                                            use_cache=not use_gradient_checkpointing
                                                            )
                if use_gradient_checkpointing:
                    self.model.gradient_checkpointing_enable()
            else:
                # 创建默认配置
                llama_config = AutoConfig(
                    hidden_size=dim,
                    # 其他必要参数
                )
                self.model = AutoModelForCausalLM(llama_config)
        else:
            self.model = AutoModelForCausalLM(config)
            
        self.dim = dim
        self.dim_head = dim_head  # 保存 dim_head 属性
        

        # 创建新的RandomFourierEmbed实例，确保weights是float16
        fourier_embed = RandomFourierEmbed(dim)
        fourier_embed.weights.data = fourier_embed.weights.data.to(torch.float16)
        
        # 重新创建time_mlp
        self.time_mlp = nn.Sequential(
            fourier_embed,
            Linear(dim + 1, dim * 4).to(torch.float16),
            nn.SiLU(),
            Linear(dim * 4, dim).to(torch.float16)
        )
    
    def forward(
        self,
        x,
        times=None,
        attn_mask=None,
        modality_positions=None,
        is_any_modality=None,
        rotary_emb=None,
        cache=None,
        decode_length=None,
        modality_only=False,
        causal_mask=False,
        return_kv_cache=False
    ):
        device, batch = x.device, x.shape[0]
        
        # 处理时间条件
        if exists(times):
            if times.ndim == 0:
                times = repeat(times, ' -> b', b = batch)

            if times.ndim == 1:
                times = repeat(times, 'b -> b n', n = x.shape[1])

            time_embed = self.time_mlp(times)
            
            # 将时间条件添加到输入中
            x = x + time_embed
        
        # 适配 Llama 的输入格式
        attention_mask = None
        if exists(attn_mask):
            attention_mask = (~attn_mask).float() * -10000.0
        
        # 处理 cache
        past_key_values = None
        if exists(cache):
            past_key_values = cache
        
        # 调用 Llama 模型
        outputs = self.model(
            inputs_embeds=x,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=return_kv_cache,
            output_hidden_states=True,
        )
        
        hidden_states = outputs.hidden_states[-1]  # 获取最后一层的隐藏状态
        
        if return_kv_cache:
            return hidden_states, outputs.past_key_values
        
        return hidden_states

class LLMFlow(Module):
    @beartype
    def __init__(
        self,
        *,
        num_text_tokens,
        transformer: dict | LlamaTransformer,  # 添加 LlamaTransformer 支持
        dim_latent: int | tuple[int, ...] | None = None,
        channel_first_latent: bool | tuple[bool, ...] = False,
        add_pos_emb: bool | tuple[bool, ...] = False,
        modality_encoder: Module | tuple[Module, ...] | None = None,
        modality_decoder: Module | tuple[Module, ...] | None = None,
        pre_post_transformer_enc_dec: tuple[Module, Module] | tuple[tuple[Module, Module], ...] | None = None,
        modality_default_shape: tuple[int, ...] | tuple[tuple[int, ...], ...] | None = None,
        fallback_to_default_shape_if_invalid = False,
        modality_num_dim: int | tuple[int, ...] | None = None,
        to_modality_shape_fn: Callable | tuple[Callable, ...] = default_to_modality_shape_fn,
        ignore_index = -1,
        flow_loss_weight = 1.,
        text_loss_weight = 1.,
        velocity_consistency_loss_weight = 0.1,
        reconstruction_loss_weight = 0.,
        modality_encoder_decoder_requires_batch_dim = True, # whether the modality encoder / decoder requires batch dimension, will auto assume it is needed
        odeint_kwargs: dict = dict(
            atol = 1e-5,
            rtol = 1e-5,
            method = 'midpoint'
        ),
    ):
        super().__init__()

        # transformer

        if isinstance(transformer, dict):
            # 检查是否使用 Llama
            if transformer.get('use_llama', False):
                transformer.pop('use_llama', None)
                transformer = LlamaTransformer(**transformer)
            else:
                transformer = Transformer(**transformer)

        assert isinstance(transformer, (Transformer, LlamaTransformer)), "transformer must be a Transformer or LlamaTransformer instance"
        self.transformer = transformer
        dim = transformer.dim

        self.dim = dim

        # latent and model dimension not the same
        # make it work for 1 modality for now

        dim_latent = default(dim_latent, dim)

        self.dim_latents = cast_tuple(dim_latent)

        # number of modalities

        self.num_modalities = len(self.dim_latents)

        # whether the latents are accepted to be channel first or channel last
        # if channel first, will be rearrange(c ... -> ... c -> (...) c)

        self.channel_first_latent = cast_tuple(channel_first_latent, self.num_modalities)
        assert len(self.channel_first_latent) == self.num_modalities

        # functions for converting the sampled language model meta string back to modality shape of tuple[int, ...]

        self.to_modality_shape_fn = cast_tuple(to_modality_shape_fn, self.num_modalities)

        # specifying the number of dimensions for the modality, which will be hard validated

        self.modality_num_dim = cast_tuple(modality_num_dim, self.num_modalities)
        assert len(self.modality_num_dim) == self.num_modalities

        # whether to add an extra axial positional embedding per modality

        self.add_pos_emb = cast_tuple(add_pos_emb, self.num_modalities)
        assert len(self.add_pos_emb) == self.num_modalities

        self.pos_emb_mlp = ModuleList([])

        for modality_add_pos_emb, modality_ndim in zip(self.add_pos_emb, self.modality_num_dim):

            if not modality_add_pos_emb:
                self.pos_emb_mlp.append(None)
                continue

            assert exists(modality_ndim), '`modality_num_dim` must be set if you wish to automatically inject axial positional embeddings'

            pos_generating_mlp = ContinuousAxialPositionalEmbedding(
                dim = dim,
                num_axial_dims = modality_ndim,
            )

            self.pos_emb_mlp.append(pos_generating_mlp)

        # modality encoders and decoders

        # modality_encoder = cast_tuple(modality_encoder, 1 if exists(modality_encoder) else self.num_modalities)
        # modality_decoder = cast_tuple(modality_decoder, 1 if exists(modality_decoder) else self.num_modalities)

        # self.modality_encoder = ModuleList(modality_encoder)
        # self.modality_decoder = ModuleList(modality_decoder)
        # 修改1: 正确处理单模态编码器/解码器
        if isinstance(modality_encoder, Module):
            self.modality_encoder = ModuleList([modality_encoder])
        else:
            self.modality_encoder = ModuleList(modality_encoder or [])

        if isinstance(modality_decoder, Module):
            self.modality_decoder = ModuleList([modality_decoder])
        else:
            self.modality_decoder = ModuleList(modality_decoder or [])

        # 确保编码器/解码器数量匹配
        assert len(self.modality_encoder) == self.num_modalities, \
            f"Expected {self.num_modalities} encoders, got {len(self.modality_encoder)}"
        assert len(self.modality_decoder) == self.num_modalities, \
            f"Expected {self.num_modalities} decoders, got {len(self.modality_decoder)}"

        # 修改2: 单模态专用处理
        # if self.num_modalities == 1:
        #     self.single_modality_encoder = self.modality_encoder[0]
        #     self.single_modality_decoder = self.modality_decoder[0]

        # auto handle batch dimension for modality encoder / decoder

        self.maybe_add_temp_batch_dim = add_temp_batch_dim if modality_encoder_decoder_requires_batch_dim else identity

        # default token lengths for respective modality
        # fallback if the language model does not come up with valid dimensions

        if not exists(modality_default_shape) or is_bearable(modality_default_shape, tuple[int, ...]):
            modality_default_shape = (modality_default_shape,) * self.num_modalities

        self.modality_default_shape = modality_default_shape

        assert len(self.modality_default_shape) == self.num_modalities

        self.fallback_to_default_shape_if_invalid = fallback_to_default_shape_if_invalid

        # store number of text tokens

        self.num_text_tokens = num_text_tokens

        # entire "sentence" start and end id

        num_text_special_ids = 2

        self.sos_id, self.eos_id = num_text_tokens, (num_text_tokens + 1)

        # modality meta, start and end tokens - termed [mom] [som] [eom] in this repo

        num_modality_special_ids = self.num_modalities * 2
        som_eom_tensor = torch.arange(num_modality_special_ids) + num_text_tokens + num_text_special_ids # shift to the very end
        som_eom_tensor = rearrange(som_eom_tensor, '(start_end m) -> start_end m', start_end = 2)

        # modality meta, start and end ids

        self.som_ids, self.eom_ids = som_eom_tensor.tolist()

        # char tokenizing for modality meta information

        meta_token_offset = num_text_tokens + num_text_special_ids + num_modality_special_ids

        self.meta_id = meta_token_offset

        self.char_tokenizer = partial(char_tokenize, offset = meta_token_offset + 1)
        self.decode_chars = partial(decode_chars, offset = meta_token_offset + 1)

        num_meta_tokens = 128 + 1

        # prepare pre-post transformer encoder / decoder, for the learnable unets as in paper

        if is_bearable(pre_post_transformer_enc_dec, tuple[Module, Module]):
            pre_post_transformer_enc_dec = (pre_post_transformer_enc_dec,)

        pre_post_transformer_enc_dec = cast_tuple(pre_post_transformer_enc_dec, self.num_modalities)
        assert len(pre_post_transformer_enc_dec) == self.num_modalities

        # latent to model and back
        # by default will be Linear, with or without rearranges depending on channel_first_latent setting
        # can also be overridden for the unet down/up as in the paper with `pre_post_transformer_enc_dec: tuple[Module, Module]`

        latent_to_model_projs = []
        model_to_latent_projs = []

        for (
            dim_latent,
            one_channel_first_latent,
            enc_dec,
         ) in zip(
            self.dim_latents,
            self.channel_first_latent,
            pre_post_transformer_enc_dec
        ):

            pre_attend_enc, post_attend_dec = default(enc_dec, (None, None))

            latent_to_model_proj = Linear(dim_latent, dim) if dim_latent != dim else nn.Identity()
            model_to_latent_proj = Linear(dim, dim_latent, bias = False)

            if one_channel_first_latent:
                latent_to_model_proj = nn.Sequential(Rearrange('b d ... -> b ... d'), latent_to_model_proj)
                model_to_latent_proj = nn.Sequential(model_to_latent_proj, Rearrange('b ... d -> b d ...'))

                if exists(pre_attend_enc):
                    pre_attend_enc = nn.Sequential(pre_attend_enc, Rearrange('b d ... -> b ... d'))

                if exists(post_attend_dec):
                    post_attend_dec = nn.Sequential(Rearrange('b ... d -> b d ...'), post_attend_dec)

            latent_to_model_projs.append(default(pre_attend_enc, latent_to_model_proj))
            model_to_latent_projs.append(default(post_attend_dec, model_to_latent_proj))

        self.latent_to_model_projs = ModuleList(latent_to_model_projs)
        self.model_to_latent_projs = ModuleList(model_to_latent_projs)

        # relative positions

        # 根据 transformer 类型选择合适的 rotary_emb 初始化
        if isinstance(transformer, LlamaTransformer):
            self.rotary_emb = RotaryEmbedding(transformer.dim_head)
        else:
            self.rotary_emb = RotaryEmbedding(32)

        # embeddings and un-embeddings

        effective_num_text_tokens = num_text_tokens + num_text_special_ids + num_modality_special_ids + num_meta_tokens

        self.text_embed = nn.Embedding(effective_num_text_tokens, dim)

        self.to_text_logits = Linear(dim, effective_num_text_tokens, bias = False)

        text_only_mask = torch.arange(effective_num_text_tokens) < num_text_tokens
        self.register_buffer('text_only_logits_mask', text_only_mask, persistent = False)

        # loss related

        self.ignore_index = ignore_index
        self.flow_loss_weight = flow_loss_weight
        self.text_loss_weight = text_loss_weight

        # velocity consistency weight - only added if EMA model is passed in during training

        self.velocity_consistency_loss_weight = velocity_consistency_loss_weight

        # additional reconstruction loss, through the decoder

        self.has_recon_loss = reconstruction_loss_weight > 0.
        self.reconstruction_loss_weight = reconstruction_loss_weight

        # flow sampling related

        self.odeint_fn = partial(odeint, **odeint_kwargs)

        # dummy loss

        self.register_buffer('zero', tensor(0.), persistent = False)


    @property
    def device(self):
        return next(self.parameters()).device

    @cache
    def get_modality_info(
        self,
        modality_type: int | None = None
    ) -> ModalityInfo:

        modality_type = default(modality_type, 0)

        modality_encoder = self.modality_encoder[modality_type]
        modality_decoder = self.modality_decoder[modality_type]
        latent_to_model = self.latent_to_model_projs[modality_type]
        model_to_latent = self.model_to_latent_projs[modality_type]

        add_pos_emb = self.add_pos_emb[modality_type]
        pos_emb_mlp = self.pos_emb_mlp[modality_type]
        modality_num_dim = self.modality_num_dim[modality_type]

        dim_latent = self.dim_latents[modality_type]

        default_shape = self.modality_default_shape[modality_type]

        som_id = self.som_ids[modality_type]
        eom_id = self.eom_ids[modality_type]

        to_shape_fn = self.to_modality_shape_fn[modality_type]

        channel_first_latent = self.channel_first_latent[modality_type]

        return ModalityInfo(
            encoder = modality_encoder,
            decoder = modality_decoder,
            latent_to_model = latent_to_model,
            model_to_latent = model_to_latent,
            add_pos_emb = add_pos_emb,
            pos_emb_mlp = pos_emb_mlp,
            num_dim = modality_num_dim,
            dim_latent = dim_latent,
            default_shape = default_shape,
            som_id = som_id,
            eom_id = eom_id,
            to_shape_fn = to_shape_fn,
            channel_first_latent = channel_first_latent,
            modality_type = modality_type
        )

    def get_all_modality_info(self) -> list[ModalityInfo]:
        return [self.get_modality_info(i) for i in range(self.num_modalities)]

    def get_modality_shape(
        self,
        modality: Float['...'],
        modality_type: int | None  = None
    ) -> tuple[int, ...]:

        mod = self.get_modality_info(modality_type)

        if mod.channel_first_latent:
            modality = rearrange(modality, 'c ... -> ... c')

        return tuple(modality.shape[:-1])

    def parameters_without_encoder_decoder(self):
        return (
            set(self.parameters()) -
            set(self.modality_encoder.parameters()) -
            set(self.modality_decoder.parameters())
        )

    def create_dataloader(
        self,
        *args,
        **kwargs
    ):
        return create_dataloader(*args, **kwargs)

    def create_ema(
        self,
        beta = 0.99,
        *ema_kwargs
    ) -> EMA:

        ema = EMA(
            self,
            beta = beta,
            forward_method_names = (
                'sample',
                'generate_text_only',
                'generate_modality_only',
                'generate_protein'
            )
        )

        return ema

    @torch.no_grad()
    @eval_decorator
    @typecheck
    def sample(
        self,
        prompt: ModalitySample | Tensor | tuple[int, Float['...']] | None = None,
        max_length = 2048,
        text_temperature = 0.2,
        text_min_p = 0.1,
        cache_kv = False,
        fixed_modality_shape: tuple[int, ...] | None = None,
        init_modality_noise: Float['n d'] | None = None,
        modality_steps = 32,
        return_unprocessed_modalities = False
    ) -> ModalitySample:

        device = self.device

        # handle edge case where there are no text tokens

        if self.num_text_tokens == 0:
            logger.warning(f'you have `num_text_tokens` set to 0, so `sample` will be forwarded to `generate_modality_only(batch_size: int, modality_type: int)` method')

            return self.generate_modality_only(batch_size = 1)

        # take care of prompt being a raw tensor, either text or raw modality (image, video, actions, latents, etc)

        if is_tensor(prompt) and prompt.dtype == torch.float: # is modality with type 0 implicit
            prompt = (0, prompt)

        if is_tensor(prompt) and prompt.dtype in (torch.int, torch.long): # is text only prompt
            prompt = [prompt]
        
        elif isinstance(prompt, tuple):
            modality_type, modality = prompt

            mod = self.get_modality_info(modality_type)

            if exists(mod.encoder):
                with torch.no_grad():
                    mod.encoder.eval()
                    modality = self.maybe_add_temp_batch_dim(mod.encoder)(modality).detach()

            modality_shape_tuple = self.get_modality_shape(modality, modality_type)
            modality_shape_str = join([*map(str, modality_shape_tuple)], ',')
            modality_meta_info = self.char_tokenizer(modality_shape_str, device = device)

            prompt = [
                tensor([self.meta_id]),
                modality_meta_info,
                tensor([mod.som_id]),
                (modality_type, modality),
                tensor([mod.eom_id]),
            ]

        # sos

        init_text_seq = tensor([self.sos_id], device = device)

        # just take care of prompt being zero dimensions

        modality_sample = [init_text_seq, *default(prompt, [])]

        # take care of moving to device

        modality_sample = tree_map_tensor(modality_sample, lambda t: t.to(device))
        modality_sample = tree_map_tensor(modality_sample, lambda t: rearrange(t, '-> 1') if t.ndim == 0 else t)

        modality_sample = concat_contiguous_text(modality_sample)

        *_, last_modality_sample = modality_sample

        curr_length = 0
        curr_modality_id = None
        modality_shape = None
        num_past_modalities = 0  # starts off with no modalities in output

        text_is_greedy = text_temperature == 0.
        is_decoding_text = True  # starts off with text decoding, and alternates with modalities depending on [som] tokens detected

        def maybe_transition_to_modality_decoding(seq):
            nonlocal modality_shape
            nonlocal is_decoding_text
            nonlocal curr_modality_id

            sampled_token_id = seq[-1]

            if sampled_token_id not in self.som_ids:
                return

            curr_modality_id = self.som_ids.index(sampled_token_id)

            if exists(fixed_modality_shape):
                modality_shape = fixed_modality_shape

            # get the tokens after the modality meta id

            maybe_meta_tensor = get_tokens_since_rightmost_id(seq, self.meta_id)

            mod = self.get_modality_info(curr_modality_id)

            default_shape = mod.default_shape
            maybe_modality_num_dim = mod.num_dim
            meta_str_to_modality_shape = mod.to_shape_fn

            if maybe_meta_tensor.numel() > 0:
                meta_tensor = maybe_meta_tensor[:-1]
                meta_str = self.decode_chars(meta_tensor)

                if not meta_str.isdigit() or int(meta_str) <= 0:

                    assert exists(default_shape), 'invalid modality meta information detected, please set `modality_default_shape` in order to properly fallback'
                    modality_shape = default_shape
                else:
                    modality_shape = meta_str_to_modality_shape(meta_str)

            modality_shape = default(modality_shape, default_shape)

            if self.fallback_to_default_shape_if_invalid:

                if exists(maybe_modality_num_dim) and len(modality_shape) != maybe_modality_num_dim:
                    logger.warning(f'invalid modality shape {modality_shape} for modality {curr_modality_id}. falling back to default shape {default_shape}')
                    modality_shape = default_shape

            assert exists(modality_shape), f'language model did not produce a proper modality shape for modality type {curr_modality_id} - please set a fallback shape with `modality_default_shape`'
            assert not exists(maybe_modality_num_dim) or maybe_modality_num_dim == len(modality_shape), f'expected modality type {curr_modality_id} to have {maybe_modality_num_dim} dimensions but language model produced a shape of {modality_shape}'

            is_decoding_text = False

        # determine if to transition from start

        maybe_transition_to_modality_decoding(last_modality_sample)

        cache = None

        with tqdm(total = max_length) as pbar:

            while curr_length <= max_length:

                if is_decoding_text:
                    pbar.set_description('decoding text')

                    *_, seq = modality_sample

                    logits, new_kv_cache = self.forward(
                        [modality_sample],
                        return_loss = False,
                        cache = cache,
                        decode_length = 1,
                        decoding_text_or_modality = 'text',
                        return_kv_cache = True
                    )

                    logits = logits[0][-1]

                    if text_is_greedy:
                        sampled = logits.argmax(dim = -1, keepdim = True)
                    else:
                        logits = min_p_filter(logits, min_p = text_min_p)

                        probs = (logits / text_temperature).softmax(dim = -1)
                        sampled = torch.multinomial(probs, 1)

                    seq = torch.cat((seq, sampled), dim = -1)
                    modality_sample[-1] = seq

                    pbar.update(1)
                    curr_length += 1

                    if cache_kv:
                        cache = new_kv_cache

                    sampled_token_id = sampled.item()

                    if sampled_token_id == self.eos_id:
                        logger.info(f'detecting an end of string token [{self.eos_id}], terminating sampling early')
                        break

                    maybe_transition_to_modality_decoding(seq)

                else:
                    assert exists(curr_modality_id)
                    pbar.set_description(f'decoding modality [{curr_modality_id}]')

                    mod = self.get_modality_info(curr_modality_id)

                    modality_length = math.prod(modality_shape)

                    if exists(init_modality_noise):
                        noise = init_modality_noise[:modality_length, :mod.dim_latent]
                    else:
                        assert exists(modality_length)
                        noise = torch.randn((modality_length, mod.dim_latent), device = device)

                    assert noise.shape == (modality_length, mod.dim_latent)

                    noise = noise.reshape(*modality_shape, mod.dim_latent)

                    if mod.channel_first_latent:
                        noise = rearrange(noise, '... d -> d ...')

                    new_kv_cache = None

                    def ode_step_fn(step_times, denoised):
                        nonlocal new_kv_cache

                        step_times = rearrange(step_times, ' -> 1 1') # batch size of 1
                        step_times = F.pad(step_times, (num_past_modalities, 0), value = 1.) # past decoded modalities receive a time conditioning of 1.

                        (embeds, get_pred_flows), new_kv_cache = self.forward(
                            [[*modality_sample, (curr_modality_id, denoised)]],
                            times = step_times,
                            return_embed = True,
                            cache = cache,
                            decode_length = modality_length,
                            return_kv_cache = True,
                            decoding_text_or_modality = 'modality'
                        )

                        parse_embed = get_pred_flows[curr_modality_id][-1]

                        parsed_embed = parse_embed(embeds, need_splice = not exists(cache))

                        flow = add_temp_batch_dim(mod.model_to_latent)(parsed_embed)

                        return flow

                    times = torch.linspace(0, 1, modality_steps, device = device)

                    trajectory = self.odeint_fn(ode_step_fn, noise, times)

                    # add the sampled modality tokens

                    sampled_modality = trajectory[-1]

                    modality_sample.append((curr_modality_id, sampled_modality))

                    # add the appropriate [eom]

                    eom_id = mod.eom_id
                    modality_sample.append(tensor([eom_id], device = device))

                    # set kv cache if needed

                    if cache_kv:
                        cache = new_kv_cache

                    # back to decoding text

                    pbar.update(modality_length)
                    curr_length += modality_length

                    num_past_modalities += 1
                    curr_modality_id = None
                    modality_length = None

                    is_decoding_text = True

        logger.info(f'sampling stopped at length: {curr_length} / {max_length}')

        if return_unprocessed_modalities:
            return modality_sample

        # post process modality sample, decoding modality types if they have a decoder

        for mod in self.get_all_modality_info():
            decoder_fn = default(mod.decoder, nn.Identity())

            with torch.no_grad():
                decoder_fn.eval()
                modality_sample = apply_fn_modality_type(decoder_fn, modality_sample, modality_type = mod.modality_type)

        return modality_sample
    
    def protein_reconstruction_loss(self, seq_logits, coords_pred, seq_target, coords_target):
        """Calculate reconstruction loss for both sequence and structure"""
        # Sequence loss (cross entropy)
        seq_loss = F.cross_entropy(seq_logits.view(-1, 20), seq_target.view(-1))
        
        # Coordinate loss (MSE)
        coord_loss = F.mse_loss(coords_pred, coords_target)
        
        # RMSD loss for overall structure quality
        rmsd_loss = torch.sqrt(((coords_pred - coords_target) ** 2).sum(dim=-1)).mean()
        
        return seq_loss + coord_loss + 0.1 * rmsd_loss

    @typecheck
    def forward_protein(
        self,
        seq_tensor: Int['b n'],
        coords_tensor: Float['b n 3'],
        return_loss = True
    ):
        # Iterate over the modality_encoder if it's a ModuleList
        if isinstance(self.modality_encoder, ModuleList):
            z, mu, logvar = [], [], []
            for encoder in self.modality_encoder:
                z_i, mu_i, logvar_i = encoder(seq_tensor, coords_tensor)
                z.append(z_i)
                mu.append(mu_i)
                logvar.append(logvar_i)
            # Concatenate results
            z = torch.cat(z, dim=1)
            mu = torch.cat(mu, dim=1)
            logvar = torch.cat(logvar, dim=1)
        else:
            z, mu, logvar = self.modality_encoder(seq_tensor, coords_tensor)

        # Iterate over the modality_decoder if it's a ModuleList
        if isinstance(self.modality_decoder, ModuleList):
            seq_logits, coords_pred = [], []
            for decoder in self.modality_decoder:
                seq_logits_i, coords_pred_i = decoder(z)
                seq_logits.append(seq_logits_i)
                coords_pred.append(coords_pred_i)
            # Concatenate results
            seq_logits = torch.cat(seq_logits, dim=1)
            coords_pred = torch.cat(coords_pred, dim=1)
        else:
            seq_logits, coords_pred = self.modality_decoder(z)
        
        if not return_loss:
            return (seq_logits, coords_pred), (mu, logvar)
            
        recon_loss = self.protein_reconstruction_loss(seq_logits, coords_pred, seq_tensor, coords_tensor)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        loss = recon_loss + 0.1 * kl_loss
        return loss, (recon_loss, kl_loss)

    @torch.no_grad()
    @eval_decorator
    @typecheck
    def generate_protein(
        self,
        batch_size: int = 1,
        modality_type: int | None = None,
        fixed_modality_shape: tuple[int, ...] | None = None,
        modality_steps = 16,
        return_unprocessed_modalities = False
    ) -> Tensor:

        device = self.device

        if self.num_modalities > 1:
            assert exists(modality_type), '`modality_type` must be explicitly passed in on forward when training on greater than 1 modality'

        mod = self.get_modality_info(modality_type)

        modality_shape = default(fixed_modality_shape, mod.default_shape)

        assert exists(modality_shape)

        noise = torch.randn((batch_size, *modality_shape, mod.dim_latent), device = device)

        if mod.channel_first_latent:
            noise = rearrange(noise, 'b ... d -> b d ...')

        def ode_step_fn(step_times, denoised):

            step_times = repeat(step_times, ' -> b', b = batch_size)

            flow = self.forward_protein(
                seq_tensor,
                coords_tensor,
                times = step_times,
                modality_type = modality_type,
                encode_modality = False,
                return_loss = False
            )

            return flow

        times = torch.linspace(0., 1., modality_steps, device = device)
        trajectory = self.odeint_fn(ode_step_fn, noise, times)

        # add the sampled modality tokens

        sampled_modality = trajectory[-1]

        # decode

        if exists(mod.decoder):
            mod.decoder.eval()
            sampled_modality = mod.decoder(sampled_modality)

        return sampled_modality

    @typecheck
    def forward_text(
        self,
        text: Int['b n'],
        return_loss = True,
        return_embed = False,
        cache: Tensor | None = None,
        return_kv_cache = False
    ) -> (
        Scalar |
        Float['b n d'] |
        tuple[Float['b n d'], list[Float['...']]]
    ):

        device = self.device
        text = text.to(device)

        if return_loss:
            text, labels = text[:, :-1], text[:, 1:]

        # embed text

        text = text.masked_fill(text == -1, 0)
        tokens = self.text_embed(text)

        # rotary

        seq_len = tokens.shape[-2]
        pos = torch.arange(seq_len, device = device)

        rotary_emb = self.rotary_emb(pos)

        # attention

        embed, kv_cache = self.transformer(
            tokens,
            rotary_emb = rotary_emb,
            causal_mask = True,
            cache = cache,
            return_kv_cache = True
        )

        # text unembedding

        logits = self.to_text_logits(embed)

        if not return_loss:
            if not return_kv_cache:
                return logits

            return logits, kv_cache

        logits = logits.masked_fill(~self.text_only_logits_mask, max_neg_value(logits))

        loss = F.cross_entropy(
            rearrange(logits, 'b n l -> b l n'),
            labels,
            ignore_index = self.ignore_index
        )

        return loss

    @torch.no_grad()
    @eval_decorator
    @typecheck
    def generate_text_only(
        self,
        prompt: Int['b n'],
        seq_len: int,
        temperature = 1.5,
        min_p = 0.1,
    ) -> Int['b no']:

        prompt_seq_len, out = prompt.shape[-1], prompt.clone()
        sample_num_times = max(0, seq_len - prompt_seq_len)

        for _ in tqdm(range(sample_num_times)):
            logits = self.forward_text(out, return_loss = False)
            logits = logits[:, -1]

            logits = min_p_filter(logits, min_p = min_p)

            logits.masked_fill_(~self.text_only_logits_mask, max_neg_value(logits))

            sample = gumbel_sample(logits, temperature = temperature, dim = -1)

            out = cat((out, sample), dim = -1)

        return out[..., prompt_seq_len:]

    @typecheck
    def forward_modality(
        self,
        modalities: dict[str, Float['b ...']] | Float['b ...'],
        times: Float['b'] | None = None,
        modality_type: int | None = None,
        encode_modality: bool = True,
        velocity_consistency_ema_model: LLMFlow | None = None,
        velocity_consistency_delta_time = 1e-5,
        return_loss = True,
        return_loss_breakdown = False
    ) -> Scalar | Float['b ...']:
        requires_velocity_consistency = exists(velocity_consistency_ema_model)

        # Handle the case where modalities is a Dictionary
        if isinstance(modalities, dict):
            # Convert dictionary values to the correct device
            modalities = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in modalities.items()
            }
            # Get batch size from the first tensor in the dictionary
            batch_size = next(iter(modalities.values())).shape[0]
        elif isinstance(modalities, torch.Tensor):
            modalities = modalities.to(self.device)
            batch_size = modalities.shape[0]
        else:
            raise ValueError(f"Expected modalities to be a tensor or dictionary, got {type(modalities)}")

        orig_modalities = modalities

        if self.num_modalities > 1:
            assert exists(modality_type), '`modality_type` must be explicitly passed in on forward when training on greater than 1 modality'

        modality_type = default(modality_type, 0)

        mod = self.get_modality_info(modality_type)

        # maybe modality encode
        if encode_modality and exists(mod.encoder):
            with torch.no_grad():
                if isinstance(mod.encoder, ModuleList):
                    # Handle the case where encoder requires pairwise_state
                    if hasattr(mod.encoder[modality_type], 'forward') and 'pairwise_state' in mod.encoder[modality_type].forward.__code__.co_varnames:
                        # Create a dummy pairwise_state if needed
                        '''modalities.keys:
                        dict_keys(['aatype', 'residue_index', 'seq_length', 'all_atom_positions', 'all_atom_mask', 'resolution', 'is_distillation', 'seq_mask', 'msa_mask', 'msa_row_mask', 'atom14_atom_exists', 'residx_atom14_to_atom37', 'residx_atom37_to_atom14', 'atom37_atom_exists', 'atom14_gt_exists', 'atom14_gt_positions', 'atom14_alt_gt_positions', 'atom14_alt_gt_exists', 'atom14_atom_is_ambiguous', 'rigidgroups_gt_frames', 'rigidgroups_gt_exists', 'rigidgroups_group_exists', 'rigidgroups_group_is_ambiguous', 'rigidgroups_alt_gt_frames', 'pseudo_beta', 'pseudo_beta_mask', 'backbone_rigid_tensor', 'backbone_rigid_mask', 'chi_angles_sin_cos', 'chi_mask', 'extra_msa', 'extra_msa_mask', 'extra_msa_row_mask', 'bert_mask', 'true_msa', 'extra_has_deletion', 'extra_deletion_value', 'msa_feat', 'target_feat', 'use_clamped_fape', 'name', 'seqres'])
                        '''
                        if isinstance(modalities, dict):
                            # If modalities is a dict, we need to extract the sequence tensor
                            # Assuming the sequence tensor has key 'sequence' or similar
                            sequence_key = next(k for k in modalities.keys() if 'sequence_state' in k.lower())
                            sequence_tensor = modalities[sequence_key]
                            seq_len = sequence_tensor.shape[1]
                        else:
                            seq_len = modalities.shape[1]
                            
                        pairwise_state = torch.zeros((batch_size, seq_len, seq_len, 1), device=self.device)
                        
                        # Pass the sequence tensor to the encoder
                        if isinstance(modalities, dict):
                            modalities = mod.encoder[modality_type](sequence_tensor, pairwise_state).detach()
                        else:
                            modalities = mod.encoder[modality_type](modalities, pairwise_state).detach()
                    else:
                        # If no pairwise_state needed, handle dictionary input
                        if isinstance(modalities, dict):
                            sequence_key = next(k for k in modalities.keys() if 'sequence_state' in k.lower())
                            sequence_tensor = modalities[sequence_key]
                            modalities = mod.encoder[modality_type](sequence_tensor).detach()
                        else:
                            modalities = mod.encoder[modality_type](modalities).detach()
                else:
                    mod.encoder.eval()
                    # Handle dictionary input for non-ModuleList encoder
                    if isinstance(modalities, dict):
                        sequence_key = next(k for k in modalities.keys() if 'sequence_state' in k.lower())
                        sequence_tensor = modalities[sequence_key]
                        modalities = mod.encoder(sequence_tensor).detach()
                    else:
                        modalities = mod.encoder(modalities).detach()

        # shapes and device
        if isinstance(modalities, tuple):
            _, tokens = modalities  # s_s, s_z
        else:
            tokens = modalities

        batch, device = tokens.shape[0], tokens.device

        # times
        if not exists(times):
            times = torch.rand((batch,), device = device)

        if return_loss:
            if requires_velocity_consistency:
                orig_times = times.clone()
                times = times * (1. - velocity_consistency_delta_time) # make sure times are max of 1. - small delta, for velocity consistency

            padded_times = append_dims(times, tokens.ndim - 1)

            noise = torch.randn_like(tokens)

            noised_tokens = padded_times * tokens + (1. - padded_times) * noise

            flow = tokens - noise

        else:
            noised_tokens = tokens

        # from latent to model tokens
        noised_tokens = mod.latent_to_model(noised_tokens)

        # axial positions
        if mod.add_pos_emb:
            assert exists(mod.num_dim), f'modality_num_dim must be set for modality {modality_type} if further injecting axial positional embedding'

            _, *axial_dims, _ = noised_tokens.shape

            assert len(axial_dims) == mod.num_dim, f'received modalities of ndim {len(axial_dims)} but expected {mod.num_dim}'

        # maybe transform
        noised_tokens, inverse_pack_axial_dims = pack_one_with_inverse(noised_tokens, 'b * d')

        # maybe add axial pos emb
        if mod.add_pos_emb:
            axial_pos_emb = mod.pos_emb_mlp(tensor(axial_dims), flatten = True)
            noised_tokens = noised_tokens + axial_pos_emb

        # attention
        embed = self.transformer(
            noised_tokens,
            times = times,
            modality_only = True,
        )

        embed = inverse_pack_axial_dims(embed)

        pred_flow = mod.model_to_latent(embed)

        if not return_loss:
            return pred_flow

        # flow loss
        flow_loss = F.mse_loss(pred_flow, flow)

        # maybe velocity consistency loss
        velocity_loss = self.zero

        if requires_velocity_consistency:
            with torch.no_grad():
                flow_with_delta_time = velocity_consistency_ema_model.forward_modality(
                    modalities = modalities,
                    modality_type = modality_type,
                    times = orig_times + velocity_consistency_delta_time,
                    encode_modality = False, # modality already encoded
                    return_loss = False
                )

            velocity_loss = F.mse_loss(flow, flow_with_delta_time)

        # maybe recon loss
        recon_loss = self.zero

        if self.has_recon_loss:
            assert encode_modality

            recon = noise + pred_flow * (1. - padded_times)

            if exists(mod.decoder):
                with torch.no_grad():
                    if isinstance(mod.decoder, nn.Module):
                        mod.decoder.eval()
                        recon = mod.decoder(recon)
                    else:
                        recon = mod.decoder(recon)

            recon_loss = F.mse_loss(
                recon,
                orig_modalities
            )

        # total loss
        total_loss = (
            flow_loss +
            velocity_loss * self.velocity_consistency_loss_weight +
            recon_loss * self.reconstruction_loss_weight
        )

        if not return_loss_breakdown:
            return total_loss

        return total_loss, (flow_loss, velocity_loss, recon_loss)

    @torch.no_grad()
    @eval_decorator
    @typecheck
    def generate_modality_only(
        self,
        batch_size: int = 1,
        modality_type: int | None = None,
        fixed_modality_shape: tuple[int, ...] | None = None,
        modality_steps = 16,
        return_unprocessed_modalities = False
    ) -> Tensor:
        
        device = self.device

        if self.num_modalities > 1:
            assert exists(modality_type), '`modality_type` must be explicitly passed in on forward when training on greater than 1 modality'

        mod = self.get_modality_info(modality_type)

        modality_shape = default(fixed_modality_shape, mod.default_shape)

        assert exists(modality_shape)

        noise = torch.randn((batch_size, *modality_shape, mod.dim_latent), device = device)

        if mod.channel_first_latent:
            noise = rearrange(noise, 'b ... d -> b d ...')

        def ode_step_fn(step_times, denoised):

            step_times = repeat(step_times, ' -> b', b = batch_size)

            flow = self.forward_modality(
                denoised,
                times = step_times,
                modality_type = modality_type,
                encode_modality = False,
                return_loss = False
            )

            return flow

        times = torch.linspace(0., 1., modality_steps, device = device)
        trajectory = self.odeint_fn(ode_step_fn, noise, times)

        # add the sampled modality tokens

        sampled_modality = trajectory[-1]

        # decode

        if exists(mod.decoder):
            mod.decoder.eval()
            sampled_modality = mod.decoder(sampled_modality)
            if isinstance(sampled_modality, tuple):
                seq_logits, coords_pred = sampled_modality
                return seq_logits, coords_pred
            elif isinstance(sampled_modality, dict):
                return sampled_modality
        return sampled_modality

    @typecheck
    def forward_seq_coord(
        self,
        seq_tensor: Int['b n'],
        coords_tensor: Float['b n 3'],
        batch: dict,
        times: Float['b'] | None = None,
        modality_type: int | None = None,
        return_loss = True,
        return_loss_breakdown = False
    ) -> Scalar | tuple:
        """
        Forward pass for protein sequence and coordinate data.
        This method handles the encoding of protein data and flow matching training.
        
        Args:
            seq_tensor: Tensor of amino acid sequence indices [batch, seq_len]
            coords_tensor: Tensor of 3D coordinates [batch, seq_len, 3]
            times: Optional time values for the flow matching process [batch]
            modality_type: Modality type index (default: 0)
            return_loss: Whether to return the loss
            return_loss_breakdown: Whether to return a breakdown of loss components
            
        Returns:
            Loss value or tuple of outputs depending on parameters
        """
        batch, device = seq_tensor.shape[0], seq_tensor.device
        
        # 使用 modality_encoder 将序列和坐标编码为潜在表示
        latent, mu, logvar = self.modality_encoder[0](seq_tensor, coords_tensor)

        # 计算KL散度损失
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
        
        # 将潜在表示输入到flow matching训练中
        if return_loss:
            flow_loss = self.forward_modality(
                modalities=latent,
                times=times,
                modality_type=modality_type,
                encode_modality=False,  # 已经编码过了
                return_loss=True,
                return_loss_breakdown=False
            )
            
            # 计算总损失
            total_loss = flow_loss + 0.01 * kl_loss
            
            if not return_loss_breakdown:
                return total_loss
            
            return total_loss, (flow_loss, kl_loss)
        else:
            # 用于生成（推理）
            return self.forward_modality(
                modalities=latent,
                times=times,
                modality_type=modality_type,
                encode_modality=False,
                return_loss=False
            )
        
    @typecheck
    def forward(
        self,
        modalities: (
            list[ModalitySample] |
            Int['b n'] |
            Float['b ...']
        ),
        times: Float['b m'] | None = None,
        num_modalities_to_times_fn: Callable[[Int['b']], Float['b m']] | None = None, # allows a researcher to customize the times (noise level) based on the modality lengths in a given sample 
        modality_type: int | None = None,
        cache: Tensor | None = None,
        decode_length: int | None = None,
        decoding_text_or_modality: Literal['text', 'modality'] | None = None,
        velocity_consistency_ema_model: LLMFlow | EMA | None = None,
        velocity_consistency_delta_time = 1e-3,
        return_only_pred_flows = False,
        return_loss = True,
        return_breakdown = False,
        return_embed = False,
        return_kv_cache = False,
    ) -> (
        Float['b _ l'] |
        tuple[Float['b _ d'], GetPredFlows] |
        tuple[tuple[Float['b _ _'], GetPredFlows], Tensor] |
        Scalar |
        tuple[Scalar, LossBreakdown] |
        list[Float['b _ _']]
    ):
        is_decoding = exists(decoding_text_or_modality)

        is_text_only = is_tensor(modalities) and modalities.dtype in (torch.int, torch.long)
        is_modality_only = is_tensor(modalities) and modalities.dtype == torch.float

        # handle ema model being passed in for velocity consistency loss

        if isinstance(velocity_consistency_ema_model, EMA):
            assert isinstance(velocity_consistency_ema_model.ema_model, LLMFlow)
            velocity_consistency_ema_model = velocity_consistency_ema_model.ema_model

        need_velocity_matching = not is_decoding and exists(velocity_consistency_ema_model)

        # return loss

        return_loss &= not (return_embed or is_decoding)

        if is_text_only:

            forward_text_kwargs = dict(
                return_loss = return_loss,
                return_embed = return_embed,
                cache = cache,
                return_kv_cache = return_kv_cache
            )

            return self.forward_text(modalities, **forward_text_kwargs)

        if is_modality_only:
            assert return_loss

            forward_modality_kwargs = dict(
                modality_type = modality_type,
                velocity_consistency_ema_model = velocity_consistency_ema_model
            )

            return self.forward_modality(modalities, **forward_modality_kwargs)

        batch = len(modalities)
        device = self.device
        tensor_ = partial(tensor, device = device)

        # save a copy for ema model for velocity matching

        if need_velocity_matching:
            velocity_modalities = modalities

            if isinstance(velocity_modalities, list):
                velocity_modalities = [modality.copy() for modality in velocity_modalities]

        # add "sentence" start and end tokens when training

        if return_loss or need_velocity_matching:
            modalities = modalities.copy()

            for i, modality in enumerate(modalities):
                modalities[i] = [
                    tensor_([self.sos_id]),
                    *modality,
                    tensor_([self.eos_id])
                ]

        # need axial pos emb

        need_axial_pos_emb = any(self.add_pos_emb)

        # standardize modalities to be tuple - type 0 modality is implicit if not given
        # also store modality lengths for determining noising times

        num_modalities = []

        for batch_modalities in modalities:
            batch_num_modalities = 0

            for ind, modality in enumerate(batch_modalities):
                if is_tensor(modality) and modality.dtype == torch.float:
                    modality = (0, modality)

                if not isinstance(modality, tuple):
                    continue

                modality_type, modality_tensor = modality
                batch_modalities[ind] = modality
                batch_num_modalities += 1

            num_modalities.append(batch_num_modalities)

        num_modalities = tensor_(num_modalities)

        # determine the times

        if not exists(times):
            if is_empty(num_modalities) or num_modalities.amax().item() == 0:
                times = torch.empty((batch, 0), device = device, dtype = torch.float)
            else:
                num_modalities_to_times_fn = default(num_modalities_to_times_fn, default_modality_length_to_time_fn)

                if exists(num_modalities_to_times_fn):
                    times = num_modalities_to_times_fn(num_modalities)
        
        # if needs velocity matching, make sure times are in the range of 0 - (1. - <velocity consistency delta time>)

        if need_velocity_matching:
            orig_times = times.clone()
            times = times * (1. - velocity_consistency_delta_time)

        # process list of text and modalities interspersed with one another

        modality_positions = []
        modality_tokens = []
        modality_pos_emb = []

        text = []

        # auto move all tensors to device of model

        modalities = tree_map_tensor(modalities, lambda t: t.to(device))

        # for all modalities, batch process same shaped modalities of the same type

        if not is_decoding:
            for mod in self.get_all_modality_info():
                encode_fn = default(mod.encoder, nn.Identity())

                with torch.no_grad():
                    encode_fn.eval()
                    modalities = apply_fn_modality_type(encode_fn, modalities, modality_type = mod.modality_type)

        # for parsing out the predicted flow from flattened sequence of tokens coming out of transformer

        flows = defaultdict(list) # store flows for loss

        get_pred_flows: GetPredFlows = defaultdict(list) # functions for parsing modalities from Float['b n d'] for model back to latents or pixel space

        def model_to_pred_flow(batch_index, start_index, modality_length, unpack_fn):

            def inner(embed: Float['b n d'], need_splice = True) -> Float['...']:
                embed = embed[batch_index]

                if need_splice:
                    embed = embed[start_index:(start_index + modality_length)]

                embed = unpack_fn(embed)
                return embed

            return inner

        # for going from predicted flow -> reconstruction

        get_recon_losses: Callable[[Tensor], Tensor] = defaultdict(list)

        def get_recon_loss(noise, times, modality):

            def inner(pred_flow):
                recon_modality = noise + pred_flow * (1. - times)
                return F.mse_loss(modality, recon_modality)

            return inner

        # prepare storing of sizes of all modalities that require axial positions, for delayed application for efficiency

        pos_emb_max_axial_dims: dict[int, list[Tensor]] = defaultdict(list)

        # go through all modality samples and do necessary transform

        for batch_index, batch_modalities in enumerate(modalities):

            modality_index = 0
            batch_modality_positions = []
            batch_modality_tokens = []
            batch_modality_pos_emb = []

            batch_text = []

            offset = 0

            for modality in batch_modalities:
                # if non-text modality detected and not given as a tuple
                # cast to (int, Tensor) where int is defaulted to type 0 (convenience for one modality)

                is_text = not isinstance(modality, tuple)
                is_modality = not is_text

                if is_text:
                    modality_tensor = modality
                else:
                    modality_type, modality_tensor, *_ = modality

                # auto move modality tensor to correct device

                mod = self.get_modality_info(modality_type)

                if is_modality:
                    assert 0 <= modality_type < self.num_modalities, f'received a modality index that is out of range. only {self.num_modalities} modalities specified'

                    channel_dim = 0 if mod.channel_first_latent else -1
                    assert mod.dim_latent == modality_tensor.shape[channel_dim], f'mismatch for modality latent dimension - expected {mod.dim_latent} but received {modality_tensor.shape[-1]} - modality shape is {tuple(modality_tensor.shape)}, perhaps you need to set `channel_first_latent` to the correct value'

                # auto ward against scalars (lone start end tokens)

                if modality_tensor.dtype in (torch.int, torch.long) and modality_tensor.ndim == 0:
                    modality_tensor = rearrange(modality_tensor, '-> 1')

                # handle text

                if is_text:
                    assert modality_tensor.ndim == 1
                    text_length = modality_tensor.shape[0]

                    batch_text.append(modality_tensor)
                    zeros = torch.zeros(text_length, self.dim, device = device)

                    batch_modality_tokens.append(zeros)

                    offset += text_length

                    if need_axial_pos_emb:
                        batch_modality_pos_emb.append(zeros)

                    continue

                # otherwise handle a modality

                # get times for noising the modality

                modality_time = times[batch_index, modality_index]

                # noise

                if return_loss:
                    noise = torch.randn_like(modality_tensor)

                    noised_modality = modality_tensor * modality_time + noise * (1. - modality_time)

                    # the flow is the (data - noise)

                    modality_flow = modality_tensor - noise

                    # append to flow for loss

                    flows[modality_type].append(modality_flow)

                    modality_tensor = noised_modality

                    # store function for deriving reconstruction loss from decoder

                    get_recon_losses[modality_type].append(get_recon_loss(noise, modality_time, modality_tensor))

                # go through maybe encoder

                modality_tensor = add_temp_batch_dim(mod.latent_to_model)(modality_tensor)

                # gather the modality length

                modality_shape_tuple = modality_tensor.shape[:-1]
                modality_length = math.prod(modality_shape_tuple)

                text_tensor = torch.full((modality_length,), -1, device = device) # text is all -1 here, so text labels are not learned on

                # only add modality meta information when not returning embedding, which only occurs when sampling modality

                succeed_modality_tokens = precede_modality_tokens = 0

                if not return_embed:
                    # add the [som] and [eom] tokens for the modality type

                    som_id, eom_id = mod.som_id, mod.eom_id

                    # start by just storing the token length of the modality

                    modality_shape_str = join([*map(str, modality_shape_tuple)], ',')
                    modality_meta_info = self.char_tokenizer(modality_shape_str, device = device)

                    precede_modality_tokens = len(modality_meta_info) + 2
                    succeed_modality_tokens = 1

                    text_tensor = cat((
                        tensor_([self.meta_id]),
                        modality_meta_info,
                        tensor_([som_id]),
                        text_tensor,
                        tensor_([eom_id])
                    ))

                batch_modality_positions.append((modality_type, offset + precede_modality_tokens, modality_length)) # offset + preceding meta tag length (which includes the modality start token)

                # store parsing out back to shape

                modality_tensor, unpack_modality_shape = pack_one_with_inverse(modality_tensor, '* d')

                inverse_fn = model_to_pred_flow(batch_index, offset + precede_modality_tokens, modality_length, unpack_modality_shape)

                get_pred_flows[modality_type].append(inverse_fn)

                # increment offset

                offset += modality_length + precede_modality_tokens + succeed_modality_tokens # +2 due to [som] and [eom] - then account for meta start id and modality shape information (or eventually any meta information about modality)

                modality_tensor = F.pad(modality_tensor, (0, 0, precede_modality_tokens, succeed_modality_tokens))

                batch_modality_tokens.append(modality_tensor)
                batch_text.append(text_tensor)

                # handle axial positional embedding

                if need_axial_pos_emb:

                    if exists(mod.pos_emb_mlp):
                        pos_emb_max_axial_dims[modality_type].append(tensor(modality_shape_tuple))
                        pos_emb = (modality_type, modality_shape_tuple, (precede_modality_tokens, succeed_modality_tokens))

                    else:
                        pos_emb = torch.zeros(text_tensor.shape[0], self.dim, device = device)

                    batch_modality_pos_emb.append(pos_emb)

            text.append(cat(batch_text))

            if need_axial_pos_emb:
                modality_pos_emb.append(batch_modality_pos_emb)

            modality_tokens.append(cat(batch_modality_tokens))
            modality_positions.append(batch_modality_positions)

            modality_index += 1

        if return_loss:
            total_tokens = sum([t.numel() for t in text])

        text = pad_sequence(text, padding_value = -1)

        modality_tokens = pad_sequence(modality_tokens, padding_value = 0.)

        # handle modality positional embedding

        if need_axial_pos_emb:
            pos_emb_max_axial_dims = {mod_type: stack(sizes, dim = -1).amax(dim = -1) for mod_type, sizes in pos_emb_max_axial_dims.items()}
            factorized_pos_emb = {mod_type: self.get_modality_info(mod_type).pos_emb_mlp(max_size, return_factorized = True) for mod_type, max_size in pos_emb_max_axial_dims.items()}

            # lazy evaluate the modality positional embedding from the factorized positional embedding from maximum axial dims

            evaluated_pos_emb = []

            for batch_modality_pos_emb in modality_pos_emb:
                evaluated_batch_pos_emb = []

                for maybe_pos_emb_config in batch_modality_pos_emb:

                    if is_tensor(maybe_pos_emb_config):
                        evaluated_batch_pos_emb.append(maybe_pos_emb_config)
                        continue

                    mod_type, mod_size, padding = maybe_pos_emb_config

                    mod_info = self.get_modality_info(mod_type)
                    mod_factorized_pos_emb = factorized_pos_emb[mod_type]

                    mod_pos_emb = mod_info.pos_emb_mlp.combine_factorized(mod_factorized_pos_emb, mod_size, flatten = True)
                    mod_pos_emb = F.pad(mod_pos_emb, (0, 0, *padding), value = 0.) # handle padding for preceding and succeeding meta tokens

                    evaluated_batch_pos_emb.append(mod_pos_emb)

                evaluated_pos_emb.append(cat(evaluated_batch_pos_emb, dim = -2))

            modality_pos_emb = pad_sequence(evaluated_pos_emb, padding_value = 0.)

        # handle training mode and removal of last token

        if return_loss:
            modality_tokens = modality_tokens[:, :-1]

            if need_axial_pos_emb:
                modality_pos_emb = modality_pos_emb[:, :-1]

        # if returning loss, split text for next token prediction

        if return_loss:
            text, text_labels = text[:, :-1], text[:, 1:]

        # derive is_modality mask for flow on the right tokens + flow loss

        batch, seq_len, device = *text.shape, text.device

        assert len(modality_positions) == batch

        if isinstance(modality_positions, list):
            modality_positions = modality_positions_to_tensor(modality_positions, device = device)

        if modality_positions.shape[-1] == 2: # Int['b m 2'] -> Int['b m 3'] if type is not given (one modality)
            modality_positions = F.pad(modality_positions, (1, 0))

        # for now use dummy padding modality position info if empty (all zeros)

        if modality_positions.numel() == 0:
            modality_positions = F.pad(modality_positions, (0, 0, 0, 1))

        # sort the modalities tensor and sanitize, readying for noising of modalities

        modality_positions, sorted_indices = order_modality_positions_by_seq_offset(modality_positions)

        is_modalities = modality_positions_to_is_modality_mask(seq_len, modality_positions, num_modalities = self.num_modalities, device = device)

        is_any_modality = reduce(is_modalities, 'b t m n -> b n', 'any')

        # embed text

        text = text.masked_fill(text == -1, 0)

        text_tokens = self.text_embed(text)

        # maybe add the axial positional embedding

        if need_axial_pos_emb:
            modality_tokens = modality_tokens + modality_pos_emb

        # intersperse the modalities with the text for the joint transformer + flow system

        tokens = einx.where('b n, b n d, b n d', is_any_modality, modality_tokens, text_tokens)

        # derive rotary positions

        rotary_positions = derive_rotary_positions_from_modality_positions(seq_len, modality_positions)

        rotary_emb = self.rotary_emb(rotary_positions)
        rotary_emb = rearrange(rotary_emb, 'b n d -> b 1 n d')

        # take care of cache

        is_any_modality_when_decoding = None

        if exists(cache):
            assert exists(decode_length), '`decode_length` must be passed in on forward for modality sampling. think of a cleaner way on some future date'
            assert exists(decoding_text_or_modality)

            if decoding_text_or_modality == 'text':
                decode_length = 1

            is_any_modality_when_decoding = decoding_text_or_modality == 'modality'
            modality_positions = None

        # times

        times_per_token = einsum(is_modalities.float(), times, 'b t m n, b m -> b t n')

        times_cond = reduce(times_per_token, 'b t n -> b n', 'sum')

        # attention

        embed, kv_cache = self.transformer(
            tokens,
            times = times_cond,
            rotary_emb = rotary_emb,
            modality_positions = modality_positions,
            is_any_modality = is_any_modality_when_decoding,
            cache = cache,
            decode_length = decode_length,
            return_kv_cache = True
        )

        # early return for embedding for decoding modality

        if return_embed:
            if not return_kv_cache:
                return (embed, get_pred_flows)

            return (embed, get_pred_flows), kv_cache

        # text unembedding

        text_logits = self.to_text_logits(embed)

        if not return_loss:
            if not return_kv_cache:
                return text_logits

            return text_logits, kv_cache

        # flow loss

        pred_flows = []
        recon_losses = []

        for modality_id in range(self.num_modalities):
            mod = self.get_modality_info(modality_id)

            modality_get_pred_flows = get_pred_flows[modality_id]
            modality_get_recon_losses = get_recon_losses[modality_id]

            modality_pred_flows = []
            modality_recon_losses = []

            for get_pred_flow, get_recon_loss in zip(modality_get_pred_flows, modality_get_recon_losses):

                pred_flow = get_pred_flow(embed)
                pred_flow = add_temp_batch_dim(mod.model_to_latent)(pred_flow) 
                modality_pred_flows.append(pred_flow)

                if not return_loss or not self.has_recon_loss:
                    continue

                modality_recon_losses.append(get_recon_loss(pred_flow))

            pred_flows.append(modality_pred_flows)
            recon_losses.append(modality_recon_losses)

        # early return for velocity consistency ema model

        if return_only_pred_flows:
            return pred_flows

        # text autoregressive loss

        text_labels = text_labels.masked_fill(is_any_modality, self.ignore_index)

        text_loss = F.cross_entropy(
            rearrange(text_logits, 'b n l -> b l n'),
            text_labels,
            ignore_index = self.ignore_index
        )

        text_loss_weight = (text_labels != self.ignore_index).sum() / total_tokens

        # calculate flow losses

        flow_losses = []

        modality_loss_weights = []

        for modality_id, (pred_flow, is_one_modality) in enumerate(zip(pred_flows, is_modalities.unbind(dim = 1))):
            mod = self.get_modality_info(modality_id)

            is_one_modality = reduce(is_one_modality, 'b m n -> b n', 'any')
            modality_loss_weight = is_one_modality.sum() / total_tokens

            modality_flows = flows[modality_id]

            pack_pattern = 'd *' if mod.channel_first_latent else '* d'

            modality_pred_flow, _ = pack(pred_flow, pack_pattern)
            modality_flows, _ = pack(modality_flows, pack_pattern)

            flow_loss = F.mse_loss(
                modality_pred_flow,
                modality_flows
            )

            modality_loss_weights.append(modality_loss_weight)

            flow_losses.append(flow_loss)

        modality_loss_weights = stack(modality_loss_weights)

        # only the token positions that are not modalities have autoregressive loss

        total_loss = (
            text_loss * text_loss_weight * self.text_loss_weight +
            (stack(flow_losses) * modality_loss_weights).sum() * self.flow_loss_weight
        )

        # whether to handle velocity consistency
        # for straightening the flow, from consistency flow matching paper https://arxiv.org/abs/2407.02398

        velocity_match_losses = None

        if need_velocity_matching:

            with torch.no_grad():
                velocity_consistency_ema_model.eval()

                ema_pred_flows = velocity_consistency_ema_model(
                    velocity_modalities,
                    times = orig_times + velocity_consistency_delta_time,
                    return_only_pred_flows = True
                )

            velocity_match_losses = []

            for ema_pred_flow, pred_flow in zip(ema_pred_flows, pred_flows):

                pack_pattern = 'd *' if mod.channel_first_latent else '* d'
                pred_flow, _ = pack(pred_flow, pack_pattern)
                ema_pred_flow, _ = pack(ema_pred_flow, pack_pattern)

                velocity_match_loss = F.mse_loss(
                    pred_flow,
                    ema_pred_flow
                )

                velocity_match_losses.append(velocity_match_loss)

            total_loss = (
                total_loss +
                (stack(velocity_match_losses) * modality_loss_weights).sum() * self.velocity_consistency_loss_weight
            )

        # maybe reconstruction loss

        if self.has_recon_loss:

            averaged_recon_losses = []

            for modality_recon_loss in recon_losses:
                averaged_recon_losses.append(sum(modality_recon_loss) / len(modality_recon_loss))

            total_loss = (
                total_loss +
                (stack(averaged_recon_losses) * modality_loss_weights).sum() * self.reconstruction_loss_weight
            )

        # return total loss if no breakdown needed

        if not return_breakdown:
            return total_loss

        return total_loss, LossBreakdown(total_loss, text_loss, flow_losses, velocity_match_losses, recon_losses)


