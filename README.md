
FlowProGen: Protein Design Based on Large Language Models and Flow Matching    
FlowProGen: Controllable Protein Backbone Generation based on Flow Matching and LLM
FlowProGen: Latent-level Controllable Protein Backbone Generation based on Flow Matching


This code is adapted from two codebases:
 - [Transfusion - Pytorch](https://github.com/lucidrains/transfusion-pytorch), which is a Pytorch implementation of [Transfusion: Predict the Next Token and Diffuse Images with One Multi-Modal Model](https://www.arxiv.org/abs/2408.11039) from Meta AI
 - [AlphaFlow](https://github.com/bjing2016/alphaflow), which is a Pytorch implementation of [AlphaFold Meets Flow Matching for Generating Protein Ensembles](https://arxiv.org/abs/2402.04845).


## Usage

train_latent_only_dillm_esmfold.py


## Installing OpenFold

```
cd FlowProGen
git clone git@github.com:aqlaboratory/openfold.git
python setup.py install
```

```
extra_cuda_flags = [
    '-std=c++17',
    '-maxrregcount=50',
    '-U__CUDA_NO_HALF_OPERATORS__',
    '-U__CUDA_NO_HALF_CONVERSIONS__',
    '--expt-relaxed-constexpr',
    '--expt-extended-lambda'
]
```
如果安装Openfold时报gcc++17错误，记得在setup.py中把'-std=c++14'改为'-std=c++17'

## Error fix:
1. cannot import name 'flash_attn_unpadded_kvpacked_func' from 'flash_attn.flash_attn_interface'
```
from flash_attn.flash_attn_interface import flash_attn_varlen_kvpacked_func
```
2. ModuleNotFoundError: No module named 'torch._six'
```
try:
    from torch._six import inf
except ModuleNotFoundError:
    from torch import inf
```