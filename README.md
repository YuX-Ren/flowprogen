
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
git checkout 103d0370ad9ce07579c20fa9c889a632f9b16618  (we need to use this commit version)
python setup.py install
or directly (pip install 'openfold @ git+ssh://git@github.com/aqlaboratory/openfold.git@103d037')
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
2. Will encounter the error "ModuleNotFoundError: No module named 'torch._six'" in lower version deepspeed, like 0.7.0 or 0.5.10
```
try:
    from torch._six import inf
except ModuleNotFoundError:
    from torch import inf
```

# Three key files
pdb_cluster
pdb_mmcif.csv
pdb_mmcif_msa.csv

# install TMscore cli
```
wget https://zhanggroup.org/TM-score/TMscore.cpp
g++ -O3 -static -o TMscore TMscore.cpp
mv TMscore /usr/local/bin/
chmod +x /usr/local/bin/TMscore
```