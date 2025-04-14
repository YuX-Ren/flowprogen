import pandas as pd
from alphaflow.data.data_modules import OpenFoldSingleDataset, OpenFoldBatchCollator
from alphaflow.config import model_config
from torch.utils.data import DataLoader
config = model_config(
    'initial_training',
    train=True, 
    low_prec=True
)

# trainset = OpenFoldSingleDataset(data_dir='/share/project/xiaohongwang/Datasets/pdb_mmcif/data_npz',
#                                  alignment_dir='/home/hwxiao/mycodes/LLM/DiLLM/alphaflow/alignment_dir/cameo2022',
#                                  pdb_chains=pd.read_csv('/home/hwxiao/mycodes/LLM/DiLLM/alphaflow/splits/cameo2022.csv',
#                                                         index_col='name'),
#                                 config=config.data
#                                 )

## swissprot_pdb_v4
trainset = OpenFoldSingleDataset(data_dir='/share/project/linwenjun/swissprot_pdb_v4_data_npz',
                                 alignment_dir='',
                                 pdb_chains=pd.read_csv('',
                                                        index_col='name'),
                                config=config.data
                                )

train_loader = DataLoader(trainset, 
                          batch_size=1, 
                          collate_fn=OpenFoldBatchCollator(),
                          num_workers=8,
                          pin_memory=True,
                          shuffle=True
                          )

for batch in train_loader:
    print(batch)

data_iter = iter(train_loader)
for i in range(10):
    batch = next(data_iter)
    print(batch)
