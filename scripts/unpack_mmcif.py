import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--mmcif_dir', type=str, required=True, default='/share/project/xiaohongwang/Datasets/pdb_mmcif')
parser.add_argument('--outdir', type=str, default='/share/project/xiaohongwang/Datasets/pdb_mmcif_data_npz')
parser.add_argument('--outcsv', type=str, default='./pdb_mmcif.csv')
parser.add_argument('--num_workers', type=int, default=8)
args = parser.parse_args()

import warnings, tqdm, os, io, logging
import pandas as pd
import numpy as np
from multiprocessing import Pool
import sys;sys.path.append('.')
from dillm.data.data_pipeline import DataPipeline
from openfold.data import mmcif_parsing

pipeline = DataPipeline(template_featurizer=None)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mmcif_processing.log'),
        logging.StreamHandler()
    ]
)

def main():
    if not os.path.exists(args.mmcif_dir):
        logging.error(f"MMCIF directory does not exist: {args.mmcif_dir}")
        return
        
    logging.info(f"Processing MMCIF files in: {args.mmcif_dir}")
    
    dirs = os.listdir(args.mmcif_dir)
    files = []
    for dir in dirs:
        try:
            dir_path = f"{args.mmcif_dir}/{dir}"
            if os.path.isdir(dir_path):
                files.extend([f"{f}" for f in os.listdir(dir_path)])
        except Exception as e:
            logging.error(f"Error accessing directory {dir}: {str(e)}")
    
    logging.info(f"Found {len(files)} MMCIF files")
    
    if args.num_workers > 1:
        logging.info(f"Using {args.num_workers} worker processes")
        p = Pool(args.num_workers)
        p.__enter__()
        __map__ = p.imap
    else:
        logging.info("Using single process mode")
        __map__ = map
        
    results = []
    error_count = 0
    for result in tqdm.tqdm(__map__(unpack_mmcif, files), total=len(files)):
        if result is not None:
            if isinstance(result, list) and len(result) > 0:
                results.append(result)
            else:
                error_count += 1
                
    logging.info(f"Processing completed. Success: {len(results)}, Failed: {error_count}")
    
    if args.num_workers > 1:
        p.__exit__(None, None, None)
        
    info = []
    for result in results:
        info.extend(result)
    
    if len(info) > 0:
        df = pd.DataFrame(info).set_index('name')
        df.to_csv(args.outcsv)
        logging.info(f"Save results to {args.outcsv}")
    else:
        logging.warning("No files were successfully processed, no CSV was created")
    
def unpack_mmcif(name):
    if len(name) >= 4:
        path = f"{args.mmcif_dir}/{name[1:3]}/{name}"
    else:
        path = f"{args.mmcif_dir}/{name}"
    with open(path, 'r') as f:
        mmcif_string = f.read()

    
    mmcif = mmcif_parsing.parse(
        file_id=name[:-4], mmcif_string=mmcif_string
    )
    if mmcif.mmcif_object is None:
        logging.info(f"Could not parse {name}. Skipping...")
        return []
    else:
        mmcif = mmcif.mmcif_object

    out = []
    for chain, seq in mmcif.chain_to_seqres.items():
        out.append({
            "name": f"{name[:-4]}_{chain}",
            "release_date":  mmcif.header["release_date"],
            "seqres": seq,
            "resolution": mmcif.header["resolution"],
        })
        
        data = pipeline.process_mmcif(mmcif=mmcif, chain_id=chain)
        '''
        data: ['aatype', 'between_segment_residues', 'domain_name', 
               'residue_index', 'seq_length', 'sequence', 'all_atom_positions', 
               'all_atom_mask', 'resolution', 'release_date', 'is_distillation']
        '''
        out_dir = f"{args.outdir}/{name[1:3]}"
        os.makedirs(out_dir, exist_ok=True)
        out_path = f"{out_dir}/{name[:-4]}_{chain}.npz"
        np.savez(out_path, **data)
        
    
    return out
    
if __name__ == "__main__":
    main()