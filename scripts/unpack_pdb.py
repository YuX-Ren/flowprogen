import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--pdb_dir', type=str, required=True)
parser.add_argument('--outdir', type=str, default='./data')
parser.add_argument('--outcsv', type=str, default='./swissprot_pdb_v4_chains.csv')
parser.add_argument('--num_workers', type=int, default=15)
args = parser.parse_args()

import warnings, tqdm, os, io, logging
import pandas as pd
import numpy as np
from multiprocessing import Pool
import sys;sys.path.append('.')
from dillm.data.data_pipeline import DataPipeline
from openfold.data import mmcif_parsing
from Bio.PDB import PDBParser

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pdb_processing.log'),
        logging.StreamHandler()
    ]
)

pipeline = DataPipeline(template_featurizer=None)
pdb_parser = PDBParser(QUIET=True)

def main():
    if not os.path.exists(args.pdb_dir):
        logging.error(f"PDB directory does not exist: {args.pdb_dir}")
        return
        
    logging.info(f"Processing PDB files in: {args.pdb_dir}")
    
    files = [f for f in os.listdir(args.pdb_dir) 
             if os.path.isfile(os.path.join(args.pdb_dir, f)) 
             and (f.endswith('.pdb') or f.endswith('.ent') or f.endswith('.gz'))]
    
    logging.info(f"Found {len(files)} PDB files")
    
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
    for result in tqdm.tqdm(__map__(unpack_pdb, files), total=len(files)):
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

def unpack_pdb(name):
    path = f"{args.pdb_dir}/{name}"
    print(path)
    data = pipeline.process_pdb(pdb_path=path, alignment_dir=None, is_distillation=False, chain_id=None, _structure_index=None, alignment_index=None)   
    '''
    data: ['aatype', 'between_segment_residues', 'domain_name', 
           'residue_index', 'seq_length', 'sequence', 'all_atom_positions', 
           'all_atom_mask', 'resolution', 'is_distillation']
    '''
    os.makedirs(args.outdir, exist_ok=True)
    out_path = f"{args.outdir}/{name[:-4]}.npz"
    np.savez(out_path, **data)
    out = []
    out.append({
        "name": name[:-4],
        "seqres": data['sequence'][0].decode('utf-8'),
    })
    return out
    
if __name__ == "__main__":
    try:
        main()
        logging.info("PDB processing completed")
    except Exception as e:
        logging.error(f"Error occurred during script execution: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())