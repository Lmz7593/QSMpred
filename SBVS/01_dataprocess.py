import os
import json
import argparse

import torch
import pandas as pd
from tqdm import tqdm

from utils import (
    get_molecule_fingerprint, extract_esm_features, extract_chemberta_embeddings,
)


def main():
    parser = argparse.ArgumentParser(description='Data preprocessing (pruned pipeline)')
    parser.add_argument('--mode', required=True, choices=['train', 'test'])
    parser.add_argument('--input_csv', required=True)
    parser.add_argument('--output_dir', default=None,
                        help='default: train -> data/qs/processed, test -> data/test/processed')
    parser.add_argument('--esm_model', default='esm2_t30_150M_UR50D')
    parser.add_argument('--esm_layers', default='29,30',
                        help='ESM layers to cache (1-based); the model averages the last '
                             '`use_last_k` of them, default 2')
    parser.add_argument('--fp_bits', type=int, default=256, help='ECFP bits, 0=disable')
    parser.add_argument('--use_bert', action='store_true', help='also cache ChemBERTa CLS embeddings')
    parser.add_argument('--bert_model_path', default=None,
                        help='local ChemBERTa dir, default <project>/.models/ChemBERTa-77M-MLM')
    parser.add_argument('--bert_batch_size', type=int, default=32)
    parser.add_argument('--protein_col', default='protein', help='protein sequence column name')
    parser.add_argument('--smiles_col', default='smiles', help='SMILES column name')
    parser.add_argument('--affinity_col', default='affinity', help='affinity column name')
    parser.add_argument('--max_residues', type=int, default=0,
                        help='truncate protein sequences (0 = keep full; ESM-2 supports 1022)')
    args = parser.parse_args()

    project_root = os.path.dirname(os.path.abspath(__file__))
    os.environ['TORCH_HOME'] = os.path.join(project_root, '.models')

    if args.output_dir is None:
        args.output_dir = os.path.join(
            project_root, 'data', 'qs' if args.mode == 'train' else 'test', 'processed')
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output dir: {args.output_dir}")

    df = pd.read_csv(args.input_csv)
    rename_map = {}
    for src, dst in ((args.protein_col, 'protein'), (args.smiles_col, 'smiles'),
                     (args.affinity_col, 'affinity')):
        if src and src != dst and src in df.columns:
            rename_map[src] = dst
    if rename_map:
        print(f"Renaming columns: {rename_map}")
        df = df.rename(columns=rename_map)

    required = ['protein', 'smiles', 'affinity'] if args.mode == 'train' else ['protein', 'smiles']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    if args.max_residues > 0:
        lengths = df['protein'].astype(str).str.len()
        n_trunc = int((lengths > args.max_residues).sum())
        print(f"Protein truncation: {n_trunc}/{len(df)} rows longer than {args.max_residues} "
              f"residues (max {int(lengths.max())}) -> truncated to {args.max_residues}")
        df['protein'] = df['protein'].astype(str).str.slice(0, args.max_residues)

    ids = df['id'].tolist() if 'id' in df.columns else [f"sample_{i}" for i in range(len(df))]
    sequences = df['protein'].tolist()
    smiles_list = df['smiles'].tolist()
    labels = df['affinity'].tolist() if args.mode == 'train' else None

    esm_dir = os.path.join(args.output_dir, 'esm_features')
    layers = [int(x) for x in args.esm_layers.split(',')]
    feat_paths = extract_esm_features(sequences, esm_dir, args.esm_model, layers=layers)

    use_fp = args.fp_bits > 0
    mol_fingerprints = []
    if use_fp:
        n_invalid = 0
        for smi in tqdm(smiles_list, desc='ECFP'):
            fp = get_molecule_fingerprint(smi, fp_bits=args.fp_bits)
            if fp is None:
                n_invalid += 1
                fp = torch.zeros(args.fp_bits)
            mol_fingerprints.append(fp)
        if n_invalid:
            print(f"Warning: {n_invalid} invalid SMILES replaced by zero fingerprints")

    mol_bert_embeddings = None
    if args.use_bert:
        bert_path = args.bert_model_path or os.path.join(project_root, '.models',
                                                         'ChemBERTa-77M-MLM')
        if not os.path.isdir(bert_path):
            raise FileNotFoundError(f"ChemBERTa model dir not found: {bert_path}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Extracting ChemBERTa embeddings (device={device})...")
        mol_bert_embeddings = extract_chemberta_embeddings(
            smiles_list, bert_path, device, batch_size=args.bert_batch_size)

    data_dict = {
        'ids': ids,
        'smiles': smiles_list,
        'prot_feat_paths': feat_paths,
        'num_data': len(df),
    }
    if use_fp:
        data_dict['mol_fingerprints'] = mol_fingerprints
    if mol_bert_embeddings is not None:
        data_dict['mol_bert_embeddings'] = mol_bert_embeddings
    if args.mode == 'train':
        data_dict['labels'] = labels
    else:
        data_dict['other_columns'] = {
            col: df[col].tolist() for col in df.columns if col not in ['id', 'smiles']
        }

    out_file = 'dataset.pt' if args.mode == 'train' else 'test_data.pt'
    save_path = os.path.join(args.output_dir, out_file)
    torch.save(data_dict, save_path)
    with open(os.path.join(args.output_dir, 'preprocess_config.json'), 'w',
              encoding='utf-8') as f:
        json.dump(vars(args), f, indent=2, ensure_ascii=False)
    print(f"Data saved to {save_path}")


if __name__ == '__main__':
    main()
