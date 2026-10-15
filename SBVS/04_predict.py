import os
import json
import argparse
import gc

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from utils import AffinityDataset, collate_fn, build_model, resolve_dims, move_batch_to_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_name', required=True)
    p.add_argument('--test_data_path', default='data/test/processed/test_data.pt')
    p.add_argument('--batch_size', type=int, default=8)
    p.add_argument('--num_workers', type=int, default=0)
    p.add_argument('--chunk_size', type=int, default=10000)
    p.add_argument('--device', default=None)
    p.add_argument('--output_csv', default=None)
    return p.parse_args()


def predict_in_chunks(model, dataset, batch_size, num_workers, device, chunk_size=10000):
    model.eval()
    all_preds, all_smiles = [], []
    total = len(dataset)
    for start in tqdm(range(0, total, chunk_size), desc='Chunks'):
        end = min(start + chunk_size, total)
        loader = DataLoader(Subset(dataset, list(range(start, end))), batch_size=batch_size,
                            shuffle=False, collate_fn=collate_fn, num_workers=num_workers,
                            pin_memory=(device.type == 'cuda'))
        chunk_preds, chunk_smiles = [], []
        with torch.no_grad():
            for batch in loader:
                batch = move_batch_to_device(batch, device)
                output = model(batch)
                chunk_preds.append(output.cpu().numpy())
                chunk_smiles.extend(batch['smiles'])
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
        if chunk_preds:
            all_preds.append(np.concatenate(chunk_preds, axis=0))
            all_smiles.extend(chunk_smiles)
        del loader
        gc.collect()
    preds = np.concatenate(all_preds, axis=0) if all_preds else np.array([])
    return preds, all_smiles


def main():
    args = parse_args()
    device = torch.device(args.device) if args.device else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    project_root = os.path.dirname(os.path.abspath(__file__))
    test_data = torch.load(args.test_data_path, map_location='cpu', weights_only=False)
    num_data = test_data['num_data']
    ids = test_data.get('ids', [f"sample_{i}" for i in range(num_data)])
    other_cols = test_data.get('other_columns', {})
    dims = resolve_dims(test_data)
    print(f"Dims: {dims}, test samples: {num_data}")

    model_dir = os.path.join(project_root, 'models', args.exp_name)
    hyper_path = os.path.join(model_dir, 'hyperparams.json')
    if not os.path.exists(hyper_path):
        alt = os.path.join(project_root, 'results', args.exp_name, 'hyperparams.json')
        if os.path.exists(alt):
            hyper_path = alt
        else:
            raise FileNotFoundError(f"Hyperparameter file not found: {hyper_path}")
    with open(hyper_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    print(f"Model config: d_model={config.get('d_model')} "
          f"nhead={config.get('nhead')} use_last_k={config.get('use_last_k')}")

    model = build_model(config, dims).to(device)
    model_path = os.path.join(model_dir, 'model_best.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
    print(f"Model loaded: {model_path}")

    norm_path = os.path.join(model_dir, 'norm_stats.pt')
    if not os.path.exists(norm_path):
        raise FileNotFoundError(f"Normalization stats not found: {norm_path}")
    norm_stats = torch.load(norm_path, map_location='cpu', weights_only=False)
    mean, std = float(norm_stats['mean']), float(norm_stats['std'])

    test_dataset = AffinityDataset(list(range(num_data)), test_data)
    print("Predicting...")
    preds_norm, smiles_list = predict_in_chunks(
        model, test_dataset, args.batch_size, args.num_workers, device, args.chunk_size)
    preds_original = preds_norm * std + mean

    out_df = pd.DataFrame({
        'id': ids, 'smiles': smiles_list,
        'predicted_affinity_norm': preds_norm.flatten(),
        'predicted_affinity': preds_original.flatten(),
    })
    for col_name, col_values in other_cols.items():
        if len(col_values) == len(out_df):
            out_df[col_name] = col_values
    results_dir = os.path.join(project_root, 'results', args.exp_name)
    os.makedirs(results_dir, exist_ok=True)
    output_path = args.output_csv or os.path.join(results_dir, f'predictions_{args.exp_name}.csv')
    out_df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
    print(out_df.head())


if __name__ == '__main__':
    main()
