import os
import json
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit

from utils import (
    AffinityDataset, collate_fn, build_model, resolve_dims, set_seed,
    train_epoch, validate, EarlyStopping,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_name', required=True)
    p.add_argument('--dataset_path', default='data/qs/processed/dataset.pt')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--d_model', type=int, default=256)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--val_ratio', type=float, default=0.1,
                   help='SMILES-grouped validation ratio for early stopping')
    p.add_argument('--patience', type=int, default=20, help='early stopping patience')
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--nhead', type=int, default=2, help='cross-attention heads')
    p.add_argument('--use_last_k', type=int, default=2,
                   help='average the last k cached ESM layers')
    p.add_argument('--rank_loss_weight', type=float, default=1.0)
    p.add_argument('--var_loss_weight', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=11)
    p.add_argument('--protein_lr_multiplier', type=float, default=2.0)
    p.add_argument('--clip_norm', type=float, default=1.0)
    p.add_argument('--scheduler_factor', type=float, default=0.5)
    p.add_argument('--scheduler_patience', type=int, default=8)
    p.add_argument('--device', default=None)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    dataset = torch.load(args.dataset_path, map_location='cpu', weights_only=False)
    dims = resolve_dims(dataset)
    num_data = dataset['num_data']
    all_indices = np.arange(num_data)
    print(f"Dims: {dims}, samples: {num_data}, val ratio: {args.val_ratio}")

    gss = GroupShuffleSplit(n_splits=1, test_size=args.val_ratio, random_state=args.seed)
    train_idx, val_idx = next(gss.split(all_indices, groups=np.array(dataset['smiles'])))
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}")

    train_dataset = AffinityDataset(train_idx, dataset)
    val_dataset = AffinityDataset(val_idx, dataset)
    train_labels = np.array([dataset['labels'][i] for i in train_idx])
    mean, std = float(np.mean(train_labels)), float(np.std(train_labels))
    if std == 0:
        std = 1.0
    train_dataset.labels = [(l - mean) / std for l in train_dataset.labels]
    val_dataset.labels = [(l - mean) / std for l in val_dataset.labels]
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn)

    config = vars(args)
    config['device'] = str(device)
    config.update(dims)
    config['num_data'] = num_data

    model = build_model(config, dims).to(device)
    param_groups = [
        {'params': model.protein_encoder.parameters(),
         'lr': args.lr * args.protein_lr_multiplier},
        {'params': model.mol_encoder.parameters(), 'lr': args.lr},
        {'params': model.interaction.parameters(), 'lr': args.lr},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=args.scheduler_factor,
        patience=args.scheduler_patience)
    early_stopper = EarlyStopping(patience=args.patience, verbose=True)

    project_root = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(project_root, 'models', args.exp_name)
    results_dir = os.path.join(project_root, 'results', args.exp_name)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    best_model_path = os.path.join(models_dir, 'model_best.pth')
    epoch_log, best_val_loss, best_epoch = [], float('inf'), 0

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, device,
            rank_weight=args.rank_loss_weight, var_weight=args.var_loss_weight,
            clip_norm=args.clip_norm)
        val_loss, _, _ = validate(model, val_loader, device)
        print(f"Epoch {epoch}: Train loss={train_loss:.4f}, Val loss={val_loss:.4f}")
        epoch_log.append({'epoch': epoch, 'train_loss': train_loss, 'val_loss': val_loss})
        scheduler.step(val_loss)
        if val_loss < best_val_loss:
            best_val_loss, best_epoch = val_loss, epoch
            torch.save(model.state_dict(), best_model_path)
        early_stopper(val_loss)
        if early_stopper.early_stop:
            print(f"Early stop at epoch {epoch} (best epoch {best_epoch}, "
                  f"val loss {best_val_loss:.4f})")
            break

    torch.save({'mean': mean, 'std': std}, os.path.join(models_dir, 'norm_stats.pt'))
    pd.DataFrame(epoch_log).to_csv(os.path.join(results_dir, 'training_log.csv'), index=False)
    with open(os.path.join(models_dir, 'hyperparams.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    with open(os.path.join(results_dir, 'hyperparams.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Best model (epoch {best_epoch}) saved to {best_model_path}")
    print("===== Training finished =====")


if __name__ == '__main__':
    main()
