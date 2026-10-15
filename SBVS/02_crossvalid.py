import os
import json
import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupKFold

from utils import (
    AffinityDataset, collate_fn, build_model, resolve_dims, set_seed,
    compute_all_metrics, EarlyStopping, train_epoch, validate,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_name', required=True)
    p.add_argument('--dataset_path', default='data/qs/processed/dataset.pt')
    p.add_argument('--n_splits', type=int, default=5)
    p.add_argument('--folds', default=None,
                   help='comma-separated fold ids to run, e.g. 0,3 (default: all folds)')
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--d_model', type=int, default=256)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--patience', type=int, default=20)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--weight_decay', type=float, default=1e-5)
    p.add_argument('--nhead', type=int, default=2, help='cross-attention heads')
    p.add_argument('--use_last_k', type=int, default=2,
                   help='average the last k cached ESM layers')
    p.add_argument('--rank_loss_weight', type=float, default=1.0)
    p.add_argument('--var_loss_weight', type=float, default=1.0)
    p.add_argument('--seed', type=int, default=10)
    p.add_argument('--init_seed', type=int, default=None,
                   help='weight-init seed; fold split stays controlled by --seed. '
                        'Default (None) keeps the original behaviour: no per-fold '
                        're-seeding, i.e. init and fold split share --seed.')
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
    print(f"Dims: {dims}")

    smiles_list = dataset['smiles']
    unique_smiles = sorted(set(smiles_list))
    sm_to_group = {sm: i for i, sm in enumerate(unique_smiles)}
    groups = np.array([sm_to_group[sm] for sm in smiles_list])
    all_indices = np.arange(len(smiles_list))
    labels_arr = np.array([dataset['labels'][i] for i in all_indices])
    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(all_indices)
    gkf = GroupKFold(n_splits=args.n_splits)
    splits = list(gkf.split(shuffled, groups=groups[shuffled]))
    fold_ids = ([int(x) for x in args.folds.split(',')] if args.folds
                else list(range(len(splits))))

    project_root = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(project_root, 'models', args.exp_name)
    results_dir = os.path.join(project_root, 'results', args.exp_name)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    assignment = np.full(len(smiles_list), -1, dtype=int)
    for fold, (_, val_idx) in enumerate(splits):
        assignment[shuffled[val_idx]] = fold
    pd.DataFrame({'sample_idx': np.arange(len(smiles_list)), 'fold': assignment}).to_csv(
        os.path.join(results_dir, 'fold_assignments.csv'), index=False)
    for fold, (_, val_idx) in enumerate(splits):
        val_orig = shuffled[val_idx]
        print(f"  fold{fold} val: n={len(val_orig)}, label mean={labels_arr[val_orig].mean():.3f}, "
              f"std={labels_arr[val_orig].std():.3f}")

    config = vars(args)
    config['device'] = str(device)
    config.update(dims)

    fold_metrics_list = []
    for fold in fold_ids:
        train_idx, val_idx = splits[fold]
        print(f"\n====== Fold {fold + 1} ======")
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

        if args.init_seed is not None:
            # keep the fold split fixed (--seed) while varying the initialisation
            set_seed(args.init_seed + fold)
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

        best_val_loss, best_epoch = float('inf'), 0
        best_path = os.path.join(models_dir, f'fold{fold}_best.pth')
        epoch_log = []
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
                torch.save(model.state_dict(), best_path)
            early_stopper(val_loss)
            if early_stopper.early_stop:
                break

        pd.DataFrame(epoch_log).to_csv(
            os.path.join(results_dir, f'fold{fold}_epoch_log.csv'), index=False)

        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False))
        _, preds_norm, _ = validate(model, val_loader, device)
        preds = preds_norm * std + mean
        labels_true = np.array([dataset['labels'][i] for i in val_idx])
        metrics = compute_all_metrics(labels_true, preds)
        metrics.update({'fold': fold, 'best_epoch': best_epoch})
        fold_metrics_list.append(metrics)
        pd.DataFrame([metrics]).to_csv(
            os.path.join(results_dir, f'fold{fold}_metrics.csv'), index=False)

        pred_dir = os.path.join(results_dir, 'predictions')
        os.makedirs(pred_dir, exist_ok=True)
        pd.DataFrame({'true': labels_true, 'pred': preds,
                      'smiles': [smiles_list[i] for i in val_idx]}).to_csv(
            os.path.join(pred_dir, f'fold{fold}_pred.csv'), index=False)
        print(f"Fold {fold} metrics: {metrics}")

        del train_dataset, val_dataset, train_loader, val_loader, model, optimizer, scheduler
        torch.cuda.empty_cache()

    metrics_df = pd.DataFrame(fold_metrics_list).sort_values('fold').reset_index(drop=True)
    metrics_df.to_csv(os.path.join(results_dir, 'fold_metrics.csv'), index=False)
    summary = {}
    for col in ['MSE', 'RMSE', 'MAE', 'R2', 'PearsonR', 'SpearmanR', 'C-index']:
        summary[f'{col}_mean'] = float(metrics_df[col].mean())
        summary[f'{col}_std'] = float(metrics_df[col].std())
    pd.DataFrame([summary]).to_csv(os.path.join(results_dir, 'summary.csv'), index=False)

    with open(os.path.join(results_dir, 'hyperparams.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    with open(os.path.join(models_dir, 'hyperparams.json'), 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print("===== Cross-validation done =====")
    print(summary)


if __name__ == '__main__':
    main()
