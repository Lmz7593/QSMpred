import os
import random
import functools
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


# --------------------------------------------------------------- feature side
def extract_esm_features(sequences, output_dir, model_name='esm2_t30_150M_UR50D',
                         layers=(29, 30)):
    """Extract per-residue ESM-2 features, one cached file per unique sequence."""
    import esm
    from tqdm import tqdm

    unique_seqs = list(dict.fromkeys(sequences))
    seq_to_id = {seq: i for i, seq in enumerate(unique_seqs)}
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    os.makedirs(output_dir, exist_ok=True)

    layers = tuple(sorted(set(int(l) for l in layers)))
    feat_paths = {}
    for seq in tqdm(unique_seqs, desc='ESM-2'):
        feat_path = os.path.join(output_dir, f'prot_{seq_to_id[seq]}_layers.pt')
        if not os.path.exists(feat_path):
            _, _, batch_tokens = batch_converter([('prot', seq)])
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=list(layers))
            layer_repr = {layer: results['representations'][layer][0, 1:-1, :].cpu()
                          for layer in layers}
            torch.save(layer_repr, feat_path, _use_new_zipfile_serialization=False)
        feat_paths[seq] = feat_path
    return [feat_paths[seq] for seq in sequences]


def extract_chemberta_embeddings(smiles, model_path, device, batch_size=32, max_length=512):
    """Extract ChemBERTa CLS embeddings as a list of CPU tensors."""
    from transformers import AutoTokenizer, AutoModel
    from tqdm import tqdm
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)
    model.eval()
    model.to(device)

    embeddings = []
    for start in tqdm(range(0, len(smiles), batch_size), desc='ChemBERTa'):
        chunk = smiles[start:start + batch_size]
        inputs = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True,
                           max_length=max_length)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings.extend(outputs.last_hidden_state[:, 0, :].cpu())
    return embeddings


def get_molecule_fingerprint(smiles, fp_bits=256):
    """ECFP (Morgan, radius 2) bit vector; None for invalid SMILES."""
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=fp_bits)
    ecfp = generator.GetFingerprint(mol)
    fp = np.zeros(fp_bits, dtype=np.float32)
    for i in range(fp_bits):
        fp[i] = ecfp.GetBit(i)
    return torch.tensor(fp, dtype=torch.float)


@functools.lru_cache(maxsize=8192)
def _load_prot_feature(path):
    return torch.load(path, map_location='cpu', weights_only=False)


# ---------------------------------------------------------------- dataset side
class AffinityDataset(Dataset):
    def __init__(self, indices, dataset_dict):
        self.indices = list(indices)
        self.prot_feat_paths = [dataset_dict['prot_feat_paths'][i] for i in self.indices]
        self.mol_fingerprints = dataset_dict.get('mol_fingerprints')
        self.mol_fingerprints = ([self.mol_fingerprints[i] for i in self.indices]
                                 if self.mol_fingerprints else None)
        bert = dataset_dict.get('mol_bert_embeddings')
        self.mol_bert = [bert[i] for i in self.indices] if bert else None
        self.labels = ([dataset_dict['labels'][i] for i in self.indices]
                       if 'labels' in dataset_dict else [0.0] * len(self.indices))
        self.smiles = [dataset_dict['smiles'][i] for i in self.indices]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        item = {
            'prot_feats': _load_prot_feature(self.prot_feat_paths[idx]),
            'label': torch.tensor(self.labels[idx], dtype=torch.float),
            'smiles': self.smiles[idx],
        }
        if self.mol_fingerprints is not None:
            item['mol_fp'] = self.mol_fingerprints[idx]
        if self.mol_bert is not None:
            item['mol_bert_embed'] = self.mol_bert[idx]
        return item


def collate_fn(batch):
    prot_feats_list = [item['prot_feats'] for item in batch]
    labels = torch.stack([item['label'] for item in batch])
    smiles = [item['smiles'] for item in batch]

    max_len = max(feat[list(feat.keys())[0]].size(0) for feat in prot_feats_list)

    padded_feats, feat_masks = [], []
    for feats_dict in prot_feats_list:
        layer_tensors = [feats_dict[layer] for layer in sorted(feats_dict.keys())]
        padded_layers = []
        for t in layer_tensors:
            pad_len = max_len - t.size(0)
            padded_layers.append(t if pad_len <= 0
                                 else torch.cat([t, torch.zeros(pad_len, t.size(1))], dim=0))
        padded_feats.append(torch.cat(padded_layers, dim=-1))
        cur_len = layer_tensors[0].size(0)
        mask = torch.ones(max_len, dtype=torch.bool)
        mask[cur_len:] = False
        feat_masks.append(mask)

    out = {
        'prot_feats': torch.stack(padded_feats),
        'prot_feat_mask': torch.stack(feat_masks),
        'labels': labels,
        'smiles': smiles,
    }
    if 'mol_fp' in batch[0]:
        out['mol_fp'] = torch.stack([item['mol_fp'] for item in batch])
    if 'mol_bert_embed' in batch[0]:
        out['mol_bert_embed'] = torch.stack([item['mol_bert_embed'] for item in batch])
    return out


# ----------------------------------------------------------------- model side
class ProteinEncoder(nn.Module):
    """ESM-2 residue features -> per-residue embedding.

    Averages the last `use_last_k` stored layers, then applies a two-layer MLP.
    Returns the residue-level tensor; pooling happens inside the interaction
    module where the padding mask is available.
    """

    def __init__(self, num_feat_layers, per_layer_dim, d_model, dropout=0.3,
                 use_last_k=2):
        super().__init__()
        self.num_feat_layers = num_feat_layers
        self.per_layer_dim = per_layer_dim
        self.use_last_k = max(1, min(use_last_k, num_feat_layers))
        self.feat_mlp = nn.Sequential(
            nn.Linear(per_layer_dim, d_model), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_model, d_model))

    def forward(self, prot_feats):
        B, L, _ = prot_feats.shape
        feats = prot_feats.view(B, L, self.num_feat_layers, self.per_layer_dim)
        esm = feats[:, :, -self.use_last_k:, :].mean(dim=2)
        return self.feat_mlp(esm)


class MolecularEncoder(nn.Module):
    """Gated fusion of the ECFP fingerprint and the ChemBERTa CLS embedding."""

    def __init__(self, fp_dim, bert_dim, d_model, dropout=0.3):
        super().__init__()
        self.use_fp = fp_dim > 0
        self.use_bert = bert_dim > 0
        if not (self.use_fp or self.use_bert):
            raise ValueError('At least one of fingerprint / ChemBERTa is required')
        if self.use_fp:
            self.fp_proj = nn.Sequential(
                nn.Linear(fp_dim, d_model), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(d_model, d_model))
        if self.use_bert:
            self.bert_proj = nn.Sequential(
                nn.Linear(bert_dim, d_model), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(d_model, d_model))
        self.n_sources = int(self.use_fp) + int(self.use_bert)
        self.gate = nn.Linear(d_model * self.n_sources, self.n_sources)

    def forward(self, fingerprints=None, bert_embed=None):
        sources = []
        if self.use_fp:
            sources.append(self.fp_proj(fingerprints))
        if self.use_bert:
            sources.append(self.bert_proj(bert_embed))
        if len(sources) == 1:
            return sources[0]
        gate = torch.sigmoid(self.gate(torch.cat(sources, dim=1)))
        return gate[:, :1] * sources[0] + gate[:, 1:] * sources[1]


class CrossResidueInteraction(nn.Module):

    def __init__(self, d_model, nhead=2, dropout=0.3):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, batch_first=True,
                                                dropout=dropout)
        self.bilinear = nn.Bilinear(d_model, d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(d_model, 1)

    @staticmethod
    def masked_mean(x, mask):
        pm = mask.unsqueeze(-1).float()
        return (x * pm).sum(dim=1) / pm.sum(dim=1).clamp(min=1e-9)

    def forward(self, prot_residues, prot_mask, mol_vec):
        mol_seq = mol_vec.unsqueeze(1)
        attn_out, _ = self.cross_attn(query=mol_seq, key=prot_residues,
                                      value=prot_residues, key_padding_mask=~prot_mask)
        mol_updated = (mol_seq + self.dropout(attn_out)).squeeze(1)
        prot_pooled = self.masked_mean(prot_residues, prot_mask)
        return self.fc_out(self.bilinear(prot_pooled, mol_updated)).squeeze(-1)


class AffinityPredictor(nn.Module):
    def __init__(self, config, dims):
        super().__init__()
        d_model = config.get('d_model', 256)
        dropout = config.get('dropout', 0.3)
        nhead = config.get('nhead', 2)

        self.protein_encoder = ProteinEncoder(
            dims.get('num_feat_layers', 2), dims.get('per_layer_dim', 640),
            d_model, dropout=dropout, use_last_k=config.get('use_last_k', 2))
        self.mol_encoder = MolecularEncoder(
            dims.get('fp_dim', 0), dims.get('bert_dim', 0), d_model, dropout=dropout)
        self.interaction = CrossResidueInteraction(d_model, nhead, dropout)

    def forward(self, batch):
        prot_residues = self.protein_encoder(batch['prot_feats'])
        mol_kwargs = {}
        if 'mol_fp' in batch:
            mol_kwargs['fingerprints'] = batch['mol_fp']
        if 'mol_bert_embed' in batch:
            mol_kwargs['bert_embed'] = batch['mol_bert_embed']
        mol_vec = self.mol_encoder(**mol_kwargs)
        return self.interaction(prot_residues, batch['prot_feat_mask'], mol_vec)


def resolve_dims(dataset_dict):
    sample = torch.load(dataset_dict['prot_feat_paths'][0], map_location='cpu',
                        weights_only=False)
    keys = sorted(sample.keys())
    per_layer = sample[keys[0]].shape[-1]
    fp = dataset_dict.get('mol_fingerprints')
    bert = dataset_dict.get('mol_bert_embeddings')
    return {
        'num_feat_layers': len(keys),
        'per_layer_dim': per_layer,
        'fp_dim': fp[0].shape[-1] if fp else 0,
        'bert_dim': bert[0].shape[-1] if bert else 0,
    }


def build_model(config, dims):
    return AffinityPredictor(config, dims)


# -------------------------------------------------------------------- losses
def rank_loss(preds, labels, margin=0.1):
    """Ranking loss over label-distinguishable pairs."""
    B = preds.size(0)
    if B < 2:
        return torch.zeros((), device=preds.device)
    idx_i, idx_j = torch.triu_indices(B, B, offset=1, device=preds.device)
    if idx_i.numel() == 0:
        return torch.zeros((), device=preds.device)
    pred_i, pred_j = preds[idx_i], preds[idx_j]
    label_i, label_j = labels[idx_i], labels[idx_j]
    target = torch.sign(label_i - label_j)
    valid = target != 0
    if not valid.any():
        return torch.zeros((), device=preds.device)
    return F.margin_ranking_loss(pred_i[valid], pred_j[valid], target[valid], margin=margin)


def within_molecule_var_loss(preds, smiles):

    groups = {}
    for i, sm in enumerate(smiles):
        groups.setdefault(sm, []).append(i)
    losses = []
    for idxs in groups.values():
        if len(idxs) > 1:
            losses.append(-torch.var(preds[idxs]))
    if not losses:
        return torch.zeros((), device=preds.device)
    return torch.stack(losses).mean()


# ------------------------------------------------------------------- metrics
def concordance_index(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    n = len(y_true)
    if n < 2:
        return 1.0
    total = correct = 0
    for i in range(n):
        for j in range(i + 1, n):
            if y_true[i] == y_true[j]:
                continue
            total += 1
            if (y_true[i] < y_true[j]) == (y_pred[i] < y_pred[j]):
                correct += 1
            elif y_pred[i] == y_pred[j]:
                correct += 0.5
    return correct / total if total else 1.0


def compute_all_metrics(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true).flatten(), np.asarray(y_pred).flatten()
    mse = mean_squared_error(y_true, y_pred)
    pcc, _ = pearsonr(y_true, y_pred)
    scc, _ = spearmanr(y_true, y_pred)
    return {
        'MSE': mse, 'RMSE': np.sqrt(mse), 'MAE': mean_absolute_error(y_true, y_pred),
        'R2': r2_score(y_true, y_pred), 'PearsonR': pcc, 'SpearmanR': scc,
        'C-index': concordance_index(y_true, y_pred),
    }


# ------------------------------------------------------------------ training
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    def __init__(self, patience=20, verbose=False, delta=0):
        self.patience, self.verbose, self.delta = patience, verbose, delta
        self.counter, self.best_loss, self.early_stop = 0, None, False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0


def move_batch_to_device(batch, device):
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            batch[k] = v.to(device)
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            batch[k] = [t.to(device) for t in v]
    return batch


def train_epoch(model, dataloader, optimizer, device, rank_weight=1.0, var_weight=1.0,
                clip_norm=1.0):
    model.train()
    total_loss, total_samples = 0.0, 0
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad()
        labels = batch['labels']
        preds = model(batch)
        loss = (F.mse_loss(preds, labels) + rank_weight * rank_loss(preds, labels)
                + var_weight * within_molecule_var_loss(preds, batch['smiles']))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        total_samples += labels.size(0)
    return total_loss / max(total_samples, 1)


def validate(model, dataloader, device):
    model.eval()
    total_loss, all_preds, all_labels = 0.0, [], []
    with torch.no_grad():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            preds = model(batch)
            total_loss += F.mse_loss(preds, batch['labels']).item() * batch['labels'].size(0)
            all_preds.append(preds.cpu().numpy())
            all_labels.append(batch['labels'].cpu().numpy())
    all_preds = np.concatenate(all_preds)
    all_labels = np.concatenate(all_labels)
    return total_loss / max(len(all_labels), 1), all_preds, all_labels
