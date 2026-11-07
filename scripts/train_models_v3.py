"""
train_models_v3.py -- Train and evaluate BDH, LSTM, and TFT on original and augmented datasets.

Models:
  - BDH:  Baby Dragon Hatchling (multi-scale 1D conv + dilated gated residual + attention pooling)
  - LSTM: Standard 2-layer LSTM baseline
  - TFT:  Lightweight Temporal Fusion Transformer for binary classification

Two experiments:
  1. Original: 502 train sequences (11 positive)
  2. Augmented: 541 train sequences (50 positive)

Both evaluated on the same held-out test set (216 sequences, 3 positive).
Validation: last 15% of original train (seq 428-501) for checkpoint selection.
Test set completely unseen during training.

Saves:
  models/bdh_v3_orig.pt, models/bdh_v3_aug.pt
  models/lstm_v3_orig.pt, models/lstm_v3_aug.pt
  models/tft_v3_orig.pt, models/tft_v3_aug.pt
  models/preprocessing_params_v3.npz
  models/model_comparison_v3.json
  documentation/training_curves_v3_orig.png
  documentation/training_curves_v3_aug.png
  documentation/pr_roc_curves_v3_orig.png
  documentation/pr_roc_curves_v3_aug.png
  documentation/confusion_matrices_v3_orig.png
  documentation/confusion_matrices_v3_aug.png
  documentation/model_comparison_bar_v3.png
"""

import sys, os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ["MPLCONFIGDIR"] = os.path.join(os.environ.get("TEMP", "C:/Temp"), "matplotlib_flare")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

from pathlib import Path
import json, random, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    accuracy_score, precision_recall_curve, roc_curve
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -- Reproducibility -----------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# -- Paths ---------------------------------------------------------------------
ROOT      = Path(__file__).resolve().parent.parent
SEQ_DIR   = ROOT / "Sequences"
MODELS    = ROOT / "models"
DOCS      = ROOT / "documentation"
MODELS.mkdir(exist_ok=True)
DOCS.mkdir(exist_ok=True)

# -- Hyperparameters -----------------------------------------------------------
SPLIT_IDX       = 502           # chronological train/test boundary
VAL_START       = 428           # last 15% of original train = validation set
BATCH_SIZE      = 16
EPOCHS          = 100
LR              = 1e-3
POS_WEIGHT_ORIG = 44.6          # ~491 neg / 11 pos
POS_WEIGHT_AUG  = 9.8           # ~491 neg / 50 pos
HIDDEN_DIM      = 64
DROPOUT         = 0.30


# ==============================================================================
# DATA LOADING & PREPROCESSING
# ==============================================================================

def compute_norm_params(X_tr):
    """Compute z-score parameters on training data only."""
    flat = X_tr.reshape(-1, X_tr.shape[-1])
    mu  = flat.mean(axis=0)
    sig = flat.std(axis=0)
    sig = np.where(sig < 1e-6, 1.0, sig)
    return mu, sig


def apply_norm(X, mu, sig):
    return np.clip((X - mu) / sig, -5.0, 5.0).astype(np.float32)


def load_original_data():
    """
    Load the original (non-augmented) dataset.
    Train: seq 0-427 (original train minus validation)
    Val:   seq 428-501 (validation -- for checkpoint selection)
    Test:  seq 502-717
    """
    X_raw = np.load(SEQ_DIR / "suit2024_X_features_v2.npy")
    y_raw = np.load(SEQ_DIR / "suit2024_y_labels_v2.npy")

    # Drop constant features (mirror of v2 pipeline)
    X_tr_raw = X_raw[:SPLIT_IDX]
    per_std  = X_tr_raw.reshape(-1, X_raw.shape[-1]).std(axis=0)
    keep     = per_std > 1e-3
    dropped  = np.where(~keep)[0].tolist()
    keep_mask = keep
    X_raw    = X_raw[:, :, keep]

    # Splits
    X_train_all = X_raw[:SPLIT_IDX]
    y_train_all = y_raw[:SPLIT_IDX]
    X_test      = X_raw[SPLIT_IDX:]
    y_test      = y_raw[SPLIT_IDX:]

    # Norm using TRAIN (0..SPLIT_IDX) stats
    mu, sig = compute_norm_params(X_train_all)
    X_train_norm = apply_norm(X_train_all, mu, sig)
    X_test_norm  = apply_norm(X_test,      mu, sig)

    # Train / Val split
    X_tr = X_train_norm[:VAL_START]
    y_tr = y_train_all[:VAL_START]
    X_va = X_train_norm[VAL_START:]
    y_va = y_train_all[VAL_START:]

    nf = X_tr.shape[-1]
    print(f"[Original] Train={len(X_tr)} ({int(y_tr.sum())}+)  "
          f"Val={len(X_va)} ({int(y_va.sum())}+)  "
          f"Test={len(X_test)} ({int(y_test.sum())}+)  "
          f"Features={nf}  dropped={dropped}", flush=True)

    # Save preprocessing params
    np.savez(MODELS / "preprocessing_params_v3.npz",
             mu=mu, sig=sig, keep_mask=keep_mask.astype(np.uint8))

    return X_tr, y_tr, X_va, y_va, X_test_norm, y_test, nf, mu, sig


def load_augmented_data(mu, sig):
    """
    Load augmented training data (541 seqs = 502 orig + 39 synthetic).
    Validation and test remain from the ORIGINAL dataset.
    Normalization uses the SAME mu/sig as original experiment (computed from orig train).
    """
    X_aug_raw = np.load(SEQ_DIR / "suit2024_X_aug_v3.npy")   # (541, 80, 17) already dropped
    y_aug_raw = np.load(SEQ_DIR / "suit2024_y_aug_v3.npy")   # (541,)

    # Original test (same as before)
    X_raw = np.load(SEQ_DIR / "suit2024_X_features_v2.npy")
    y_raw = np.load(SEQ_DIR / "suit2024_y_labels_v2.npy")
    per_std = X_raw[:SPLIT_IDX].reshape(-1, X_raw.shape[-1]).std(axis=0)
    keep    = per_std > 1e-3
    X_test  = X_raw[SPLIT_IDX:][:, :, keep]
    y_test  = y_raw[SPLIT_IDX:]

    # Normalize augmented train with SAME params
    X_aug_norm  = apply_norm(X_aug_raw, mu, sig)
    X_test_norm = apply_norm(X_test,    mu, sig)

    # Val: last 74 of original train (seq 428-501), re-normalized
    X_orig_train = apply_norm(X_aug_raw[:SPLIT_IDX], mu, sig)
    X_va = X_orig_train[VAL_START:]
    y_va = y_aug_raw[VAL_START:SPLIT_IDX]  # original labels (no synth in val)

    # Augmented train: all 541 normalized seqs
    X_tr = X_aug_norm
    y_tr = y_aug_raw

    nf = X_tr.shape[-1]
    print(f"[Augmented] Train={len(X_tr)} ({int(y_tr.sum())}+)  "
          f"Val={len(X_va)} ({int(y_va.sum())}+)  "
          f"Test={len(X_test_norm)} ({int(y_test.sum())}+)  "
          f"Features={nf}", flush=True)

    return X_tr, y_tr, X_va, y_va, X_test_norm, y_test, nf


class SolarDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ==============================================================================
# MODEL ARCHITECTURES
# ==============================================================================

# -- BDH (unchanged from v2) ---------------------------------------------------
class MultiScaleTemporalConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        third       = out_ch // 3
        rem         = out_ch - 2 * third
        self.conv3  = nn.Conv1d(in_ch, third, 3,  padding=1)
        self.conv7  = nn.Conv1d(in_ch, third, 7,  padding=3)
        self.conv15 = nn.Conv1d(in_ch, rem,   15, padding=7)
        self.norm   = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act    = nn.GELU()
    def forward(self, x):
        return self.act(self.norm(
            torch.cat([self.conv3(x), self.conv7(x), self.conv15(x)], dim=1)))


class GatedResidualBlock(nn.Module):
    def __init__(self, ch, dilation=1, dropout=0.30):
        super().__init__()
        self.conv_f = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.conv_g = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.norm   = nn.GroupNorm(min(8, ch), ch)
        self.drop   = nn.Dropout(dropout)
        r           = max(ch // 4, 4)
        self.se1    = nn.Linear(ch, r)
        self.se2    = nn.Linear(r, ch)
    def forward(self, x):
        h  = torch.tanh(self.conv_f(x)) * torch.sigmoid(self.conv_g(x))
        h  = self.drop(self.norm(h))
        se = torch.sigmoid(self.se2(F.relu(self.se1(h.mean(-1))))).unsqueeze(-1)
        return x + h * se


class BabyDragonHatchling(nn.Module):
    """BDH: multi-scale 1D conv -> dilated gated residual blocks -> temporal attention pooling."""
    def __init__(self, in_features, hidden_dim=64, num_blocks=3, dropout=0.30):
        super().__init__()
        self.in_proj    = nn.Sequential(nn.Linear(in_features, hidden_dim),
                                        nn.LayerNorm(hidden_dim), nn.GELU())
        self.multiscale = MultiScaleTemporalConv(hidden_dim, hidden_dim)
        self.blocks     = nn.ModuleList([
            GatedResidualBlock(hidden_dim, dilation=2**i, dropout=dropout)
            for i in range(num_blocks)])
        self.attn_q     = nn.Linear(hidden_dim, 1, bias=False)
        self.attn_n     = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 32), nn.GELU(),
                                        nn.Dropout(dropout), nn.Linear(32, 1))
    def forward(self, x):
        h = self.in_proj(x).transpose(1, 2)      # (B, D, T)
        h = self.multiscale(h)
        for blk in self.blocks:
            h = blk(h)
        h      = h.transpose(1, 2)               # (B, T, D)
        attn   = torch.softmax(self.attn_q(self.attn_n(h)), dim=1)
        pooled = (h * attn).sum(dim=1)
        return self.classifier(pooled)


# -- LSTM Baseline -------------------------------------------------------------
class LSTMFlarePredictor(nn.Module):
    """Standard 2-layer LSTM baseline."""
    def __init__(self, in_features, hidden_dim=64, num_layers=2, dropout=0.25):
        super().__init__()
        self.in_proj    = nn.Sequential(nn.Linear(in_features, hidden_dim),
                                        nn.LayerNorm(hidden_dim), nn.GELU())
        self.lstm       = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                                  dropout=dropout if num_layers > 1 else 0.0,
                                  batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 32), nn.GELU(),
                                        nn.Dropout(dropout), nn.Linear(32, 1))
    def forward(self, x):
        out, _ = self.lstm(self.in_proj(x))
        return self.classifier(out[:, -1, :])


# -- TFT (Lightweight Temporal Fusion Transformer) ----------------------------
class GatedResidualNetworkTFT(nn.Module):
    """
    Gated Residual Network from TFT paper.
    GLU gating with optional context vector, residual connection, layer norm.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1, context_dim=None):
        super().__init__()
        self.fc1       = nn.Linear(input_dim, hidden_dim)
        self.fc2       = nn.Linear(hidden_dim, output_dim * 2)  # for GLU
        self.elu       = nn.ELU()
        self.drop      = nn.Dropout(dropout)
        self.norm      = nn.LayerNorm(output_dim)
        self.skip      = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        if context_dim is not None:
            self.ctx_fc = nn.Linear(context_dim, hidden_dim, bias=False)
        else:
            self.ctx_fc = None

    def forward(self, x, context=None):
        h = self.elu(self.fc1(x))
        if context is not None and self.ctx_fc is not None:
            h = h + self.ctx_fc(context)
        h      = self.drop(self.fc2(h))
        h1, h2 = h.chunk(2, dim=-1)
        gated  = h1 * torch.sigmoid(h2)   # GLU
        return self.norm(gated + self.skip(x))


class VariableSelectionNetwork(nn.Module):
    """
    VSN: learns per-feature importance weights at each timestep.
    Input: (B, T, F) -> Output: (B, T, hidden_dim)
    """
    def __init__(self, num_features, hidden_dim, dropout=0.1):
        super().__init__()
        self.grn_per_feat = nn.ModuleList([
            GatedResidualNetworkTFT(1, hidden_dim, hidden_dim, dropout)
            for _ in range(num_features)
        ])
        self.grn_weights  = GatedResidualNetworkTFT(num_features, hidden_dim, num_features, dropout)

    def forward(self, x):
        B, T, F = x.shape
        feat_embeds = []
        for i, grn in enumerate(self.grn_per_feat):
            fi = x[:, :, i:i+1]                                # (B, T, 1)
            feat_embeds.append(grn(fi.reshape(B*T, 1)).reshape(B, T, -1))
        feat_stack = torch.stack(feat_embeds, dim=-2)           # (B, T, F, D)

        flat_x  = x.reshape(B*T, F)
        weights = torch.softmax(self.grn_weights(flat_x), dim=-1)  # (B*T, F)
        weights = weights.reshape(B, T, F, 1)

        out = (feat_stack * weights).sum(dim=-2)                # (B, T, D)
        return out


class TemporalFusionTransformer(nn.Module):
    """
    Lightweight TFT for binary time-series classification.
    Architecture: VSN -> LSTM encoder -> Multi-head attention -> GRN -> Attention Pooling -> MLP
    """
    def __init__(self, num_features, hidden_dim=64, lstm_layers=2,
                 num_heads=4, dropout=0.25):
        super().__init__()
        self.hidden_dim = hidden_dim

        # 1. Variable Selection Network
        self.vsn = VariableSelectionNetwork(num_features, hidden_dim, dropout)

        # 2. LSTM encoder
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, lstm_layers,
                            batch_first=True,
                            dropout=dropout if lstm_layers > 1 else 0.0)
        self.lstm_norm = nn.LayerNorm(hidden_dim)

        # 3. Multi-head self-attention
        self.attn     = nn.MultiheadAttention(hidden_dim, num_heads,
                                              dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(hidden_dim)

        # 4. Post-attention GRN
        self.post_grn  = GatedResidualNetworkTFT(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.post_norm = nn.LayerNorm(hidden_dim)

        # 5. Classifier with learned temporal pooling
        self.pool_q    = nn.Linear(hidden_dim, 1, bias=False)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, x):
        h = self.vsn(x)                                    # (B, T, D)
        lstm_out, _ = self.lstm(h)                         # (B, T, D)
        lstm_out    = self.lstm_norm(lstm_out + h)         # residual

        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)  # (B, T, D)
        attn_out    = self.attn_norm(attn_out + lstm_out)      # residual

        B, T, D     = attn_out.shape
        flat        = attn_out.reshape(B*T, D)
        post        = self.post_grn(flat).reshape(B, T, D)
        post        = self.post_norm(post + attn_out)

        weights = torch.softmax(self.pool_q(post), dim=1)  # (B, T, 1)
        pooled  = (post * weights).sum(dim=1)              # (B, D)
        return self.classifier(pooled)


# ==============================================================================
# TRAINING & EVALUATION
# ==============================================================================

@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss, logits_all, targets_all = 0.0, [], []
    for bx, by in loader:
        lg = model(bx)
        total_loss += loss_fn(lg, by).item() * len(by)
        logits_all.append(lg.cpu().numpy())
        targets_all.append(by.cpu().numpy())
    logits  = np.concatenate(logits_all).flatten()
    targets = np.concatenate(targets_all).flatten()
    probs   = torch.sigmoid(torch.tensor(logits)).numpy()
    return total_loss / len(loader.dataset), probs, targets


def train_model(model, tr_ld, va_ld, name, pos_weight, epochs=EPOCHS):
    """Train with BCE+pos_weight, CosineAnnealing. Best checkpoint by val PR-AUC."""
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)

    history = {"train_loss": [], "val_loss": [], "val_pr_auc": [], "val_recall": []}
    best_auc, best_state = -1.0, None
    n_params = sum(p.numel() for p in model.parameters())
    t0 = time.time()

    print(f"\n{'-'*62}", flush=True)
    print(f"  {name}  |  params={n_params:,}  |  {epochs} epochs  |  pos_w={pos_weight:.1f}", flush=True)
    print(f"{'-'*62}", flush=True)

    for ep in range(1, epochs + 1):
        model.train(); tl = 0.0
        for bx, by in tr_ld:
            opt.zero_grad()
            loss = loss_fn(model(bx), by)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tl += loss.item() * len(by)
        sch.step()

        tr_loss              = tl / len(tr_ld.dataset)
        va_loss, probs, tgts = evaluate(model, va_ld, loss_fn)
        pr_auc               = average_precision_score(tgts, probs)
        recall               = recall_score(tgts, (probs >= 0.5).astype(int), zero_division=0)

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["val_pr_auc"].append(pr_auc)
        history["val_recall"].append(recall)

        if pr_auc >= best_auc:
            best_auc   = pr_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % 20 == 0 or ep == epochs:
            print(f"  Ep {ep:3d}/{epochs} | TrL={tr_loss:.4f} | VaL={va_loss:.4f} "
                  f"| PR-AUC={pr_auc:.4f} | Recall={recall:.4f}", flush=True)

    model.load_state_dict(best_state)
    elapsed = time.time() - t0
    print(f"  Best val PR-AUC: {best_auc:.4f}  |  Time: {elapsed:.1f}s", flush=True)
    return model, history, best_auc, elapsed


def compute_metrics(tgt, prob, name):
    """Full metric suite including threshold sweep."""
    roc_auc = roc_auc_score(tgt, prob)
    pr_auc  = average_precision_score(tgt, prob)

    p_, r_, t_ = precision_recall_curve(tgt, prob)
    f1_arr     = np.where((p_ + r_) > 0, 2*p_*r_/(p_+r_+1e-10), 0)
    best_i     = int(np.argmax(f1_arr))
    opt_thresh = float(t_[best_i]) if best_i < len(t_) else 0.5

    def _metrics_at(th):
        pred          = (prob >= th).astype(int)
        cm            = confusion_matrix(tgt, pred)
        tn, fp, fn, tp = [int(v) for v in (
            cm.ravel() if cm.size == 4 else [cm[0,0], 0, 0, 0])]
        return {
            "threshold": round(th, 4),
            "accuracy":  round(float(accuracy_score(tgt, pred)), 4),
            "precision": round(float(precision_score(tgt, pred, zero_division=0)), 4),
            "recall":    round(float(recall_score(tgt, pred, zero_division=0)), 4),
            "f1":        round(float(f1_score(tgt, pred, zero_division=0)), 4),
            "confusion_matrix": cm.tolist(),
            "tn": tn, "fp": fp, "fn": fn, "tp": tp,
        }

    result = {
        "model_name":              name,
        "roc_auc":                 round(float(roc_auc), 4),
        "pr_auc":                  round(float(pr_auc),  4),
        "default_threshold_0.5":   _metrics_at(0.5),
        "optimal_f1_threshold":    _metrics_at(opt_thresh),
    }
    fpr, tpr, _ = roc_curve(tgt, prob)
    return result, (p_, r_), (fpr, tpr), prob


# ==============================================================================
# PLOT GENERATION
# ==============================================================================

def plot_training_curves(histories, title_suffix, save_path):
    """Plot train/val loss and val PR-AUC for all models in one experiment."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    colors = {"BDH": "#2563eb", "LSTM": "#ea580c", "TFT": "#16a34a"}
    styles_tr = {"BDH": "-",    "LSTM": "-",        "TFT": "-"}
    styles_va = {"BDH": "--",   "LSTM": "--",       "TFT": "--"}

    for model_name, hist in histories.items():
        c = colors.get(model_name, "gray")
        axes[0].plot(hist["train_loss"], color=c, ls=styles_tr[model_name],
                     lw=2, label=f"{model_name} Train")
        axes[0].plot(hist["val_loss"],   color=c, ls=styles_va[model_name],
                     lw=2, alpha=0.7, label=f"{model_name} Val")
        axes[1].plot(hist["val_pr_auc"], color=c, lw=2.5,
                     label=f"{model_name} Val PR-AUC")

    axes[0].set(xlabel="Epoch", ylabel="Weighted BCE Loss",
                title=f"Train / Val Loss -- {title_suffix}")
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="Epoch", ylabel="PR-AUC",
                title=f"Val PR-AUC Convergence -- {title_suffix}")
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}", flush=True)


def plot_pr_roc(results_dict, y_test, title_suffix, save_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    colors = {"BDH": "#2563eb", "LSTM": "#ea580c", "TFT": "#16a34a"}
    no_skill = y_test.sum() / len(y_test)

    for model_name, (metrics, pr_data, roc_data, _probs) in results_dict.items():
        c = colors.get(model_name, "gray")
        axes[0].plot(pr_data[1], pr_data[0], color=c, lw=2.5,
                     label=f"{model_name}  PR-AUC={metrics['pr_auc']:.4f}")
        axes[1].plot(roc_data[0], roc_data[1], color=c, lw=2.5,
                     label=f"{model_name}  ROC-AUC={metrics['roc_auc']:.4f}")

    axes[0].axhline(no_skill, color="gray", ls=":", label=f"No-skill ({no_skill:.3f})")
    axes[0].set(xlabel="Recall", ylabel="Precision",
                title=f"Precision-Recall Curve -- {title_suffix}",
                xlim=[0, 1.05], ylim=[0, 1.05])
    axes[0].legend(loc="upper right")

    axes[1].plot([0, 1], [0, 1], "gray", ls=":")
    axes[1].set(xlabel="FPR", ylabel="TPR",
                title=f"ROC Curve -- {title_suffix}",
                xlim=[0, 1.05], ylim=[0, 1.05])
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}", flush=True)


def plot_confusion_matrices(results_dict, title_suffix, save_path):
    n = len(results_dict)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 5), dpi=150)
    if n == 1:
        axes = [axes]

    for ax, (model_name, (metrics, *_)) in zip(axes, results_dict.items()):
        ot   = metrics["optimal_f1_threshold"]
        cm   = np.array(ot["confusion_matrix"])
        im   = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(["Neg", "Pos"]); ax.set_yticklabels(["Neg", "Pos"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title(f"{model_name}\nPR-AUC={metrics['pr_auc']:.4f}  "
                     f"Recall={ot['recall']:.2f}  Prec={ot['precision']:.2f}\n"
                     f"({title_suffix})")
        for r in range(2):
            for c_ in range(2):
                ax.text(c_, r, str(cm[r, c_]), ha="center", va="center",
                        color="white" if cm[r, c_] > cm.max()/2 else "black",
                        fontsize=14, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}", flush=True)


def plot_model_comparison_bar(orig_results, aug_results, save_path):
    """Bar chart comparing PR-AUC and ROC-AUC across all models x experiments."""
    models     = ["BDH", "LSTM", "TFT"]
    metrics    = ["pr_auc", "roc_auc"]
    metric_lbl = ["PR-AUC", "ROC-AUC"]
    x          = np.arange(len(models))
    width      = 0.18

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    colors_orig = "#2563eb"
    colors_aug  = "#16a34a"

    for ax_i, (metric, mlbl) in enumerate(zip(metrics, metric_lbl)):
        orig_vals = [orig_results.get(m, ({},))[0].get(metric, 0) for m in models]
        aug_vals  = [aug_results.get(m,  ({},))[0].get(metric, 0) for m in models]

        b1 = axes[ax_i].bar(x - width/2, orig_vals, width, label="Original",
                            color=colors_orig, alpha=0.85)
        b2 = axes[ax_i].bar(x + width/2, aug_vals,  width, label="Augmented",
                            color=colors_aug, alpha=0.85)

        axes[ax_i].set_xticks(x); axes[ax_i].set_xticklabels(models, fontsize=12)
        axes[ax_i].set_ylabel(mlbl); axes[ax_i].set_title(f"{mlbl} Comparison")
        axes[ax_i].legend()

        for bar in list(b1) + list(b2):
            h = bar.get_height()
            axes[ax_i].text(bar.get_x() + bar.get_width()/2, h + 0.005,
                            f"{h:.4f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("BDH vs LSTM vs TFT -- Original vs Augmented Training", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {save_path}", flush=True)


# ==============================================================================
# MAIN
# ==============================================================================

def run_experiment(label, X_tr, y_tr, X_va, y_va, X_te, y_te, nf, pos_weight,
                   model_suffix):
    """Run BDH + LSTM + TFT for one experiment (orig or aug)."""
    tr_ds = SolarDataset(X_tr, y_tr)
    va_ds = SolarDataset(X_va, y_va)
    te_ds = SolarDataset(X_te, y_te)
    tr_ld = DataLoader(tr_ds, BATCH_SIZE, shuffle=True,  drop_last=False)
    va_ld = DataLoader(va_ds, BATCH_SIZE, shuffle=False)
    te_ld = DataLoader(te_ds, BATCH_SIZE, shuffle=False)

    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], dtype=torch.float32))

    model_configs = [
        ("BDH",  BabyDragonHatchling(nf, HIDDEN_DIM, num_blocks=3, dropout=DROPOUT)),
        ("LSTM", LSTMFlarePredictor(nf, HIDDEN_DIM, num_layers=2, dropout=DROPOUT)),
        ("TFT",  TemporalFusionTransformer(nf, HIDDEN_DIM, lstm_layers=2,
                                           num_heads=4, dropout=DROPOUT)),
    ]

    histories  = {}
    results    = {}

    for model_name, model in model_configs:
        torch.manual_seed(SEED)
        trained, hist, best_val_auc, elapsed = train_model(
            model, tr_ld, va_ld, f"{model_name} [{label}]", pos_weight)

        # Evaluate on TEST
        _, probs, tgts = evaluate(trained, te_ld, loss_fn)
        metrics, pr_data, roc_data, _ = compute_metrics(tgts, probs, model_name)
        metrics["training_time_s"] = round(elapsed, 1)
        metrics["n_params"]        = sum(p.numel() for p in trained.parameters())
        metrics["best_val_pr_auc"] = round(best_val_auc, 4)

        # Save checkpoint
        ckpt_path = MODELS / f"{model_name.lower()}_v3_{model_suffix}.pt"
        torch.save(trained.state_dict(), ckpt_path)
        print(f"  Saved checkpoint: {ckpt_path}", flush=True)

        histories[model_name] = hist
        results[model_name]   = (metrics, pr_data, roc_data, probs)

        ot = metrics["optimal_f1_threshold"]
        print(f"  TEST  PR-AUC={metrics['pr_auc']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}  "
              f"Recall={ot['recall']:.2f}  Prec={ot['precision']:.2f}  F1={ot['f1']:.4f}", flush=True)

    return histories, results, tgts


def main():
    print("=" * 70, flush=True)
    print("Solar Flare Forecasting v3 -- BDH vs LSTM vs TFT", flush=True)
    print(f"Seed={SEED}  Epochs={EPOCHS}  Batch={BATCH_SIZE}  Hidden={HIDDEN_DIM}", flush=True)
    print("=" * 70, flush=True)

    # -- Load data --
    print("\n[1] Loading and preprocessing data...", flush=True)
    X_tr_orig, y_tr_orig, X_va, y_va, X_te, y_te, nf, mu, sig = load_original_data()
    X_tr_aug,  y_tr_aug,  X_va2, y_va2, X_te2, y_te2, nf2    = load_augmented_data(mu, sig)

    # -- Experiment 1: Original data --
    print("\n" + "="*70, flush=True)
    print("EXPERIMENT 1 -- Original Training Data (11 positives)", flush=True)
    print("="*70, flush=True)
    hist_orig, res_orig, y_te_orig = run_experiment(
        "ORIG", X_tr_orig, y_tr_orig, X_va, y_va, X_te, y_te, nf,
        pos_weight=POS_WEIGHT_ORIG, model_suffix="orig"
    )

    # -- Experiment 2: Augmented data --
    print("\n" + "="*70, flush=True)
    print("EXPERIMENT 2 -- Augmented Training Data (~50 positives)", flush=True)
    print("="*70, flush=True)
    hist_aug, res_aug, y_te_aug = run_experiment(
        "AUG", X_tr_aug, y_tr_aug, X_va2, y_va2, X_te2, y_te2, nf2,
        pos_weight=POS_WEIGHT_AUG, model_suffix="aug"
    )

    # -- Save plots --
    print("\n[3] Generating plots...", flush=True)
    plot_training_curves(hist_orig, "Original", DOCS / "training_curves_v3_orig.png")
    plot_training_curves(hist_aug,  "Augmented", DOCS / "training_curves_v3_aug.png")

    plot_pr_roc(res_orig, y_te_orig, "Original", DOCS / "pr_roc_curves_v3_orig.png")
    plot_pr_roc(res_aug,  y_te_aug,  "Augmented", DOCS / "pr_roc_curves_v3_aug.png")

    plot_confusion_matrices(res_orig, "Original",  DOCS / "confusion_matrices_v3_orig.png")
    plot_confusion_matrices(res_aug,  "Augmented", DOCS / "confusion_matrices_v3_aug.png")

    plot_model_comparison_bar(
        {m: res_orig[m] for m in ["BDH","LSTM","TFT"] if m in res_orig},
        {m: res_aug[m]  for m in ["BDH","LSTM","TFT"] if m in res_aug},
        DOCS / "model_comparison_bar_v3.png"
    )

    # -- Build comparison JSON --
    print("\n[4] Saving comparison JSON...", flush=True)
    comparison = {
        "task":     "Aditya-L1 SUIT Solar Flare Prediction (~90s Forecast Window)",
        "version":  "v3",
        "seed":     SEED,
        "n_features": int(nf),
        "val_start":  VAL_START,
        "split_idx":  SPLIT_IDX,
        "chronological_split_cutoff_utc": "2024-02-22 21:21:53",
        "augmentation": {
            "sigma_scale":    0.05,
            "n_synth":        39,
            "n_flare_events": 7,
            "n_train_pos_aug": 50,
        },
        "original": {
            "train_samples":    int(len(X_tr_orig)),
            "train_positives":  int(y_tr_orig.sum()),
            "val_samples":      int(len(X_va)),
            "val_positives":    int(y_va.sum()),
            "test_samples":     int(len(X_te)),
            "test_positives":   int(y_te.sum()),
            "pos_weight":       POS_WEIGHT_ORIG,
            "metrics":          {m: res_orig[m][0] for m in res_orig},
        },
        "augmented": {
            "train_samples":    int(len(X_tr_aug)),
            "train_positives":  int(y_tr_aug.sum()),
            "val_samples":      int(len(X_va2)),
            "val_positives":    int(y_va2.sum()),
            "test_samples":     int(len(X_te2)),
            "test_positives":   int(y_te2.sum()),
            "pos_weight":       POS_WEIGHT_AUG,
            "metrics":          {m: res_aug[m][0] for m in res_aug},
        },
    }
    with open(MODELS / "model_comparison_v3.json", "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"  Saved: {MODELS / 'model_comparison_v3.json'}", flush=True)

    # -- Print final summary --
    print("\n" + "="*70, flush=True)
    print("FINAL RESULTS SUMMARY", flush=True)
    print("="*70, flush=True)
    for exp_label, res in [("Original", res_orig), ("Augmented", res_aug)]:
        print(f"\n-- {exp_label} --", flush=True)
        print(f"{'Model':<8} {'PR-AUC':>8} {'ROC-AUC':>9} {'F1':>7} {'Recall':>8} {'Prec':>7} {'FP':>5} {'FN':>5}", flush=True)
        for mname in ["BDH", "LSTM", "TFT"]:
            if mname not in res:
                continue
            m  = res[mname][0]
            ot = m["optimal_f1_threshold"]
            print(f"{mname:<8} {m['pr_auc']:>8.4f} {m['roc_auc']:>9.4f} "
                  f"{ot['f1']:>7.4f} {ot['recall']:>8.4f} {ot['precision']:>7.4f} "
                  f"{ot['fp']:>5} {ot['fn']:>5}", flush=True)

    print("\nAll outputs saved to models/ and documentation/", flush=True)


if __name__ == "__main__":
    main()
