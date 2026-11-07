"""
scripts/rebuild_step4_train_models.py
Train and evaluate BDH, LSTM, and TFT on the authentic SUIT 2024 v4 dataset.
Event-aware, strictly chronological Train/Val/Test partitions.
No synthetic augmentation. Authentic observational data only.
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

ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
SEQ_DIR   = ROOT / "Sequences"
MODELS    = ROOT / "models"
DOCS      = ROOT / "documentation"
MODELS.mkdir(exist_ok=True)
DOCS.mkdir(exist_ok=True)

BATCH_SIZE  = 32
EPOCHS      = 80
LR          = 1e-3
HIDDEN_DIM  = 64
DROPOUT     = 0.30

# ==============================================================================
# DATA LOADING & PREPROCESSING
# ==============================================================================

def compute_norm_params(X_tr):
    flat = X_tr.reshape(-1, X_tr.shape[-1])
    mu  = flat.mean(axis=0)
    sig = flat.std(axis=0)
    sig = np.where(sig < 1e-6, 1.0, sig)
    return mu, sig

def apply_norm(X, mu, sig):
    return np.clip((X - mu) / sig, -5.0, 5.0).astype(np.float32)

def load_data():
    X_raw = np.load(SEQ_DIR / "suit2024_X_features_v4.npy")
    y_raw = np.load(SEQ_DIR / "suit2024_y_labels_v4.npy")
    with open(DATA_DIR / "dataset_splits_v4.json", "r", encoding="utf-8") as f:
        splits = json.load(f)
        
    tr_idx = splits['train_indices']
    va_idx = splits['val_indices']
    te_idx = splits['test_indices']
    
    # Drop constant features based on Train only
    X_tr_raw = X_raw[tr_idx]
    per_std  = X_tr_raw.reshape(-1, X_raw.shape[-1]).std(axis=0)
    keep     = per_std > 1e-3
    dropped  = np.where(~keep)[0].tolist()
    
    X_kept = X_raw[:, :, keep]
    
    X_train_raw = X_kept[tr_idx]
    y_train     = y_raw[tr_idx]
    X_val_raw   = X_kept[va_idx]
    y_val       = y_raw[va_idx]
    X_test_raw  = X_kept[te_idx]
    y_test      = y_raw[te_idx]
    
    # Compute z-score params on TRAIN only
    mu, sig = compute_norm_params(X_train_raw)
    X_train = apply_norm(X_train_raw, mu, sig)
    X_val   = apply_norm(X_val_raw,   mu, sig)
    X_test  = apply_norm(X_test_raw,  mu, sig)
    
    np.savez(
        MODELS / "preprocessing_params_v4.npz",
        mu=mu,
        sig=sig,
        keep_mask=keep,
        dropped_indices=dropped
    )
    
    nf = X_train.shape[-1]
    print(f"[Dataset v4] Train={len(X_train)} ({int(y_train.sum())}+)  "
          f"Val={len(X_val)} ({int(y_val.sum())}+)  "
          f"Test={len(X_test)} ({int(y_test.sum())}+)  "
          f"Active Features={nf}  dropped={dropped}", flush=True)
          
    return X_train, y_train, X_val, y_val, X_test, y_test, nf

class SolarDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

# ==============================================================================
# MODEL ARCHITECTURES
# ==============================================================================

# -- Enhanced BDH with Causal 1D Convolutions & Temporal Positional Bias --------
class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size, dilation=dilation)
    def forward(self, x):
        return self.conv(F.pad(x, (self.pad, 0)))

class MultiScaleCausalConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        third = out_ch // 3
        rem   = out_ch - 2 * third
        self.conv3  = CausalConv1d(in_ch, third, 3)
        self.conv7  = CausalConv1d(in_ch, third, 7)
        self.conv15 = CausalConv1d(in_ch, rem,   15)
        self.norm   = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act    = nn.GELU()
    def forward(self, x):
        c3 = self.conv3(x)
        c7 = self.conv7(x)
        c15 = self.conv15(x)
        return self.act(self.norm(torch.cat([c3, c7, c15], dim=1)))

class GatedCausalResidualBlock(nn.Module):
    def __init__(self, ch, dilation=1, dropout=0.30):
        super().__init__()
        self.conv_f = CausalConv1d(ch, ch, 3, dilation=dilation)
        self.conv_g = CausalConv1d(ch, ch, 3, dilation=dilation)
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

class BabyDragonHatchlingV4(nn.Module):
    """
    BDH v4: Multi-scale Causal 1D Conv -> Dilated Gated Residual Blocks ->
    Temporal Positional-Biased Attention Pooling -> Classifier
    """
    def __init__(self, in_features, hidden_dim=64, num_blocks=3, dropout=0.30):
        super().__init__()
        self.in_proj    = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.multiscale = MultiScaleCausalConv(hidden_dim, hidden_dim)
        self.blocks     = nn.ModuleList([
            GatedCausalResidualBlock(hidden_dim, dilation=2**i, dropout=dropout)
            for i in range(num_blocks)
        ])
        self.attn_q     = nn.Linear(hidden_dim, 1, bias=False)
        self.attn_n     = nn.LayerNorm(hidden_dim)
        self.pos_bias   = nn.Parameter(torch.tensor(1.2)) # Learnable recency bias
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        # x: (B, T, F)
        h = self.in_proj(x).transpose(1, 2) # (B, D, T)
        h = self.multiscale(h)
        for blk in self.blocks:
            h = blk(h)
        h = h.transpose(1, 2) # (B, T, D)
        
        # Temporal Positional Attention Pooling
        T = h.size(1)
        pos = torch.linspace(0.0, 1.0, T, device=x.device).unsqueeze(0).unsqueeze(-1)
        attn_logits = self.attn_q(self.attn_n(h)) + self.pos_bias * pos
        attn = torch.softmax(attn_logits, dim=1)
        pooled = (h * attn).sum(dim=1)
        return self.classifier(pooled)

# -- Standard LSTM Baseline ----------------------------------------------------
class LSTMFlarePredictor(nn.Module):
    def __init__(self, in_features, hidden_dim=64, num_layers=2, dropout=0.25):
        super().__init__()
        self.in_proj    = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.lstm       = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(self.in_proj(x))
        return self.classifier(out[:, -1, :])

# -- TFT (Temporal Fusion Transformer) -----------------------------------------
class GatedResidualNetworkTFT(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1   = nn.Linear(input_dim, hidden_dim)
        self.fc2   = nn.Linear(hidden_dim, output_dim * 2)
        self.elu   = nn.ELU()
        self.drop  = nn.Dropout(dropout)
        self.norm  = nn.LayerNorm(output_dim)
        self.skip  = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
    def forward(self, x):
        h = self.elu(self.fc1(x))
        h = self.drop(self.fc2(h))
        h1, h2 = h.chunk(2, dim=-1)
        gated  = h1 * torch.sigmoid(h2)
        return self.norm(gated + self.skip(x))

class VariableSelectionNetwork(nn.Module):
    def __init__(self, num_features, hidden_dim, dropout=0.1):
        super().__init__()
        self.grn_per_feat = nn.ModuleList([
            GatedResidualNetworkTFT(1, hidden_dim, hidden_dim, dropout)
            for _ in range(num_features)
        ])
        self.grn_weights = GatedResidualNetworkTFT(num_features, hidden_dim, num_features, dropout)
    def forward(self, x):
        B, T, F = x.shape
        feat_embeds = []
        for i, grn in enumerate(self.grn_per_feat):
            feat_embeds.append(grn(x[:, :, i:i+1]))
        feat_stack = torch.stack(feat_embeds, dim=-2)
        weights = torch.softmax(self.grn_weights(x), dim=-1).unsqueeze(-1)
        return (feat_stack * weights).sum(dim=-2)

class TemporalFusionTransformer(nn.Module):
    def __init__(self, in_features, hidden_dim=64, num_heads=4, dropout=0.20):
        super().__init__()
        self.vsn       = VariableSelectionNetwork(in_features, hidden_dim, dropout)
        self.lstm      = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.post_lstm = GatedResidualNetworkTFT(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.attn      = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.post_attn = GatedResidualNetworkTFT(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.classifier= nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LayerNorm(32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        vsn_out = self.vsn(x)
        lstm_out, _ = self.lstm(vsn_out)
        lstm_gated = self.post_lstm(lstm_out)
        attn_out, _ = self.attn(lstm_gated, lstm_gated, lstm_gated)
        tft_rep = self.post_attn(attn_out)
        return self.classifier(tft_rep[:, -1, :])

# ==============================================================================
# TRAINING & EVALUATION ENGINE
# ==============================================================================

def train_and_eval(model_name, model, train_loader, val_loader, test_loader, y_val, y_test, pos_weight):
    device = torch.device("cpu")
    model = model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight]))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)
    
    best_val_prauc = -1.0
    best_weights   = None
    history        = {"train_loss": [], "val_loss": [], "val_prauc": []}
    
    print(f"\n--- Training {model_name} (Pos Weight: {pos_weight:.1f}) ---")
    t0 = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X_b, y_b in train_loader:
            optimizer.zero_grad()
            out = model(X_b)
            loss = criterion(out, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=3.0)
            optimizer.step()
            total_loss += loss.item() * len(y_b)
        train_loss = total_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss_tot = 0.0
        val_preds    = []
        with torch.no_grad():
            for X_b, y_b in val_loader:
                out = model(X_b)
                v_loss = criterion(out, y_b)
                val_loss_tot += v_loss.item() * len(y_b)
                val_preds.extend(torch.sigmoid(out).cpu().squeeze(1).tolist())
        val_loss = val_loss_tot / len(val_loader.dataset)
        val_preds = np.array(val_preds)
        val_prauc = average_precision_score(y_val, val_preds)
        
        scheduler.step()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_prauc"].append(val_prauc)
        
        if val_prauc > best_val_prauc:
            best_val_prauc = val_prauc
            best_weights   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        if epoch % 10 == 0 or epoch == EPOCHS:
            print(f"  Epoch {epoch:2d}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val PR-AUC: {val_prauc:.4f} (Best: {best_val_prauc:.4f})")
            
    train_time = time.time() - t0
    
    # Load best weights
    model.load_state_dict(best_weights)
    model.eval()
    
    # Find optimal classification threshold on Validation set (maximizing F1)
    val_preds = []
    with torch.no_grad():
        for X_b, _ in val_loader:
            val_preds.extend(torch.sigmoid(model(X_b)).cpu().squeeze(1).tolist())
    val_preds = np.array(val_preds)
    
    prec_arr, rec_arr, thresh_arr = precision_recall_curve(y_val, val_preds)
    f1_arr = 2 * (prec_arr[:-1] * rec_arr[:-1]) / (prec_arr[:-1] + rec_arr[:-1] + 1e-8)
    opt_idx = np.argmax(f1_arr)
    opt_threshold = float(thresh_arr[opt_idx]) if len(thresh_arr) > 0 else 0.5
    # Clamp threshold between 0.2 and 0.8 to prevent degeneracy
    opt_threshold = float(np.clip(opt_threshold, 0.20, 0.80))
    
    # Evaluate on Held-Out Test Set
    t_inf_start = time.time()
    test_preds = []
    with torch.no_grad():
        for X_b, _ in test_loader:
            test_preds.extend(torch.sigmoid(model(X_b)).cpu().squeeze(1).tolist())
    inf_time_ms = ((time.time() - t_inf_start) / len(y_test)) * 1000.0
    test_preds = np.array(test_preds)
    
    test_binary = (test_preds >= opt_threshold).astype(int)
    
    test_prec   = precision_score(y_test, test_binary, zero_division=0)
    test_rec    = recall_score(y_test, test_binary, zero_division=0)
    test_f1     = f1_score(y_test, test_binary, zero_division=0)
    test_acc    = accuracy_score(y_test, test_binary)
    test_prauc  = average_precision_score(y_test, test_preds)
    test_rocauc = roc_auc_score(y_test, test_preds)
    cm          = confusion_matrix(y_test, test_binary).tolist()
    
    print(f"\n{model_name} Test Results (Threshold={opt_threshold:.3f}):")
    print(f"  PR-AUC:    {test_prauc:.4f}")
    print(f"  ROC-AUC:   {test_rocauc:.4f}")
    print(f"  F1 Score:  {test_f1:.4f}")
    print(f"  Precision: {test_prec:.4f}")
    print(f"  Recall:    {test_rec:.4f}")
    print(f"  Accuracy:  {test_acc:.4f}")
    print(f"  Confusion: TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]}")
    print(f"  Inference: {inf_time_ms:.2f} ms/sample | Train Time: {train_time:.1f}s")
    
    results = {
        "model_name": model_name,
        "best_val_prauc": round(float(best_val_prauc), 4),
        "optimal_threshold": round(float(opt_threshold), 4),
        "test_pr_auc": round(float(test_prauc), 4),
        "test_roc_auc": round(float(test_rocauc), 4),
        "test_f1": round(float(test_f1), 4),
        "test_precision": round(float(test_prec), 4),
        "test_recall": round(float(test_rec), 4),
        "test_accuracy": round(float(test_acc), 4),
        "confusion_matrix": cm,
        "inference_ms_per_sample": round(float(inf_time_ms), 2),
        "training_time_seconds": round(float(train_time), 1),
        "test_preds": test_preds.tolist(),
        "history": history
    }
    
    return model, results

def main():
    print("=== Step 4: Training BDH, LSTM, and TFT on Authentic Dataset v4 ===")
    X_tr, y_tr, X_va, y_va, X_te, y_te, n_feats = load_data()
    
    pos_weight = (len(y_tr) - y_tr.sum()) / (y_tr.sum() + 1e-5)
    print(f"Computed Pos Weight = {pos_weight:.2f}")
    
    train_ds = SolarDataset(X_tr, y_tr)
    val_ds   = SolarDataset(X_va, y_va)
    test_ds  = SolarDataset(X_te, y_te)
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)
    
    models_to_train = {
        "BDH":  BabyDragonHatchlingV4(n_feats, hidden_dim=HIDDEN_DIM, dropout=DROPOUT),
        "LSTM": LSTMFlarePredictor(n_feats, hidden_dim=HIDDEN_DIM, dropout=DROPOUT),
        "TFT":  TemporalFusionTransformer(n_feats, hidden_dim=HIDDEN_DIM, dropout=DROPOUT)
    }
    
    all_results = {}
    trained_models = {}
    
    for name, m in models_to_train.items():
        trained_m, res = train_and_eval(name, m, train_loader, val_loader, test_loader, y_va, y_te, pos_weight)
        all_results[name] = res
        trained_models[name] = trained_m
        torch.save(trained_m.state_dict(), MODELS / f"{name.lower()}_v4.pt")
        print(f"Saved {name} model weights to models/{name.lower()}_v4.pt")
        
    # Save comparison JSON
    # Strip long histories from concise summary
    summary_results = {}
    for k, v in all_results.items():
        summary_results[k] = {ik: iv for ik, iv in v.items() if ik not in ("history", "test_preds")}
        
    with open(MODELS / "model_comparison_v4.json", "w", encoding="utf-8") as f:
        json.dump(summary_results, f, indent=2)
    print(f"\nSaved comparison summary to {MODELS / 'model_comparison_v4.json'}")
    
    # --------------------------------------------------------------------------
    # PLOTS
    # --------------------------------------------------------------------------
    print("\nGenerating publication-quality figures...")
    
    # 1. Training Curves
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    colors = {"BDH": "#e63946", "LSTM": "#457b9d", "TFT": "#2a9d8f"}
    for idx, (name, res) in enumerate(all_results.items()):
        ax = axes[idx]
        hist = res["history"]
        ax.plot(hist["train_loss"], label="Train Loss", color=colors[name], lw=2)
        ax.plot(hist["val_loss"], label="Val Loss", color=colors[name], lw=2, linestyle="--")
        ax.set_title(f"{name} Loss Convergence", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.3)
        ax.legend()
    plt.tight_layout()
    plt.savefig(DOCS / "training_curves_v4.png", dpi=200)
    plt.close()
    
    # 2. PR and ROC Curves on Held-Out Test Set
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    for name, res in all_results.items():
        preds = np.array(res["test_preds"])
        prec, rec, _ = precision_recall_curve(y_te, preds)
        fpr, tpr, _ = roc_curve(y_te, preds)
        ax1.plot(rec, prec, label=f"{name} (PR-AUC = {res['test_pr_auc']:.3f})", color=colors[name], lw=2.5)
        ax2.plot(fpr, tpr, label=f"{name} (ROC-AUC = {res['test_roc_auc']:.3f})", color=colors[name], lw=2.5)
        
    ax1.axhline(y=y_te.mean(), color="gray", linestyle=":", label=f"No-Skill Baseline ({y_te.mean():.3f})")
    ax1.set_title("Precision-Recall Curves (Held-Out Test Set)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Recall", fontsize=11)
    ax1.set_ylabel("Precision", fontsize=11)
    ax1.set_ylim(-0.02, 1.05)
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")
    
    ax2.plot([0, 1], [0, 1], color="gray", linestyle=":", label="Random Chance (0.500)")
    ax2.set_title("ROC Curves (Held-Out Test Set)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("False Positive Rate", fontsize=11)
    ax2.set_ylabel("True Positive Rate", fontsize=11)
    ax2.set_ylim(-0.02, 1.05)
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(DOCS / "pr_roc_curves_v4.png", dpi=200)
    plt.close()
    
    # 3. Confusion Matrices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for idx, (name, res) in enumerate(all_results.items()):
        ax = axes[idx]
        cm = np.array(res["confusion_matrix"])
        im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
        ax.set_title(f"{name} Confusion Matrix\n(F1={res['test_f1']:.3f})", fontsize=12, fontweight="bold")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["No Flare (0)", "Flare (1)"])
        ax.set_yticklabels(["No Flare (0)", "Flare (1)"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                text_col = "white" if val > cm.max() / 2 else "black"
                ax.text(j, i, f"{val}", ha="center", va="center", color=text_col, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(DOCS / "confusion_matrices_v4.png", dpi=200)
    plt.close()
    
    # 4. Model Comparison Bar Chart
    fig, ax = plt.subplots(figsize=(10, 5))
    metrics_names = ["test_pr_auc", "test_roc_auc", "test_f1", "test_precision", "test_recall"]
    display_names = ["PR-AUC", "ROC-AUC", "F1 Score", "Precision", "Recall"]
    x = np.arange(len(display_names))
    width = 0.25
    
    for idx, (name, col) in enumerate(colors.items()):
        vals = [summary_results[name][m] for m in metrics_names]
        ax.bar(x + (idx - 1) * width, vals, width, label=name, color=col)
        for i, v in enumerate(vals):
            ax.text(x[i] + (idx - 1) * width, v + 0.02, f"{v:.2f}", ha="center", fontsize=8, fontweight="bold")
            
    ax.set_title("Model Comparison on Held-Out Test Set (Authentic Observational Data v4)", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, fontsize=11)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(DOCS / "model_comparison_bar_v4.png", dpi=200)
    plt.close()
    
    print("All figures successfully saved to documentation/")

if __name__ == '__main__':
    main()

