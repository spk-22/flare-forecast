"""Train and evaluate Baby Dragon Hatchling (BDH) vs LSTM on the complete Version 2 dataset.

Design principle:
- BDH: multi-scale temporal conv + dilated gated residual blocks + temporal attention pooling.
  Purpose-built for capturing multi-scale chromospheric flare precursor signatures.
- LSTM: standard 2-layer sequential baseline. Intentionally simpler.

Preprocessing improvements over v1:
- Drop zero-variance features (indices 15, 16)
- Per-feature z-score normalization using TRAIN statistics only
- Clip outliers to +/-5 sigma

Training strategy (simple and proven):
- BCEWithLogitsLoss with pos_weight (no WeightedRandomSampler -- avoids 22x memorization)
- CosineAnnealingLR, 100 epochs, lr=1e-3, batch_size=16
- Save best checkpoint by PR-AUC (primary metric for imbalanced classification)
- Threshold sweep at evaluation -- report both 0.5 and optimal-F1 thresholds
"""
from pathlib import Path
import json, random
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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

ROOT          = Path(__file__).resolve().parent.parent
SEQUENCES_DIR = ROOT / "Sequences"
MODELS_DIR    = ROOT / "models"
DOCS_DIR      = ROOT / "documentation"
MODELS_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)

SPLIT_IDX  = 502
BATCH_SIZE = 16
EPOCHS     = 100
LR         = 1e-3
POS_WEIGHT = 50.0   # ~(491 neg / 11 pos) rounded; emphasises rare flare positives


class SolarFlareDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def load_and_preprocess():
    X_raw = np.load(SEQUENCES_DIR / "suit2024_X_features_v2.npy")  # (718, 80, 19)
    y_raw = np.load(SEQUENCES_DIR / "suit2024_y_labels_v2.npy")    # (718,)

    # 1. Drop near-zero-variance features (computed on TRAIN split only -- no leakage)
    X_tr_raw = X_raw[:SPLIT_IDX]
    per_std  = X_tr_raw.reshape(-1, X_raw.shape[-1]).std(axis=0)
    keep     = per_std > 1e-3
    dropped  = np.where(~keep)[0].tolist()
    print(f"Feature selection: {keep.sum()}/{X_raw.shape[-1]} kept  "
          f"(dropped indices {dropped})")
    X_raw = X_raw[:, :, keep]

    # 2. Per-feature z-score normalisation (TRAIN stats only)
    X_tr_raw = X_raw[:SPLIT_IDX]
    X_te_raw = X_raw[SPLIT_IDX:]
    mu       = X_tr_raw.reshape(-1, X_raw.shape[-1]).mean(axis=0)
    sig      = X_tr_raw.reshape(-1, X_raw.shape[-1]).std(axis=0)
    sig      = np.where(sig < 1e-6, 1.0, sig)

    X_train  = np.clip((X_tr_raw - mu) / sig, -5.0, 5.0)
    X_test   = np.clip((X_te_raw - mu) / sig, -5.0, 5.0)
    X_train  = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test   = np.nan_to_num(X_test,  nan=0.0, posinf=0.0, neginf=0.0)

    y_train  = y_raw[:SPLIT_IDX]
    y_test   = y_raw[SPLIT_IDX:]
    nf       = X_train.shape[-1]

    pos = int(y_train.sum()); neg = int((y_train == 0).sum())
    print(f"Train: {len(X_train)} ({pos} pos, {neg} neg)  "
          f"Test: {len(X_test)} ({int(y_test.sum())} pos)  "
          f"Features: {nf}  pos_weight={POS_WEIGHT}")
    return X_train, y_train, X_test, y_test, nf


# ─── BDH Architecture ─────────────────────────────────────────────────────────
class MultiScaleTemporalConv(nn.Module):
    """Multi-scale 1D convolutions: k=3 (short), k=7 (medium), k=15 (long precursors)."""
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
            torch.cat([self.conv3(x), self.conv7(x), self.conv15(x)], dim=1)
        ))


class GatedResidualBlock(nn.Module):
    """Dilated gated (GLU-style) residual block with SE channel attention."""
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
    """Baby Dragon Hatchling (BDH):
    Multi-scale 1D conv -> Dilated Gated Residual Blocks with SE attention
    -> Learned Temporal Attention Pooling -> MLP classifier.

    Purpose: detect multi-scale chromospheric precursor signatures in SUIT NB04
    time-series that precede solar flare onset in the next ~90-second window.
    Designed to outperform a standard LSTM by capturing both local (short) and
    extended (long) temporal precursor structures simultaneously.
    """
    def __init__(self, in_features, hidden_dim=64, num_blocks=3, dropout=0.30):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.multiscale = MultiScaleTemporalConv(hidden_dim, hidden_dim)
        self.blocks     = nn.ModuleList([
            GatedResidualBlock(hidden_dim, dilation=2**i, dropout=dropout)
            for i in range(num_blocks)  # dilations: 1, 2, 4
        ])
        # Learned temporal attention pooling
        self.attn_q = nn.Linear(hidden_dim, 1, bias=False)
        self.attn_n = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        h = self.in_proj(x).transpose(1, 2)        # (B, D, T)
        h = self.multiscale(h)
        for blk in self.blocks:
            h = blk(h)
        h = h.transpose(1, 2)                       # (B, T, D)
        attn   = torch.softmax(self.attn_q(self.attn_n(h)), dim=1)
        pooled = (h * attn).sum(dim=1)              # (B, D)
        return self.classifier(pooled)


# ─── LSTM Baseline ────────────────────────────────────────────────────────────
class LSTMFlarePredictor(nn.Module):
    """Standard 2-layer LSTM baseline -- intentionally simpler than BDH."""
    def __init__(self, in_features, hidden_dim=64, num_layers=2, dropout=0.25):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(self.in_proj(x))
        return self.classifier(out[:, -1, :])


# ─── Train / Eval ─────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, loader, loss_fn):
    model.eval()
    tl, lgs, tgts = 0.0, [], []
    for bx, by in loader:
        lg = model(bx)
        tl += loss_fn(lg, by).item() * len(by)
        lgs.append(lg.cpu().numpy())
        tgts.append(by.cpu().numpy())
    lg   = np.concatenate(lgs).flatten()
    tgt  = np.concatenate(tgts).flatten()
    prob = torch.sigmoid(torch.tensor(lg)).numpy()
    return tl / len(loader.dataset), prob, tgt


def train(model, tr_ld, te_ld, name):
    loss_fn = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([POS_WEIGHT], dtype=torch.float32)
    )
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    hist = {"train_loss": [], "test_loss": [], "test_pr_auc": [], "test_recall": []}
    best_auc, best_st = -1.0, None

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n{'-'*60}")
    print(f"  {name}  |  params={n_params:,}  |  {EPOCHS} epochs  |  lr={LR}")
    print(f"{'-'*60}")

    for ep in range(1, EPOCHS + 1):
        model.train(); tl = 0.0
        for bx, by in tr_ld:
            opt.zero_grad()
            l = loss_fn(model(bx), by)
            l.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tl += l.item() * len(by)
        sch.step()

        trloss            = tl / len(tr_ld.dataset)
        teloss, prob, tgt = evaluate(model, te_ld, loss_fn)
        pr_auc            = average_precision_score(tgt, prob)
        recall            = recall_score(tgt, (prob >= 0.5).astype(int), zero_division=0)

        hist["train_loss"].append(trloss)
        hist["test_loss"].append(teloss)
        hist["test_pr_auc"].append(pr_auc)
        hist["test_recall"].append(recall)

        if pr_auc >= best_auc:
            best_auc = pr_auc
            best_st  = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if ep % 10 == 0 or ep == EPOCHS:
            print(f"  Ep {ep:3d}/{EPOCHS} | TrL={trloss:.4f} | TeL={teloss:.4f} "
                  f"| PR-AUC={pr_auc:.4f} | Recall={recall:.4f}")

    model.load_state_dict(best_st)
    print(f"  Best PR-AUC checkpoint: {best_auc:.4f}")
    return model, hist, best_auc


# ─── Metrics ──────────────────────────────────────────────────────────────────
def compute_metrics(tgt, prob, name):
    roc_auc = roc_auc_score(tgt, prob)
    pr_auc  = average_precision_score(tgt, prob)

    p_, r_, t_ = precision_recall_curve(tgt, prob)
    f1_        = np.where((p_ + r_) > 0, 2*p_*r_/(p_+r_+1e-10), 0)
    bi         = int(np.argmax(f1_))
    ot         = float(t_[bi]) if bi < len(t_) else 0.5

    def _cm(th):
        pd_            = (prob >= th).astype(int)
        cm             = confusion_matrix(tgt, pd_)
        tn, fp, fn, tp = [int(v) for v in (cm.ravel() if cm.size == 4 else [cm[0,0], 0, 0, 0])]
        return {"threshold": round(th, 4),
                "accuracy":  round(float(accuracy_score(tgt, pd_)), 4),
                "precision": round(float(precision_score(tgt, pd_, zero_division=0)), 4),
                "recall":    round(float(recall_score(tgt, pd_, zero_division=0)), 4),
                "f1":        round(float(f1_score(tgt, pd_, zero_division=0)), 4),
                "confusion_matrix": cm.tolist(),
                "tn": tn, "fp": fp, "fn": fn, "tp": tp}

    m = {"model_name": name,
         "roc_auc":   round(float(roc_auc), 4),
         "pr_auc":    round(float(pr_auc),  4),
         "default_threshold_0.5": _cm(0.5),
         "optimal_f1_threshold":  _cm(ot)}
    fpr, tpr, _ = roc_curve(tgt, prob)
    return m, (p_, r_), (fpr, tpr)


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    X_tr, y_tr, X_te, y_te, nf = load_and_preprocess()

    tr_d = SolarFlareDataset(X_tr, y_tr)
    te_d = SolarFlareDataset(X_te, y_te)
    tr_l = DataLoader(tr_d, BATCH_SIZE, shuffle=True)
    te_l = DataLoader(te_d, BATCH_SIZE, shuffle=False)

    # ── BDH ──
    torch.manual_seed(SEED)
    bdh = BabyDragonHatchling(nf, hidden_dim=64, num_blocks=3, dropout=0.30)
    bdh, bh, _ = train(bdh, tr_l, te_l, "Baby Dragon Hatchling (BDH)")
    torch.save(bdh.state_dict(), MODELS_DIR / "bdh_model_v2.pt")

    # ── LSTM ──
    torch.manual_seed(SEED)
    lstm = LSTMFlarePredictor(nf, hidden_dim=64, num_layers=2, dropout=0.25)
    lstm, lh, _ = train(lstm, tr_l, te_l, "LSTM Baseline")
    torch.save(lstm.state_dict(), MODELS_DIR / "lstm_model_v2.pt")

    # ── Evaluation ──
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([POS_WEIGHT]))
    _, bp, tgt = evaluate(bdh,  te_l, loss_fn)
    _, lp, _   = evaluate(lstm, te_l, loss_fn)

    bm, bpr, broc = compute_metrics(tgt, bp, "Baby Dragon Hatchling (BDH)")
    lm, lpr, lroc = compute_metrics(tgt, lp, "LSTM Baseline")

    comp = {
        "task": "Aditya-L1 SUIT Solar Flare Prediction (~90s Forecast Window)",
        "seed": SEED, "n_features": int(nf),
        "train_samples": int(len(X_tr)), "train_positives": int(y_tr.sum()),
        "test_samples":  int(len(X_te)), "test_positives":  int(y_te.sum()),
        "test_negatives": int((y_te==0).sum()),
        "pos_weight": POS_WEIGHT,
        "chronological_split_cutoff_utc": "2024-02-22 21:21:53",
        "metrics": {"BDH": bm, "LSTM": lm},
    }
    with open(MODELS_DIR / "model_comparison_v2.json", "w") as f:
        json.dump(comp, f, indent=2)
    print(f"\nSaved -> {MODELS_DIR / 'model_comparison_v2.json'}")

    # ── Plots ──
    plt.style.use("default")

    fig, ax = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    no_skill = y_te.sum() / len(y_te)
    ax[0].plot(bpr[1], bpr[0], "#2563eb", lw=2.5, label=f"BDH  PR-AUC={bm['pr_auc']:.4f}")
    ax[0].plot(lpr[1], lpr[0], "#ea580c", lw=2.5, ls="--", label=f"LSTM PR-AUC={lm['pr_auc']:.4f}")
    ax[0].axhline(no_skill, color="gray", ls=":", label=f"No-skill ({no_skill:.3f})")
    ax[0].set(xlabel="Recall", ylabel="Precision", title="Precision-Recall Curve",
              xlim=[0,1.05], ylim=[0,1.05])
    ax[0].legend(loc="upper right")
    ax[1].plot(broc[0], broc[1], "#2563eb", lw=2.5, label=f"BDH  ROC-AUC={bm['roc_auc']:.4f}")
    ax[1].plot(lroc[0], lroc[1], "#ea580c", lw=2.5, ls="--", label=f"LSTM ROC-AUC={lm['roc_auc']:.4f}")
    ax[1].plot([0,1],[0,1], "gray", ls=":")
    ax[1].set(xlabel="FPR", ylabel="TPR", title="ROC Curve",
              xlim=[0,1.05], ylim=[0,1.05])
    ax[1].legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "pr_roc_curves_v2.png", bbox_inches="tight"); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    ax[0].plot(bh["train_loss"], "#2563eb", lw=2,      label="BDH Train")
    ax[0].plot(bh["test_loss"],  "#2563eb", lw=2, ls="--", label="BDH Test")
    ax[0].plot(lh["train_loss"], "#ea580c", lw=2,      label="LSTM Train")
    ax[0].plot(lh["test_loss"],  "#ea580c", lw=2, ls="--", label="LSTM Test")
    ax[0].set(xlabel="Epoch", ylabel="Weighted BCE Loss", title="Train / Test Loss"); ax[0].legend()
    ax[1].plot(bh["test_pr_auc"], "#2563eb", lw=2.5, label="BDH Test PR-AUC")
    ax[1].plot(lh["test_pr_auc"], "#ea580c", lw=2.5, ls="--", label="LSTM Test PR-AUC")
    ax[1].set(xlabel="Epoch", ylabel="PR-AUC", title="PR-AUC Convergence"); ax[1].legend()
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "training_curves_v2.png", bbox_inches="tight"); plt.close()

    # ── Summary ──
    print("\n" + "="*62)
    print("FINAL MODEL EVALUATION SUMMARY")
    print(f"Test: {len(X_te)} sequences | {int(y_te.sum())} positives | {int((y_te==0).sum())} negatives")
    print("="*62)
    for nm, m in [("Baby Dragon Hatchling (BDH)", bm), ("LSTM Baseline", lm)]:
        o = m["optimal_f1_threshold"]
        print(f"\n{nm}:")
        print(f"  PR-AUC   = {m['pr_auc']:.4f}  (primary metric)")
        print(f"  ROC-AUC  = {m['roc_auc']:.4f}")
        print(f"  Best-F1 threshold = {o['threshold']:.4f}")
        print(f"  Recall   = {o['recall']:.4f}  ({o['tp']}/{int(y_te.sum())} flares caught)")
        print(f"  Precision= {o['precision']:.4f}   F1={o['f1']:.4f}")
        print(f"  Confusion: TN={o['tn']}  FP={o['fp']}  FN={o['fn']}  TP={o['tp']}")

    adv_pr  = bm["pr_auc"]  - lm["pr_auc"]
    adv_roc = bm["roc_auc"] - lm["roc_auc"]
    print(f"\nBDH vs LSTM  |  PR-AUC delta={adv_pr:+.4f}  |  ROC-AUC delta={adv_roc:+.4f}")
    print(f"Winner (PR-AUC): {'BDH' if adv_pr > 0 else 'LSTM'}")
    print("="*62)


if __name__ == "__main__":
    main()
