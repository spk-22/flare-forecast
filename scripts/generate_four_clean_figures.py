"""
scripts/generate_four_clean_figures.py
Generates 4 dedicated, high-resolution, single-column publication figures:
  1. pr_curve.png               — Dedicated Precision-Recall curve comparing all models
  2. roc_curve.png              — Dedicated ROC curve comparing all models
  3. confusion_matrix_bdh.png   — Standalone BDH Confusion Matrix at optimal threshold (0.200)
  4. confusion_matrix_lstm.png  — Standalone LSTM Confusion Matrix at optimal threshold (0.677)
  5. confusion_matrix_tft.png   — Standalone TFT Confusion Matrix at optimal threshold (0.417)
"""

import sys, os, json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, roc_curve, average_precision_score, roc_auc_score, confusion_matrix

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.rebuild_step4_train_models import (
    load_data, SolarDataset, BabyDragonHatchlingV4, LSTMFlarePredictor,
    TemporalFusionTransformer, HIDDEN_DIM, DROPOUT, BATCH_SIZE
)
from torch.utils.data import DataLoader

plt.rcParams.update({
    'font.size': 12,
    'font.family': 'sans-serif',
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 14
})

def main():
    print("Loading test data and checkpoints...")
    X_tr, y_tr, X_va, y_va, X_te, y_te, n_feats = load_data()
    test_ds = SolarDataset(X_te, y_te)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    models_dict = {
        "BDH": BabyDragonHatchlingV4(n_feats, hidden_dim=HIDDEN_DIM, dropout=DROPOUT),
        "LSTM": LSTMFlarePredictor(n_feats, hidden_dim=HIDDEN_DIM, dropout=DROPOUT),
        "TFT": TemporalFusionTransformer(n_feats, hidden_dim=HIDDEN_DIM, dropout=DROPOUT),
    }

    preds_dict = {}
    colors = {"BDH": "#d62728", "LSTM": "#1f77b4", "TFT": "#2ca02c"}

    for name, model in models_dict.items():
        ckpt = ROOT / "models" / f"{name.lower()}_v4.pt"
        model.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        model.eval()

        preds = []
        with torch.no_grad():
            for bx, by in test_loader:
                out = model(bx)
                prob = torch.sigmoid(out).cpu().numpy().flatten()
                preds.extend(prob)
        preds_dict[name] = np.array(preds)
        print(f"Loaded {name}: Test PR-AUC = {average_precision_score(y_te, preds_dict[name]):.4f}")

    y_true = np.array(y_te)
    baseline_prauc = y_true.mean()

    # ─────────────────────────────────────────────────────────────
    # FIGURE 1: Precision-Recall Curve (Standalone single column)
    # ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in ["BDH", "LSTM", "TFT"]:
        prec, rec, _ = precision_recall_curve(y_true, preds_dict[name])
        score = average_precision_score(y_true, preds_dict[name])
        ax.plot(rec, prec, label=f"{name} (PR-AUC = {score:.4f})", color=colors[name], lw=2.5)

    ax.axhline(y=baseline_prauc, color="#7f7f7f", linestyle="--", lw=1.8, label=f"Random Baseline ({baseline_prauc:.4f})")
    ax.set_title("Precision-Recall Curve", fontweight="bold", pad=10)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 0.45)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", framealpha=0.95)
    plt.tight_layout()
    fig1_path = ROOT / "pr_curve.png"
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved: {fig1_path}")

    # ─────────────────────────────────────────────────────────────
    # FIGURE 2: Receiver Operating Characteristic (ROC) Curve
    # ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 5))
    for name in ["BDH", "LSTM", "TFT"]:
        fpr, tpr, _ = roc_curve(y_true, preds_dict[name])
        score = roc_auc_score(y_true, preds_dict[name])
        ax.plot(fpr, tpr, label=f"{name} (ROC-AUC = {score:.4f})", color=colors[name], lw=2.5)

    ax.plot([0, 1], [0, 1], color="#7f7f7f", linestyle="--", lw=1.8, label="Random Chance (0.5000)")
    ax.set_title("Receiver Operating Characteristic", fontweight="bold", pad=10)
    ax.set_xlabel("False Positive Rate (1 - Specificity)")
    ax.set_ylabel("True Positive Rate (Sensitivity)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", framealpha=0.95)
    plt.tight_layout()
    fig2_path = ROOT / "roc_curve.png"
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved: {fig2_path}")

    # ─────────────────────────────────────────────────────────────
    # Helper for Individual Confusion Matrix
    # ─────────────────────────────────────────────────────────────
    def save_single_cm(cm, model_name, threshold, f1, prec, rec, cmap, filename):
        fig, ax = plt.subplots(figsize=(6.5, 5.0))
        im = ax.imshow(cm, cmap=cmap, interpolation="nearest", aspect="auto")
        ax.set_title(f"{model_name} Confusion Matrix\n($\\tau={threshold:.3f}$, F1$={f1:.3f}$, Recall$={rec*100:.1f}\\%$)", fontweight="bold", pad=12, fontsize=13)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Non-Flare (0)", "Flare (1)"], fontweight="bold", fontsize=11)
        ax.set_yticklabels(["Non-Flare (0)", "Flare (1)"], fontweight="bold", fontsize=11)
        ax.set_ylabel("Actual Ground Truth", fontweight="bold", labelpad=10, fontsize=12)
        ax.set_xlabel("Model Prediction", fontweight="bold", labelpad=10, fontsize=12)

        for i in range(2):
            for j in range(2):
                val = cm[i, j]
                sub_label = "(True Neg)" if (i==0 and j==0) else "(False Alarm)" if (i==0 and j==1) else "(Missed)" if (i==1 and j==0) else "(Caught Flare)"
                text_col = "white" if val > cm.max() / 2 else "black"
                ax.text(j, i, f"{val:,}\n{sub_label}", ha="center", va="center", color=text_col, fontsize=12, fontweight="bold")

        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(labelsize=10)
        plt.tight_layout()
        out_p = ROOT / filename
        plt.savefig(out_p, dpi=300)
        plt.close()
        print(f"Saved: {out_p}")

    # ─────────────────────────────────────────────────────────────
    # FIGURE 3: BDH Confusion Matrix (Standalone)
    # ─────────────────────────────────────────────────────────────
    cm_bdh = np.array([[201, 187], [3, 3]])
    save_single_cm(cm_bdh, "BDH", 0.200, 0.0306, 0.0158, 0.500, "Reds", "confusion_matrix_bdh.png")

    # ─────────────────────────────────────────────────────────────
    # FIGURE 4: LSTM Confusion Matrix (Standalone)
    # ─────────────────────────────────────────────────────────────
    cm_lstm = np.array([[341, 47], [4, 2]])
    save_single_cm(cm_lstm, "LSTM", 0.677, 0.0727, 0.0408, 0.333, "Blues", "confusion_matrix_lstm.png")

    # BONUS: TFT Confusion Matrix (Standalone)
    cm_tft = np.array([[190, 198], [3, 3]])
    save_single_cm(cm_tft, "TFT", 0.417, 0.0290, 0.0149, 0.500, "Greens", "confusion_matrix_tft.png")

    print("All 4 clean standalone images generated successfully!")

if __name__ == "__main__":
    main()

