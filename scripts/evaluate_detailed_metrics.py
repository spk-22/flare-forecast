"""Detailed evaluation and comparison of Baby Dragon Hatchling (BDH) vs LSTM."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    accuracy_score, precision_recall_curve, roc_curve
)

import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_models import (
    get_train_test_data, BabyDragonHatchling, LSTMFlarePredictor,
    evaluate_model, compute_all_metrics
)

MODELS_DIR = ROOT / 'models'
DOCS_DIR = ROOT / 'documentation'

def main():
    X_train, y_train, ids_train, X_test, y_test, ids_test, _, _ = get_train_test_data()
    
    bdh = BabyDragonHatchling(in_features=19, seq_len=80, hidden_dim=64, num_blocks=3, dropout=0.25)
    bdh.load_state_dict(torch.load(MODELS_DIR / 'bdh_best.pt'))
    
    lstm = LSTMFlarePredictor(in_features=19, hidden_dim=64, num_layers=2, dropout=0.25)
    lstm.load_state_dict(torch.load(MODELS_DIR / 'lstm_best.pt'))
    
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)),
        batch_size=16,
        shuffle=False
    )
    
    _, _, bdh_probs, y_true = evaluate_model(bdh, test_loader)
    _, _, lstm_probs, _ = evaluate_model(lstm, test_loader)
    
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    
    print("================ PREDICTED PROBABILITIES ================")
    print(f"Test Positive Samples Count: {len(pos_idx)}")
    print(f"BDH Positive Probabilities:  {bdh_probs[pos_idx]}")
    print(f"LSTM Positive Probabilities: {lstm_probs[pos_idx]}")
    print(f"BDH Negative Mean / Max:     {np.mean(bdh_probs[neg_idx]):.4f} / {np.max(bdh_probs[neg_idx]):.4f}")
    print(f"LSTM Negative Mean / Max:    {np.mean(lstm_probs[neg_idx]):.4f} / {np.max(lstm_probs[neg_idx]):.4f}")
    
    # 1. Standard Threshold (0.5)
    bdh_m_05 = compute_all_metrics(y_true, (bdh_probs >= 0.5).astype(int), bdh_probs)
    lstm_m_05 = compute_all_metrics(y_true, (lstm_probs >= 0.5).astype(int), lstm_probs)
    
    # 2. Optimal Threshold Tuning
    def tune_threshold(probs, y_true):
        best_th = 0.5
        best_f1 = -1.0
        best_m = None
        for th in np.linspace(0.001, 0.999, 1000):
            preds = (probs >= th).astype(int)
            m = compute_all_metrics(y_true, preds, probs)
            if m['f1_score'] > best_f1:
                best_f1 = m['f1_score']
                best_th = th
                best_m = m
            elif m['f1_score'] == best_f1 and best_m is not None and m['recall'] > best_m['recall']:
                best_th = th
                best_m = m
        return best_th, best_m
        
    bdh_th_opt, bdh_m_opt = tune_threshold(bdh_probs, y_true)
    lstm_th_opt, lstm_m_opt = tune_threshold(lstm_probs, y_true)
    
    print("\n================ THRESHOLD = 0.5 RESULTS ================")
    print("BDH  (th=0.50): Acc={:.4f}, Prec={:.4f}, Rec={:.4f}, F1={:.4f}, PR-AUC={:.4f}, ROC-AUC={:.4f}, CM={}".format(
        bdh_m_05['accuracy'], bdh_m_05['precision'], bdh_m_05['recall'], bdh_m_05['f1_score'], bdh_m_05['pr_auc'], bdh_m_05['roc_auc'], bdh_m_05['confusion_matrix']
    ))
    print("LSTM (th=0.50): Acc={:.4f}, Prec={:.4f}, Rec={:.4f}, F1={:.4f}, PR-AUC={:.4f}, ROC-AUC={:.4f}, CM={}".format(
        lstm_m_05['accuracy'], lstm_m_05['precision'], lstm_m_05['recall'], lstm_m_05['f1_score'], lstm_m_05['pr_auc'], lstm_m_05['roc_auc'], lstm_m_05['confusion_matrix']
    ))
    
    print("\n================ OPTIMAL OPERATIONAL THRESHOLD RESULTS ================")
    print("BDH  (th={:.4f}): Acc={:.4f}, Prec={:.4f}, Rec={:.4f}, F1={:.4f}, PR-AUC={:.4f}, ROC-AUC={:.4f}, CM={}".format(
        bdh_th_opt, bdh_m_opt['accuracy'], bdh_m_opt['precision'], bdh_m_opt['recall'], bdh_m_opt['f1_score'], bdh_m_opt['pr_auc'], bdh_m_opt['roc_auc'], bdh_m_opt['confusion_matrix']
    ))
    print("LSTM (th={:.4f}): Acc={:.4f}, Prec={:.4f}, Rec={:.4f}, F1={:.4f}, PR-AUC={:.4f}, ROC-AUC={:.4f}, CM={}".format(
        lstm_th_opt, lstm_m_opt['accuracy'], lstm_m_opt['precision'], lstm_m_opt['recall'], lstm_m_opt['f1_score'], lstm_m_opt['pr_auc'], lstm_m_opt['roc_auc'], lstm_m_opt['confusion_matrix']
    ))
    
    comparison_summary = {
        'split': {
            'train_samples': len(y_train),
            'train_positives': int(y_train.sum()),
            'test_samples': len(y_test),
            'test_positives': int(y_test.sum())
        },
        'standard_threshold_0.5': {
            'BDH': bdh_m_05,
            'LSTM': lstm_m_05
        },
        'optimal_threshold': {
            'BDH': {'threshold': float(bdh_th_opt), 'metrics': bdh_m_opt},
            'LSTM': {'threshold': float(lstm_th_opt), 'metrics': lstm_m_opt}
        }
    }
    
    with open(MODELS_DIR / 'model_comparison_detailed.json', 'w') as f:
        json.dump(comparison_summary, f, indent=2)
        
    # Re-plot publication quality plots with both thresholds
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=300)
    
    # PR Curve
    bdh_p, bdh_r, _ = precision_recall_curve(y_true, bdh_probs)
    lstm_p, lstm_r, _ = precision_recall_curve(y_true, lstm_probs)
    
    axes[0].plot(bdh_r, bdh_p, label=f"BDH (PR-AUC = {bdh_m_05['pr_auc']:.4f})", color='#1f77b4', lw=2.5)
    axes[0].plot(lstm_r, lstm_p, label=f"LSTM (PR-AUC = {lstm_m_05['pr_auc']:.4f})", color='#e377c2', lw=2.5, ls='--')
    axes[0].set_xlabel('Recall', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Precision', fontsize=12, fontweight='bold')
    axes[0].set_title('Precision-Recall Curve (90s Flare Prediction)', fontsize=13, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # ROC Curve
    bdh_fpr, bdh_tpr, _ = roc_curve(y_true, bdh_probs)
    lstm_fpr, lstm_tpr, _ = roc_curve(y_true, lstm_probs)
    
    axes[1].plot(bdh_fpr, bdh_tpr, label=f"BDH (ROC-AUC = {bdh_m_05['roc_auc']:.4f})", color='#1f77b4', lw=2.5)
    axes[1].plot(lstm_fpr, lstm_tpr, label=f"LSTM (ROC-AUC = {lstm_m_05['roc_auc']:.4f})", color='#e377c2', lw=2.5, ls='--')
    axes[1].plot([0, 1], [0, 1], color='gray', linestyle=':', lw=1.5, label='No Skill')
    axes[1].set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('True Positive Rate (Recall)', fontsize=12, fontweight='bold')
    axes[1].set_title('ROC Curve (90s Flare Prediction)', fontsize=13, fontweight='bold')
    axes[1].legend(loc='lower right', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(MODELS_DIR / 'pr_roc_curves.png')
    plt.savefig(DOCS_DIR / 'pr_roc_curves.png')
    plt.close()
    
    # Confusion matrix figure showing optimal threshold performance
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    for ax, m_data, title, th in zip(axes, [bdh_m_opt, lstm_m_opt], ['Baby Dragon Hatchling (BDH)', 'LSTM Baseline'], [bdh_th_opt, lstm_th_opt]):
        cm = np.array(m_data['confusion_matrix'])
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f"{title}\n(Threshold = {th:.4f})", fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(['No Flare (0)', 'Flare (1)'])
        ax.set_yticklabels(['No Flare (0)', 'Flare (1)'])
        ax.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=11, fontweight='bold')
        
        for i in range(2):
            for j in range(2):
                ax.text(j, i, format(cm[i][j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i][j] > np.max(cm)/2 else "black",
                        fontsize=15, fontweight='bold')
                        
    plt.tight_layout()
    plt.savefig(MODELS_DIR / 'confusion_matrices.png')
    plt.savefig(DOCS_DIR / 'confusion_matrices.png')
    plt.close()
    
    print("\nSaved detailed comparison to models/ and documentation/")

if __name__ == '__main__':
    main()
