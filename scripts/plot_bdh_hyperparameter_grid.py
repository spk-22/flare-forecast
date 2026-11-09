"""
scripts/plot_bdh_hyperparameter_grid.py
Generates publication-quality figures summarizing the BDH hyperparameter grid search
for inclusion in the IEEE paper / supplementary materials.
"""

import os, json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
JSON_PATH = ROOT / "models" / "bdh_hyperparameter_grid_results.json"
OUT_PNG = ROOT / "bdh_hyperparameter_ablation.png"

with open(JSON_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

results = data["all_results_ranked"]

names = [r["name"].replace(" (Official Default)", "").replace(" (Standard Width D=128)", "").replace(" (Low Dropout p=0.05)", " (Low Drop)").replace(" (Deep Regularized)", " (Deep Reg)") for r in results]
pr_aucs = [r["test_pr_auc"] for r in results]
roc_aucs = [r["test_roc_auc"] for r in results]
params_m = [r["n_params"] / 1e6 for r in results]
f1s = [r["test_f1"] for r in results]

# Reverse for horizontal bar chart (top ranked at top)
names = names[::-1]
pr_aucs = pr_aucs[::-1]
roc_aucs = roc_aucs[::-1]
params_m = params_m[::-1]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 7), sharey=True)

y_pos = np.arange(len(names))

# Plot 1: PR-AUC
bars1 = ax1.barh(y_pos, pr_aucs, color='#1f77b4', alpha=0.85, edgecolor='black', height=0.65)
ax1.axvline(0.0152, color='red', linestyle='--', linewidth=1.5, label='Random Baseline (0.0152)')
ax1.set_xlabel('Test PR-AUC', fontsize=11, fontweight='bold')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(names, fontsize=9)
ax1.set_title('(a) Precision-Recall AUC (Sorted by Rank)', fontsize=12, fontweight='bold')
ax1.grid(axis='x', linestyle=':', alpha=0.6)
ax1.legend(loc='lower right', fontsize=10)

# Annotate bars
for bar, prauc in zip(bars1, pr_aucs):
    ax1.text(prauc + 0.0008, bar.get_y() + bar.get_height()/2, f"{prauc:.4f}", va='center', fontsize=8)

# Plot 2: ROC-AUC
bars2 = ax2.barh(y_pos, roc_aucs, color='#2ca02c', alpha=0.85, edgecolor='black', height=0.65)
ax2.axvline(0.5000, color='red', linestyle='--', linewidth=1.5, label='Chance (0.50)')
ax2.set_xlabel('Test ROC-AUC', fontsize=11, fontweight='bold')
ax2.set_title('(b) Receiver Operating Characteristic AUC', fontsize=12, fontweight='bold')
ax2.grid(axis='x', linestyle=':', alpha=0.6)
ax2.legend(loc='lower right', fontsize=10)

for bar, roc in zip(bars2, roc_aucs):
    ax2.text(roc + 0.01, bar.get_y() + bar.get_height()/2, f"{roc:.3f}", va='center', fontsize=8)

plt.tight_layout()
plt.savefig(OUT_PNG, dpi=300)
print(f"Generated {OUT_PNG} successfully!")
