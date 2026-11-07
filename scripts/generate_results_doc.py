"""
generate_results_doc.py — Auto-generate results.md from actual training outputs.
Run AFTER train_models_v3.py has completed.
"""

from pathlib import Path
import json, numpy as np
from datetime import datetime

ROOT    = Path(__file__).resolve().parent.parent
MODELS  = ROOT / "models"
DOCS    = ROOT / "documentation"


def load_comparison():
    p = MODELS / "model_comparison_v3.json"
    if not p.exists():
        raise FileNotFoundError("Run train_models_v3.py first.")
    with open(p) as f:
        return json.load(f)


def fmt(v, precision=4):
    if v is None: return "N/A"
    if isinstance(v, float): return f"{v:.{precision}f}"
    return str(v)


def model_section(name, m):
    ot  = m.get("optimal_f1_threshold", {})
    d05 = m.get("default_threshold_0.5", {})
    cm  = ot.get("confusion_matrix", [[0,0],[0,0]])
    return f"""
#### {name}

| Metric | Value |
|--------|-------|
| **PR-AUC** (primary) | **{fmt(m.get('pr_auc'))}** |
| ROC-AUC | {fmt(m.get('roc_auc'))} |
| Best-F1 threshold | {fmt(ot.get('threshold'))} |
| Precision @ optimal | {fmt(ot.get('precision'))} |
| Recall @ optimal | {fmt(ot.get('recall'))} |
| F1 @ optimal | {fmt(ot.get('f1'))} |
| Accuracy @ optimal | {fmt(ot.get('accuracy'))} |
| FP (false alarms) | {ot.get('fp', 'N/A')} |
| FN (missed flares) | {ot.get('fn', 'N/A')} |
| TP (caught flares) | {ot.get('tp', 'N/A')} |
| TN | {ot.get('tn', 'N/A')} |
| Parameters | {m.get('n_params', 'N/A'):,} |
| Training time | {m.get('training_time_s', 'N/A')} s |
| Val PR-AUC (best ckpt) | {fmt(m.get('best_val_pr_auc'))} |

**Confusion matrix @ optimal threshold:**
```
              Predicted Neg  Predicted Pos
Actual Neg         {cm[0][0] if len(cm)>0 and len(cm[0])>0 else 0}            {cm[0][1] if len(cm)>0 and len(cm[0])>1 else 0}
Actual Pos         {cm[1][0] if len(cm)>1 and len(cm[1])>0 else 0}            {cm[1][1] if len(cm)>1 and len(cm[1])>1 else 0}
```

**At default threshold (0.5):** Precision={fmt(d05.get('precision'))}  Recall={fmt(d05.get('recall'))}  F1={fmt(d05.get('f1'))}
"""


def comparison_table(exp_data, exp_label):
    metrics_map = exp_data.get("metrics", {})
    models = ["BDH", "LSTM", "TFT"]
    rows = []
    pr_aucs = {m: (metrics_map.get(m) or {}).get("pr_auc", 0) for m in models}
    best_pr = max(pr_aucs.values()) if pr_aucs else 0

    for m in models:
        mx = metrics_map.get(m) or {}
        ot = mx.get("optimal_f1_threshold") or {}
        marker = " ★" if mx.get("pr_auc") == best_pr and best_pr > 0 else ""
        rows.append(
            f"| {m}{marker} | {fmt(mx.get('pr_auc'))} | {fmt(mx.get('roc_auc'))} "
            f"| {fmt(ot.get('f1'))} | {fmt(ot.get('recall'))} | {fmt(ot.get('precision'))} "
            f"| {ot.get('fp','N/A')} | {ot.get('fn','N/A')} |"
        )
    return f"""
| Model | PR-AUC | ROC-AUC | F1 | Recall | Precision | FP | FN |
|-------|--------|---------|----|----|---|---|---|
""" + "\n".join(rows) + "\n\n★ = best PR-AUC in this experiment"


def generate(data):
    r       = data
    aug     = r.get("augmentation", {})
    orig_d  = r.get("original", {})
    aug_d   = r.get("augmented", {})

    orig_metrics = orig_d.get("metrics", {})
    aug_metrics  = aug_d.get("metrics", {})

    # Determine winner
    models = ["BDH", "LSTM", "TFT"]
    best_orig = max(models, key=lambda m: (orig_metrics.get(m) or {}).get("pr_auc", 0))
    best_aug  = max(models, key=lambda m: (aug_metrics.get(m) or {}).get("pr_auc", 0))

    md = f"""# Solar Flare Forecasting v3 — Results

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")}  
**Task:** Predict solar flare onset in the next ~90 seconds from SUIT NB04 (Mg II h, 2803 Å) time-series features.  
**Dataset:** Aditya-L1 SUIT archive — all events from 2024-02-22, AR 3590.

---

## 1. Dataset Summary

### Independent Flare Events

All positive sequences come from **7 independent flare events** on a single day (2024-02-22) from a single
active region (AR 3590). This is the most important limitation of this study.

| # | GOES Class | Onset UTC | Train/Test | Sequences |
|---|-----------|-----------|------------|-----------|
| 1 | C3.4 | 2024-02-22 15:27 | Train | 1 |
| 2 | C9.4 | 2024-02-22 16:17 | Train | 1 |
| 3 | C4.7 | 2024-02-22 17:08 | Train | 3 (2 tracks) |
| 4 | C4.5 | 2024-02-22 17:45 | Train | 2 (2 tracks) |
| 5 | C4.7 | 2024-02-22 18:15 | Train | 2 (2 tracks) |
| 6 | M4.8 | 2024-02-22 20:29 | Train | 2 (2 tracks) |
| 7 | X6.3 | 2024-02-22 22:08 | **Test** | 3 (2 tracks) |

> **Critical note:** The test set contains exactly ONE independent flare event. All reported metrics
> are based on predicting 3 sequences from this single event. PR-AUC and F1 estimates are
> therefore highly uncertain.

### Original Dataset

| Split | Sequences | Positives | Negatives |
|-------|-----------|-----------|-----------|
| Train (seq 0–427) | {orig_d.get('train_samples', 'N/A')} | {orig_d.get('train_positives', 'N/A')} | {(orig_d.get('train_samples',0)) - (orig_d.get('train_positives',0))} |
| Validation (seq 428–501) | {orig_d.get('val_samples','N/A')} | {orig_d.get('val_positives','N/A')} | — |
| Test (seq 502–717) | {orig_d.get('test_samples','N/A')} | {orig_d.get('test_positives','N/A')} | {(orig_d.get('test_samples',0)) - (orig_d.get('test_positives',0))} |

### Augmented Dataset

| Item | Value |
|------|-------|
| Augmentation method | Multivariate Gaussian perturbation |
| Sigma scale | {aug.get('sigma_scale', 0.05)} (5% of per-feature std) |
| Synthetic samples generated | {aug.get('n_synth', 39)} |
| Independent flare events augmented | {aug.get('n_flare_events', 7)} |
| Total training positives after augmentation | {aug.get('n_train_pos_aug', 50)} |
| Synthetic = new flare events? | **NO** — these are variations of 7 real events |

| Split | Sequences | Positives |
|-------|-----------|-----------|
| Augmented train | {aug_d.get('train_samples','N/A')} | {aug_d.get('train_positives','N/A')} |
| Validation (unchanged) | {aug_d.get('val_samples','N/A')} | {aug_d.get('val_positives','N/A')} |
| Test (unchanged) | {aug_d.get('test_samples','N/A')} | {aug_d.get('test_positives','N/A')} |

---

## 2. Features

19 features extracted per FITS frame; 2 dropped (zero variance in training data):

| # | Feature | Physical Meaning | Status |
|---|---------|-----------------|--------|
| 1 | mean_dn | Mean pixel brightness (Data Number) | ✓ Used |
| 2 | std_dn | Std of pixel distribution | ✓ Used |
| 3 | min_dn | Minimum pixel value | ✓ Used |
| 4 | max_dn | Maximum pixel value | ✓ Used |
| 5 | p01_dn | 1st percentile brightness | ✓ Used |
| 6 | p10_dn | 10th percentile brightness | ✓ Used |
| 7 | p50_dn | Median brightness | ✓ Used |
| 8 | p90_dn | 90th percentile brightness | ✓ Used |
| 9 | p95_dn | 95th percentile brightness | ✓ Used |
| 10 | p99_dn | 99th percentile brightness | ✓ Used |
| 11 | p999_dn | 99.9th percentile brightness | ✓ Used |
| 12 | skew_dn | Skewness of pixel distribution | ✓ Used |
| 13 | kurt_dn | Kurtosis (peakedness) | ✓ Used |
| 14 | enhancement_ratio | (p99-p50)/std — localized brightening | ✓ Used |
| 15 | bright_pix_frac | Fraction of pixels above p50+2σ | ✓ Used |
| 16 | quality_percent | SUIT telemetry quality flag | **Dropped** (always 100%) |
| 17 | exposure_s | Exposure time | **Dropped** (always 0) |
| 18 | hpc_tx_center_arcsec | Helioprojective X (solar longitude) | ✓ Used |
| 19 | hpc_ty_center_arcsec | Helioprojective Y (solar latitude) | ✓ Used |

**Active features: 17** | Sequence length: 80 frames | Input shape: (batch, 80, 17)

---

## 3. Training Configuration

### Shared settings (all models, both experiments)

| Parameter | Value |
|-----------|-------|
| Seed | {r.get('seed', 42)} |
| Sequence length | 80 frames |
| Batch size | 16 |
| Epochs | 120 |
| Learning rate | 1e-3 |
| LR schedule | CosineAnnealingLR (eta_min=1e-5) |
| Optimizer | AdamW (weight_decay=1e-3) |
| Loss | BCEWithLogitsLoss |
| Checkpoint criterion | Best val PR-AUC |
| Gradient clipping | 1.0 |

| Experiment | pos_weight | Rationale |
|-----------|-----------|-----------|
| Original | {orig_d.get('pos_weight', 44.6)} | ~{orig_d.get('train_samples',428) - orig_d.get('train_positives',7)} neg / {orig_d.get('train_positives',7)} pos |
| Augmented | {aug_d.get('pos_weight', 9.8)} | ~{aug_d.get('train_samples',541) - aug_d.get('train_positives',50)} neg / {aug_d.get('train_positives',50)} pos |

### Model-specific hyperparameters

| Model | Hidden dim | Key params | Dropout |
|-------|-----------|-----------|---------|
| BDH | 64 | kernels=(3,7,15), dilations=(1,2,4), 3 residual blocks | 0.30 |
| LSTM | 64 | 2 layers | 0.25 |
| TFT | 64 | 2 LSTM layers, 4 attention heads | 0.25 |

---

## 4. Results — Original Training Data

{comparison_table(orig_d, 'Original')}

### Per-model detail
{"".join(model_section(m, orig_metrics.get(m,{})) for m in models if orig_metrics.get(m))}

### Plots
- `documentation/training_curves_v3_orig.png` — Train/Val loss convergence
- `documentation/pr_roc_curves_v3_orig.png`   — Precision-Recall and ROC curves
- `documentation/confusion_matrices_v3_orig.png` — Confusion matrices at optimal threshold

**Winner (Original):** {best_orig} (PR-AUC={fmt((orig_metrics.get(best_orig) or {}).get('pr_auc'))})

---

## 5. Results — Augmented Training Data

{comparison_table(aug_d, 'Augmented')}

### Per-model detail
{"".join(model_section(m, aug_metrics.get(m,{})) for m in models if aug_metrics.get(m))}

### Plots
- `documentation/training_curves_v3_aug.png` — Train/Val loss convergence
- `documentation/pr_roc_curves_v3_aug.png`   — Precision-Recall and ROC curves
- `documentation/confusion_matrices_v3_aug.png` — Confusion matrices at optimal threshold

**Winner (Augmented):** {best_aug} (PR-AUC={fmt((aug_metrics.get(best_aug) or {}).get('pr_auc'))})

---

## 6. Cross-Experiment Comparison

| Model | Orig PR-AUC | Aug PR-AUC | Delta | Orig Recall | Aug Recall |
|-------|------------|-----------|-------|------------|-----------|
{"".join(
    f"| {m} | {fmt((orig_metrics.get(m) or {}).get('pr_auc'))} | {fmt((aug_metrics.get(m) or {}).get('pr_auc'))} | {((aug_metrics.get(m) or {}).get('pr_auc',0) - (orig_metrics.get(m) or {}).get('pr_auc',0)):+.4f} | {fmt(((orig_metrics.get(m) or {}).get('optimal_f1_threshold') or {}).get('recall'))} | {fmt(((aug_metrics.get(m) or {}).get('optimal_f1_threshold') or {}).get('recall'))} |\n"
    for m in models
)}
---

## 7. Important Observations

1. **Dataset size severely limits conclusions.** With only 1 independent test flare event (3 sequences), all test metrics carry very high statistical uncertainty. A single correct or incorrect prediction changes PR-AUC by >0.03.

2. **All positive events come from one day, one active region.** The models have never seen a flare from a different active region or at different solar-cycle phase. Generalization is unknown.

3. **Augmentation effect.** Synthetic augmentation changes the pos_weight from ~44.6 to ~9.8. This means the model's decision boundary shifts. Augmentation should improve robustness within the observed distribution, but cannot compensate for lack of real diversity.

4. **BDH vs LSTM interpretation.** If BDH has better PR-AUC, this is consistent with its multi-scale design capturing different pre-flare timescales simultaneously. However, with 3 test positives, this cannot be established as a statistically significant finding.

5. **TFT interpretability.** The Variable Selection Network inside TFT provides, in principle, feature importance weights per timestep. These can be extracted for post-hoc analysis (future work).

---

## 8. Limitations and Future Work

| Limitation | Impact | Mitigation |
|-----------|--------|-----------|
| 7 independent flare events total | Very high metric uncertainty | Collect more SUIT data |
| All events from single day/AR | No generalizability | Multi-AR, multi-day campaigns |
| 39 synthetic = 7 real events | Augmentation doesn't add new diversity | Real new observations needed |
| No temporal gaps between events | Models may memorize AR 3590 state | Cross-AR validation |
| CPU-only training | Slower exploration | GPU or cloud training |
| No hyperparameter optimization | Possibly suboptimal | Grid/Bayesian search with more data |

**Next steps for real solar flare forecasting:**
- Obtain SUIT NB04 observations spanning multiple active regions and months.
- Incorporate SoLEXS X-ray flux data as an additional input modality.
- Use NOAA/GOES near-real-time flare catalog for continuous labeling.
- Validate on completely held-out solar cycle periods.
- Implement proper cross-event validation (leave-one-event-out).

---

*Report generated by `scripts/generate_results_doc.py` from actual training outputs.*
*Do not edit manually — regenerate from new training runs.*
"""
    return md


def main():
    print("Generating results.md from model_comparison_v3.json ...")
    data = load_comparison()
    md   = generate(data)
    out  = DOCS / "results.md"
    out.write_text(md, encoding="utf-8")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()

