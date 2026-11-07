# Extension Plan — Solar Flare Forecasting v3

## Inspection Findings

### Dataset Reality
| Item | Value |
|------|-------|
| Total sequences | 718 (502 train / 216 test) |
| Input shape | (718, 80, 19) → after drop: (718, 80, 17) |
| Total positives | 14 |
| **Independent flare events (unique onset times)** | **7** |
| Train positives | 11 (spanning 6 flare events) |
| Test positives | 3 (X6.3 super-flare, 2 sequences from AR3590 track + 1 full-disk) |
| Dropped features | indices 15 (`quality_percent`=100 always) and 16 (`exposure_s`=0 always) |
| Active features | 17 |

### 7 Independent Flare Events in Dataset
| # | GOES Class | Onset UTC | Train/Test | Seqs |
|---|-----------|-----------|------------|------|
| 1 | C3.4 | 2024-02-22 15:27 | Train | 1 |
| 2 | C9.4 | 2024-02-22 16:17 | Train | 1 |
| 3 | C4.7 | 2024-02-22 17:08 | Train | 3 (2 tracks) |
| 4 | C4.5 | 2024-02-22 17:45 | Train | 2 (2 tracks) |
| 5 | C4.7 | 2024-02-22 18:15 | Train | 2 (2 tracks) |
| 6 | M4.8 | 2024-02-22 20:29 | Train | 2 (2 tracks) |
| 7 | X6.3 | 2024-02-22 22:08 | **Test** | 3 (2 tracks) |

> **Critical finding:** All 7 events are from a single day (2024-02-22), all from AR 3590. The test split contains exactly ONE independent event (X6.3). This severely limits generalizability claims. We must be explicit about this.

### Key Constraints Identified
1. **Train positives = 11 sequences from 6 flare events**; augmentation must be event-level (not sequence-level) to avoid leakage.
2. The two "tracks" (FullDisk 2048×2048 and AR3590 704×704) observe the same physical flare from different FOVs — they are NOT independent events.
3. `quality_percent` and `exposure_s` are always constant → correctly dropped.
4. `hpc_tx_center_arcsec` and `hpc_ty_center_arcsec` encode spatial track identity, not flare precursor signal — must be handled carefully in augmentation.

---

## PART 1 — Augmentation Plan

### True Event Count in Training
- **6 independent training flare events** (C3.4, C9.4, two C4.7, C4.5, M4.8)
- **11 training positive sequences** (some events have 2 sequences from 2 tracks)

### Augmentation Target
- Cannot claim 50 independent events — we have 6.
- **Target: ~39 synthetic sequences** to reach ~50 total training positives.
- Each synthetic sample is clearly tagged `is_synthetic=True`.
- Augmentation is grouped by flare event, then track, to avoid cross-event mixing.

### Method: **Multivariate Gaussian Perturbation on the full 80-frame sequence**

**Why:** 
- We have 11 training positive sequences, each of shape (80, 17). 
- Features are correlated (brightness percentiles are rank-ordered by definition).
- Gaussian noise preserves the temporal structure of each sequence.
- We perturb independently per timestep using the **empirical covariance** computed only from the 11 training positive sequences, scaled by a conservative factor (σ_scale = 0.05 — 5% of the per-feature standard deviation).
- WCS coordinate features (tx, ty) are perturbed by a tiny amount (< 1 arcsec jitter) since they encode FOV identity.

**Constraints enforced:**
- Synthetic values are clipped to [min_seen - 0.5σ, max_seen + 0.5σ] per feature.
- Rank ordering of percentile features (p01 ≤ p10 ≤ p50 ≤ p90 ≤ p95 ≤ p99 ≤ p999) is enforced by sorting after perturbation.
- `bright_pix_frac` is clipped to [0, 1].
- Each synthetic sample references its parent sequence index.

**What we do NOT do:**
- No SMOTE (would interpolate between events, destroying temporal structure).
- No time-warp / DTW (the temporal ordering of a pre-flare sequence is physically meaningful).
- No cross-event mixing.

**Saved as:** `Sequences/suit2024_X_aug_v3.npy`, `Sequences/suit2024_y_aug_v3.npy`, `Sequences/suit2024_aug_meta_v3.csv`

---

## PART 2 — Three-Model Training Plan

### Models
1. **BDH** — existing architecture, no changes to core design; just re-train on augmented data.
2. **LSTM** — existing architecture, re-train on augmented data.
3. **TFT (Temporal Fusion Transformer)** — new addition. Lightweight implementation, not the full 300M-parameter paper version. We build a training-time-reasonable TFT with:
   - Variable selection network (VSN) per timestep
   - LSTM encoder for local temporal processing
   - Multi-head self-attention for long-range patterns
   - Gated residual networks (GRN) throughout
   - Final gate + MLP classifier

### Training Protocol
- **Two experiments**: original data (11 train pos) and augmented data (~50 train pos).
- Chronological split unchanged: idx 0-501 train, 502-717 test.
- Same pos_weight, same seed, same metrics.
- **Validation split**: last 15% of train sequences = sequences 428–501 (for hyperparameter selection; never test).
- Best checkpoint selected by PR-AUC on validation set.
- Final metrics reported on **test set only**.
- Save preprocessing parameters (mu, sigma, keep_mask) for inference.

### Saved Files
- `models/bdh_v3_orig.pt`, `models/bdh_v3_aug.pt`
- `models/lstm_v3_orig.pt`, `models/lstm_v3_aug.pt`
- `models/tft_v3_orig.pt`, `models/tft_v3_aug.pt`
- `models/preprocessing_params_v3.npz` (mu, sigma, keep_mask)
- `models/model_comparison_v3.json`

---

## PART 3 — Documentation

### results.md
Full results doc with actual metrics, confusion matrices, PR/ROC curves.

### model_explanation.md
Plain-language explanation of all three models, what they learn, trade-offs.

---

## PART 4 — Dashboard (Two Pages)

### Technology
- **Single-file HTML** (like existing dashboard.html) with embedded CSS/JS.
- Flask or FastAPI backend for ZIP inference (no frontend framework needed).
- Alternative: purely static page with pre-loaded results + a small Python inference server.

**Decision:** Build a Flask inference server (`app.py`) that:
- Serves the two-page HTML dashboard at `/`
- Accepts POST `/predict` for ZIP upload → runs all 3 models → returns JSON
- Serves static results at `/results`
- Uses saved model checkpoints + saved preprocessing params

### Page 1: Solar Flare Prediction
- ZIP upload widget
- Timeline visualization (3–4 sample frames across the 80-frame window)
- Date/timestamp strip for each displayed frame
- Prediction table (BDH / LSTM / TFT × Prediction / Probability)
- Model agreement indicator
- Validation status

### Page 2: Model Training Results
- Dataset stats (original vs augmented)
- Three-model comparison table
- PR curves, ROC curves, confusion matrices
- Training loss curves
- Per-model detail panel (click to expand)

---

## Implementation Sequence

| Step | Script | Output |
|------|--------|--------|
| 1 | `scripts/augment_v3.py` | `Sequences/suit2024_X_aug_v3.npy` etc. |
| 2 | `scripts/train_models_v3.py` | 6 model checkpoints + comparison JSON |
| 3 | `scripts/generate_plots_v3.py` | All documentation plots |
| 4 | `documentation/results.md` | Full results document |
| 5 | `documentation/model_explanation.md` | Plain-language explanations |
| 6 | `app.py` | Flask inference server |
| 7 | `templates/dashboard.html` | Two-page dashboard |

---

## Honest Limitations (Must Be Documented)

1. **All 7 flare events are from a single day, single active region (AR 3590).** Results cannot be generalized to other active regions or time periods.
2. **The 39 synthetic samples are variations of 6 real events, not 39 new events.** Augmentation increases diversity within events but cannot replace new observations.
3. **Test set contains only 1 independent flare event (X6.3, 3 sequences).** PR-AUC and F1 estimates have very high variance.
4. **TFT is designed for multi-variate forecasting; binary classification on 11 real positives is a severe under-fitting risk.**
5. All results should be treated as **proof-of-concept**, not production-ready solar flare forecasting.
