"""
augment_v3.py — Statistically sound synthetic augmentation of positive training samples.

Strategy: Multivariate Gaussian perturbation on the full (80, 17) sequence.
  - Computes empirical covariance from the 11 training positive sequences.
  - Applies conservative σ_scale=0.05 (5% of per-feature std) perturbation.
  - Enforces physical constraints: percentile ordering, [0,1] clipping for fractions.
  - Groups augmentation by flare event (not sequence) to prevent leakage.
  - Augments ONLY training positives — test set untouched.

Outputs:
  Sequences/suit2024_X_aug_v3.npy   — augmented training X (orig_train + synthetic)
  Sequences/suit2024_y_aug_v3.npy   — corresponding labels
  Sequences/suit2024_aug_meta_v3.csv — which samples are synthetic + parent info

Reproducibility: SEED = 42, sigma_scale = 0.05
"""

from pathlib import Path
import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────────────────────
SEED        = 42
SIGMA_SCALE = 0.05        # 5% of per-feature std — conservative
SPLIT_IDX   = 502         # chronological split (unchanged)
TARGET_POS  = 50          # desired total positives in training
ROOT        = Path(__file__).resolve().parent.parent
SEQ_DIR     = ROOT / "Sequences"

rng = np.random.default_rng(SEED)

FEAT_NAMES = [
    'mean_dn', 'std_dn', 'min_dn', 'max_dn',
    'p01_dn', 'p10_dn', 'p50_dn', 'p90_dn', 'p95_dn', 'p99_dn', 'p999_dn',
    'skew_dn', 'kurt_dn', 'enhancement_ratio', 'bright_pix_frac',
    'quality_percent', 'exposure_s',
    'hpc_tx_center_arcsec', 'hpc_ty_center_arcsec',
]

# Percentile feature indices (after original 19 features, pre-drop)
# These must remain sorted: p01 <= p10 <= p50 <= p90 <= p95 <= p99 <= p999
PERCENTILE_COLS_ORIG = [4, 5, 6, 7, 8, 9, 10]  # p01..p999 in original 19-feat space

# WCS coordinate indices (original 19-feat): tx=17, ty=18
WCS_COLS_ORIG = [17, 18]

# bright_pix_frac index (original 19-feat): 14
BPIXFRAC_COL_ORIG = 14


def load_raw():
    """Load the raw 19-feature arrays and drop constant features exactly as train_models_v2 does."""
    X_raw = np.load(SEQ_DIR / "suit2024_X_features_v2.npy")   # (718, 80, 19)
    y_raw = np.load(SEQ_DIR / "suit2024_y_labels_v2.npy")     # (718,)

    # Drop near-zero-variance features using TRAIN stats only (mirror of train_models_v2.py)
    X_tr_raw = X_raw[:SPLIT_IDX]
    per_std  = X_tr_raw.reshape(-1, X_raw.shape[-1]).std(axis=0)
    keep     = per_std > 1e-3
    dropped  = np.where(~keep)[0].tolist()
    print(f"Drop indices {dropped} (zero-variance features: "
          f"{[FEAT_NAMES[i] for i in dropped]})")

    # Map PERCENTILE_COLS to new indices after dropping
    keep_positions = np.where(keep)[0].tolist()
    pct_cols_new   = [keep_positions.index(c) for c in PERCENTILE_COLS_ORIG if c in keep_positions]
    bpf_col_new    = keep_positions.index(BPIXFRAC_COL_ORIG) if BPIXFRAC_COL_ORIG in keep_positions else None
    wcs_cols_new   = [keep_positions.index(c) for c in WCS_COLS_ORIG if c in keep_positions]

    X = X_raw[:, :, keep]
    y = y_raw
    n_feats = X.shape[-1]
    print(f"Active features: {n_feats}/{X_raw.shape[-1]}")
    print(f"Active feature names: {[FEAT_NAMES[i] for i in keep_positions]}")

    return X, y, n_feats, pct_cols_new, bpf_col_new, wcs_cols_new, keep_positions


def identify_flare_events(seq_csv_path):
    """Return a dict mapping flare_onset_time -> list of train sequence indices."""
    df = pd.read_csv(seq_csv_path)
    df_train = df.iloc[:SPLIT_IDX].copy()
    df_train['seq_index'] = range(SPLIT_IDX)

    pos_train = df_train[df_train['label'] == 1].copy()
    events = {}
    for onset, grp in pos_train.groupby('target_flare_start_utc'):
        events[onset] = grp['seq_index'].tolist()

    print(f"\nIndependent training flare events: {len(events)}")
    for onset, idxs in events.items():
        cls = pos_train.loc[pos_train['target_flare_start_utc']==onset, 'target_flare_class'].values[0]
        print(f"  {onset}  class={cls}  seqs={idxs}")
    return events


def enforce_constraints(X_synth, pct_cols, bpf_col, feat_min, feat_max):
    """
    Enforce physical constraints on synthetic samples:
    1. Percentile ordering: p01 <= p10 <= p50 <= p90 <= p95 <= p99 <= p999
    2. bright_pix_frac in [0, 1]
    3. Clip all features to [feat_min - 0.5*range, feat_max + 0.5*range]
    """
    n, T, F = X_synth.shape

    # 1. Clip to extended observed range
    feat_range = feat_max - feat_min
    lo = feat_min - 0.5 * feat_range
    hi = feat_max + 0.5 * feat_range
    X_synth = np.clip(X_synth, lo[None, None, :], hi[None, None, :])

    # 2. Enforce percentile ordering per timestep
    if len(pct_cols) >= 2:
        pct_arr = X_synth[:, :, pct_cols]          # (n, T, 7)
        pct_arr = np.sort(pct_arr, axis=-1)         # sort along feature axis
        X_synth[:, :, pct_cols] = pct_arr

    # 3. bright_pix_frac in [0, 1]
    if bpf_col is not None:
        X_synth[:, :, bpf_col] = np.clip(X_synth[:, :, bpf_col], 0.0, 1.0)

    return X_synth


def augment(X_pos_train, events_map, X_all_train, pct_cols, bpf_col, wcs_cols, n_feats):
    """
    Generate synthetic positive sequences via multivariate Gaussian perturbation.
    
    For each training flare event:
      - Gather the sequences for that event (1–3 sequences).
      - Compute per-timestep, per-feature std from these sequences.
      - Draw scaled Gaussian noise and add to each original sequence to create copies.
      - WCS coordinates get tiny jitter (< 1 arcsec std).
    
    Returns: X_synthetic (n_synth, 80, n_feats), parent_info list
    """
    n_train_pos = X_pos_train.shape[0]   # 11
    n_synth_needed = TARGET_POS - n_train_pos  # 39
    print(f"\nGenerating {n_synth_needed} synthetic samples from {n_train_pos} originals "
          f"across {len(events_map)} flare events...")

    # Per-event allocation (roughly proportional to number of sequences per event)
    event_items = list(events_map.items())
    n_events    = len(event_items)
    base        = n_synth_needed // n_events
    extras      = n_synth_needed % n_events
    per_event   = [base + (1 if i < extras else 0) for i in range(n_events)]

    # Compute per-feature statistics over all training positives (for constraint clipping)
    feat_min = X_pos_train.reshape(-1, n_feats).min(axis=0)
    feat_max = X_pos_train.reshape(-1, n_feats).max(axis=0)
    feat_std = X_pos_train.reshape(-1, n_feats).std(axis=0)
    feat_std = np.where(feat_std < 1e-6, 1.0, feat_std)

    # WCS jitter std: 0.5 arcsec (nearly invisible in the coordinate space)
    WCS_JITTER = 0.5

    synth_X    = []
    parent_info = []

    for i, ((onset, seq_idxs), n_copies) in enumerate(zip(event_items, per_event)):
        # Grab original sequences for this event
        event_seqs = X_all_train[seq_idxs]   # (k, 80, n_feats) where k = 1..3

        # Compute per-timestep per-feature std across the event sequences
        if len(event_seqs) > 1:
            event_std = event_seqs.std(axis=0)  # (80, n_feats)
        else:
            # Only 1 sequence: use global positive std, scaled down
            event_std = np.full((80, n_feats), feat_std[None, :] * 0.5)

        noise_scale = SIGMA_SCALE * np.maximum(event_std, feat_std[None, :] * 0.01)

        for j in range(n_copies):
            # Pick parent: cycle through available sequences for this event
            parent_idx = seq_idxs[j % len(seq_idxs)]
            parent_seq = X_all_train[parent_idx].copy()   # (80, n_feats)

            # Gaussian noise with the event-specific scale
            noise = rng.standard_normal((80, n_feats)) * noise_scale

            # Suppress WCS jitter to <1 arcsec
            if wcs_cols:
                for wc in wcs_cols:
                    noise[:, wc] = rng.standard_normal(80) * WCS_JITTER

            synth_seq = parent_seq + noise
            synth_X.append(synth_seq)
            parent_info.append({
                'parent_seq_index': parent_idx,
                'flare_event_onset': onset,
                'copy_number': j,
                'is_synthetic': True,
            })

    X_synthetic = np.array(synth_X, dtype=np.float32)  # (n_synth, 80, n_feats)
    X_synthetic = enforce_constraints(X_synthetic, pct_cols, bpf_col, feat_min, feat_max)
    print(f"Generated {len(X_synthetic)} synthetic sequences.")
    return X_synthetic, parent_info


def build_augmented_dataset(X, y, X_synthetic, parent_info, n_feats):
    """
    Build the augmented training set:
      Original train  (502 seqs) + synthetic positives (39 seqs)
    
    Returns arrays ready for training, plus metadata CSV.
    """
    X_train_orig = X[:SPLIT_IDX]
    y_train_orig = y[:SPLIT_IDX]
    X_test       = X[SPLIT_IDX:]
    y_test       = y[SPLIT_IDX:]

    n_synth = len(X_synthetic)
    y_synthetic = np.ones(n_synth, dtype=np.float32)

    # Augmented train: original + synthetic positives
    X_aug = np.concatenate([X_train_orig, X_synthetic], axis=0).astype(np.float32)
    y_aug = np.concatenate([y_train_orig, y_synthetic], axis=0).astype(np.float32)

    print(f"\nAugmented training set:")
    print(f"  Original train:  {len(X_train_orig)} seqs ({int(y_train_orig.sum())} pos, {int((y_train_orig==0).sum())} neg)")
    print(f"  Synthetic:       {n_synth} seqs (all positive)")
    print(f"  Total aug train: {len(X_aug)} seqs ({int(y_aug.sum())} pos, {int((y_aug==0).sum())} neg)")
    print(f"  Test (unchanged): {len(X_test)} seqs ({int(y_test.sum())} pos)")

    # Metadata
    meta_rows = []
    for i in range(len(X_train_orig)):
        meta_rows.append({
            'aug_index':          i,
            'is_synthetic':       False,
            'parent_seq_index':   i,
            'flare_event_onset':  '',
            'copy_number':        0,
            'label':              int(y_train_orig[i]),
        })
    for j, info in enumerate(parent_info):
        meta_rows.append({
            'aug_index':         len(X_train_orig) + j,
            'is_synthetic':      True,
            'parent_seq_index':  info['parent_seq_index'],
            'flare_event_onset': info['flare_event_onset'],
            'copy_number':       info['copy_number'],
            'label':             1,
        })
    meta_df = pd.DataFrame(meta_rows)
    return X_aug, y_aug, X_test, y_test, meta_df


def save_outputs(X_aug, y_aug, meta_df):
    SEQ_DIR.mkdir(exist_ok=True)
    np.save(SEQ_DIR / "suit2024_X_aug_v3.npy", X_aug)
    np.save(SEQ_DIR / "suit2024_y_aug_v3.npy", y_aug)
    meta_df.to_csv(SEQ_DIR / "suit2024_aug_meta_v3.csv", index=False)
    print(f"\nSaved:")
    print(f"  {SEQ_DIR / 'suit2024_X_aug_v3.npy'}  shape={X_aug.shape}")
    print(f"  {SEQ_DIR / 'suit2024_y_aug_v3.npy'}  shape={y_aug.shape}")
    print(f"  {SEQ_DIR / 'suit2024_aug_meta_v3.csv'}")


def print_augmentation_report(X_pos_train, X_synthetic, pct_cols, bpf_col, feat_names_active):
    """Compare original positive vs synthetic statistics."""
    print("\n" + "="*72)
    print("AUGMENTATION QUALITY REPORT")
    print("="*72)
    orig = X_pos_train.reshape(-1, X_pos_train.shape[-1])
    synth = X_synthetic.reshape(-1, X_synthetic.shape[-1])
    print(f"{'Feature':<22} {'Orig_mean':>10} {'Synth_mean':>10} {'Orig_std':>10} {'Synth_std':>10}")
    for i, name in enumerate(feat_names_active):
        print(f"{name:<22} {orig[:,i].mean():>10.4f} {synth[:,i].mean():>10.4f} "
              f"{orig[:,i].std():>10.4f} {synth[:,i].std():>10.4f}")
    print("="*72)


def main():
    print("="*60)
    print("SUIT Solar Flare Dataset — Augmentation v3")
    print(f"Seed={SEED}  sigma_scale={SIGMA_SCALE}  target_pos={TARGET_POS}")
    print("="*60)

    X, y, n_feats, pct_cols, bpf_col, wcs_cols, keep_positions = load_raw()
    feat_names_active = [FEAT_NAMES[i] for i in keep_positions]

    events_map = identify_flare_events(ROOT / "suit2024_sequences_labelled_v2.csv")

    X_train_all = X[:SPLIT_IDX]
    y_train_all = y[:SPLIT_IDX]
    pos_train_idxs = np.where(y_train_all == 1)[0]
    X_pos_train = X_train_all[pos_train_idxs]  # (11, 80, 17)
    print(f"\nTraining positives: {len(X_pos_train)}")

    X_synthetic, parent_info = augment(
        X_pos_train, events_map, X_train_all, pct_cols, bpf_col, wcs_cols, n_feats
    )

    print_augmentation_report(X_pos_train, X_synthetic, pct_cols, bpf_col, feat_names_active)

    X_aug, y_aug, X_test, y_test, meta_df = build_augmented_dataset(
        X, y, X_synthetic, parent_info, n_feats
    )

    save_outputs(X_aug, y_aug, meta_df)

    print("\nDone. Augmentation complete.")
    print(f"  IMPORTANT: The {len(X_synthetic)} synthetic samples are variations of")
    print(f"  {len(events_map)} real independent flare events, NOT {len(X_synthetic)} new events.")


if __name__ == "__main__":
    main()

