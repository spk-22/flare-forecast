"""Perform rigorous validation checks on constructed sequences."""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent
SEQ_META_CSV = ROOT / 'Sequences' / 'suit2024_sequences_metadata.csv'
SEQ_MEM_CSV = ROOT / 'Sequences' / 'suit2024_sequence_members.csv'
NPZ_PATH = ROOT / 'Sequences' / 'suit2024_sequences_features.npz'

df_meta = pd.read_csv(SEQ_META_CSV)
df_mem = pd.read_csv(SEQ_MEM_CSV)
npz = np.load(NPZ_PATH)
X, y = npz['X'], npz['y']

print("================ VALIDATION CHECKS ================")
print(f"1. Total sequences: {len(df_meta)}")
print(f"   Feature tensor X shape: {X.shape}, labels y shape: {y.shape}")
assert len(df_meta) == len(X) == len(y), "Mismatch in metadata and tensor lengths!"

print(f"2. Positive samples: {np.sum(y == 1)}, Negative samples: {np.sum(y == 0)}")
print(f"   Positive rate: {np.mean(y == 1)*100:.2f}%")

# Check 1: Chronological ordering within sequences
print("\n3. Checking chronological ordering of member frames...")
chrono_errors = 0
for seq_id, grp in df_mem.groupby('sequence_id'):
    times = pd.to_datetime(grp['timestamp_utc']).tolist()
    if not all(t1 < t2 for t1, t2 in zip(times[:-1], times[1:])):
        chrono_errors += 1
print(f"   Chronological ordering errors: {chrono_errors} (Passed: {chrono_errors == 0})")

# Check 2: Target occurs after final input observation
print("\n4. Checking that target timestamp > cutoff t0 timestamp...")
target_timing_errors = 0
for _, row in df_meta.iterrows():
    t0 = pd.to_datetime(row['cutoff_t0_utc'])
    target_t = pd.to_datetime(row['target_time_utc'])
    if target_t <= t0:
        target_timing_errors += 1
print(f"   Target timing errors: {target_timing_errors} (Passed: {target_timing_errors == 0})")

# Check 3: History span <= 3 hours (180 minutes)
print("\n5. Checking maximum history duration cap (<= 180 min)...")
history_cap_errors = 0
for _, row in df_meta.iterrows():
    if row['history_duration_minutes'] > 180.0:
        history_cap_errors += 1
print(f"   History cap violations: {history_cap_errors} (Max history: {df_meta['history_duration_minutes'].max():.2f} min)")

# Check 4: No target frame inside input sequence members
print("\n6. Checking zero temporal leakage (target frame not in member frames)...")
leak_errors = 0
mem_map = set(zip(df_mem['sequence_id'], df_mem['frame_id']))
for _, row in df_meta.iterrows():
    if (row['sequence_id'], row['target_frame_id']) in mem_map:
        leak_errors += 1
print(f"   Temporal leakage errors: {leak_errors} (Passed: {leak_errors == 0})")

# Check 5: Filter consistency
print("\n7. Checking filter consistency across sequences...")
assert (df_meta['filter'] == 'NB04').all(), "Mixed filters found in NB04 sequence dataset!"
print("   All sequences strictly use NB04 (Mg II h, 2803 Å) filter. Passed.")

print("\n================ EXAMPLE POSITIVE SEQUENCES ================")
pos_samples = df_meta[df_meta['label'] == 1]
for idx, (_, row) in enumerate(pos_samples.iterrows(), 1):
    print(f"\n--- Positive Example #{idx} ---")
    print(f"Sequence ID:             {row['sequence_id']}")
    print(f"Filter:                  {row['filter']}")
    print(f"Obs ID / ROI / Shape:    {row['obs_id']} / ROI {row['roi_id']} / {row['shape_y']}x{row['shape_x']}")
    print(f"Spatial Solar Center:    Tx={row['hpc_tx_center_arcsec']:.2f}\", Ty={row['hpc_ty_center_arcsec']:.2f}\"")
    print(f"Input Sequence Span:     {row['start_utc']} --> {row['cutoff_t0_utc']} ({row['history_duration_minutes']:.2f} min, {row['frame_count']} frames)")
    print(f"Target Timestamp:        {row['target_time_utc']} (Horizon: {row['target_horizon_seconds']:.1f}s)")
    print(f"Target Frame ID:         {row['target_frame_id']}")
    print(f"Target Solar Flare:      Class {row['target_flare_class']} | ID {row['target_flare_id']} | Start: {row['target_flare_start_utc']} | AR: {row['target_flare_ar']}")
    print(f"Label:                   {row['label']}")

print("\n================ EXAMPLE NEGATIVE SEQUENCES ================")
neg_samples = df_meta[df_meta['label'] == 0].sample(2, random_state=42)
for idx, (_, row) in enumerate(neg_samples.iterrows(), 1):
    print(f"\n--- Negative Example #{idx} ---")
    print(f"Sequence ID:             {row['sequence_id']}")
    print(f"Filter:                  {row['filter']}")
    print(f"Obs ID / ROI / Shape:    {row['obs_id']} / ROI {row['roi_id']} / {row['shape_y']}x{row['shape_x']}")
    print(f"Spatial Solar Center:    Tx={row['hpc_tx_center_arcsec']:.2f}\", Ty={row['hpc_ty_center_arcsec']:.2f}\"")
    print(f"Input Sequence Span:     {row['start_utc']} --> {row['cutoff_t0_utc']} ({row['history_duration_minutes']:.2f} min, {row['frame_count']} frames)")
    print(f"Target Timestamp:        {row['target_time_utc']} (Horizon: {row['target_horizon_seconds']:.1f}s)")
    print(f"Target Frame ID:         {row['target_frame_id']}")
    print(f"Label:                   {row['label']}")
