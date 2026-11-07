"""Inspect sequence construction, positive samples, and target frames."""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'suit2024'
df_inv = pd.read_csv(ROOT / 'all_fits_inventory.csv')
df_inv['dt'] = pd.to_datetime(df_inv['date_obs'])
df_inv = df_inv.sort_values('dt').reset_index(drop=True)

goes_2024 = pd.read_csv(ROOT / 'labels' / 'goes_xrs_flare_report_2024.csv')
goes_2026 = pd.read_csv(ROOT / 'labels' / 'goes_xrs_flare_report_2026.csv')
goes_all = pd.concat([goes_2024, goes_2026], ignore_index=True)
goes_all['start_dt'] = pd.to_datetime(goes_all['start_time'])
goes_all['end_dt'] = pd.to_datetime(goes_all['end_time'])

df_nb04 = df_inv[df_inv['filter'] == 'NB04'].copy().reset_index(drop=True)

# Build sequences with sliding window of 80 frames (and test variable 2-3h window)
tracks = {}
for (obs_id, roi_id, shape), grp in df_nb04.groupby(['obs_id', 'roi_id', 'shape']):
    grp = grp.sort_values('dt').reset_index(drop=True)
    grp_name = f"{obs_id}_ROI{roi_id}_{shape}"
    tracks[grp_name] = grp

print("Constructing candidate sequences with target frame verification:")
seq_records = []
for tname, grp in tracks.items():
    if len(grp) < 81: # need at least 80 history + 1 target frame
        continue
    times = grp['dt'].tolist()
    files = grp['filename'].tolist()
    
    # For each end index of input sequence: from 79 up to len(grp)-2 (so that end+1 is the target frame)
    for end_idx in range(79, len(grp) - 1):
        start_idx = end_idx - 79
        history_frames = files[start_idx:end_idx + 1]
        history_times = times[start_idx:end_idx + 1]
        
        target_idx = end_idx + 1
        target_frame = files[target_idx]
        target_time = times[target_idx]
        
        t0 = history_times[-1]
        t_start = history_times[0]
        history_duration_min = (t0 - t_start).total_seconds() / 60.0
        target_delta_s = (target_time - t0).total_seconds()
        
        # Check flare onset in (t0, target_time] or (t0, t0 + 90s]
        # Target M or X class flares
        mx_hits = goes_all[
            (goes_all['flare_class'].str.startswith(('M', 'X'), na=False)) &
            (goes_all['start_dt'] > t0) &
            (goes_all['start_dt'] <= target_time)
        ]
        
        # Also check any flare (C/M/X)
        all_hits = goes_all[
            (goes_all['start_dt'] > t0) &
            (goes_all['start_dt'] <= target_time)
        ]
        
        seq_records.append({
            'seq_id': f"{tname}_{start_idx:04d}_{end_idx:04d}",
            'track': tname,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'target_idx': target_idx,
            't_start': t_start,
            't0': t0,
            'target_time': target_time,
            'history_min': round(history_duration_min, 2),
            'target_delta_s': round(target_delta_s, 2),
            'label_mx': int(len(mx_hits) > 0),
            'label_all': int(len(all_hits) > 0),
            'mx_flare_class': mx_hits.iloc[0]['flare_class'] if len(mx_hits) > 0 else '',
            'all_flare_class': all_hits.iloc[0]['flare_class'] if len(all_hits) > 0 else ''
        })

df_seqs = pd.DataFrame(seq_records)
print(f"Total sequences constructed: {len(df_seqs)}")
print(f"Positive M/X samples: {df_seqs['label_mx'].sum()}, Negative M/X: {len(df_seqs) - df_seqs['label_mx'].sum()}")
print(f"Positive Any Flare samples: {df_seqs['label_all'].sum()}, Negative Any: {len(df_seqs) - df_seqs['label_all'].sum()}")

if df_seqs['label_mx'].sum() > 0:
    print("\nM/X Positive sequence details:")
    print(df_seqs[df_seqs['label_mx'] == 1][['seq_id', 't0', 'target_time', 'target_delta_s', 'mx_flare_class']])
