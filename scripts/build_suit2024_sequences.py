"""Construct temporal sequences from complete SUIT 2024 dataset with WCS grouping and flare labeling."""
from pathlib import Path
import csv
import json
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FRAMES_CSV = ROOT / 'suit2024_frames_features.csv'
GOES_2024_CSV = ROOT / 'labels' / 'goes_xrs_flare_report_2024.csv'
GOES_2026_CSV = ROOT / 'labels' / 'goes_xrs_flare_report_2026.csv'
SEQUENCES_DIR = ROOT / 'Sequences'

def parse_time(value):
    val_str = str(value).replace('Z', '')
    if '.' in val_str:
        prefix, fraction = val_str.split('.', 1)
        val_str = prefix + '.' + fraction[:6]
    return datetime.fromisoformat(val_str)

def main():
    SEQUENCES_DIR.mkdir(exist_ok=True)
    
    # Load frame features
    df_frames = pd.read_csv(FRAMES_CSV)
    print(f"Loaded {len(df_frames)} frames from {FRAMES_CSV}")
    
    # Filter to NB04 channel
    df_nb04 = df_frames[df_frames['filter'] == 'NB04'].copy()
    df_nb04['dt'] = pd.to_datetime(df_nb04['timestamp_utc'])
    df_nb04 = df_nb04.sort_values('dt').reset_index(drop=True)
    print(f"Total NB04 frames: {len(df_nb04)}")
    
    # Load flare catalogs
    goes_2024 = pd.read_csv(GOES_2024_CSV)
    goes_2026 = pd.read_csv(GOES_2026_CSV)
    goes_all = pd.concat([goes_2024, goes_2026], ignore_index=True)
    goes_all['start_dt'] = pd.to_datetime(goes_all['start_time'])
    goes_all['end_dt'] = pd.to_datetime(goes_all['end_time'])
    
    # Spatial grouping:
    # Group by (obs_id, roi_id, shape_y, shape_x)
    tracks = defaultdict(list)
    for _, row in df_nb04.iterrows():
        track_key = (
            str(row['obs_id']),
            str(row['roi_id']),
            int(row['shape_y']),
            int(row['shape_x'])
        )
        tracks[track_key].append(row.to_dict())
        
    print(f"Identified {len(tracks)} spatial tracks in NB04.")
    
    HISTORY_WINDOW_FRAMES = 80 # ~2 hours at nominal 90s cadence
    MAX_HISTORY_HOURS = 3.0
    
    sequences = []
    members = []
    
    feature_cols = [
        'mean_dn', 'std_dn', 'min_dn', 'max_dn',
        'p01_dn', 'p10_dn', 'p50_dn', 'p90_dn', 'p95_dn', 'p99_dn', 'p999_dn',
        'skew_dn', 'kurt_dn', 'enhancement_ratio', 'bright_pix_frac',
        'quality_percent', 'exposure_s', 'hpc_tx_center_arcsec', 'hpc_ty_center_arcsec'
    ]
    
    feature_matrices = []
    sequence_labels = []
    sequence_ids = []
    
    for (obs_id, roi_id, shape_y, shape_x), track in tracks.items():
        track.sort(key=lambda r: r['timestamp_utc'])
        times = [parse_time(r['timestamp_utc']) for r in track]
        n_frames = len(track)
        
        tx_c = float(np.mean([r['hpc_tx_center_arcsec'] for r in track]))
        ty_c = float(np.mean([r['hpc_ty_center_arcsec'] for r in track]))
        
        # We need at least HISTORY_WINDOW_FRAMES + 1 target frame
        if n_frames < HISTORY_WINDOW_FRAMES + 1:
            print(f"Skipping track {obs_id}_ROI{roi_id}_{shape_y}x{shape_x}: {n_frames} frames (need >= {HISTORY_WINDOW_FRAMES + 1})")
            continue
            
        print(f"Processing track {obs_id}_ROI{roi_id}_{shape_y}x{shape_x} ({n_frames} frames)...")
        
        for end_idx in range(HISTORY_WINDOW_FRAMES - 1, n_frames - 1):
            start_idx = end_idx - (HISTORY_WINDOW_FRAMES - 1)
            target_idx = end_idx + 1
            
            window = track[start_idx:end_idx + 1]
            window_times = times[start_idx:end_idx + 1]
            
            t_start = window_times[0]
            t0 = window_times[-1]
            target_frame = track[target_idx]
            target_time = times[target_idx]
            
            history_span_min = (t0 - t_start).total_seconds() / 60.0
            history_span_hrs = history_span_min / 60.0
            target_delta_s = (target_time - t0).total_seconds()
            
            # Check maximum history cap (<= 3 hours)
            if history_span_hrs > MAX_HISTORY_HOURS:
                continue
                
            # Inter-frame gaps within historical window
            gaps = [(r - l).total_seconds() for l, r in zip(window_times[:-1], window_times[1:])]
            max_gap = max(gaps) if gaps else 0.0
            
            # Target observation check:
            # Flare onset occurs in the target observation window (t0, target_time]
            # M or X class flares
            mx_events = goes_all[
                (goes_all['flare_class'].str.startswith(('M', 'X'), na=False)) &
                (goes_all['start_dt'] > t0) &
                (goes_all['start_dt'] <= target_time + timedelta(seconds=15))
            ]
            
            # All qualifying flares (C/M/X)
            all_events = goes_all[
                (goes_all['start_dt'] > t0) &
                (goes_all['start_dt'] <= target_time + timedelta(seconds=15))
            ]
            
            is_positive = int(len(mx_events) > 0)
            fl_ev = mx_events.iloc[0] if len(mx_events) > 0 else (all_events.iloc[0] if len(all_events) > 0 else None)
            
            seq_id = f"{obs_id}_ROI{roi_id}_{shape_y}x{shape_x}_{start_idx:05d}_{end_idx:05d}"
            
            seq_meta = {
                'sequence_id': seq_id,
                'label': is_positive,
                'label_definition': '1 = GOES M/X-class solar flare onset in next observation window; 0 = no M/X onset',
                'target_flare_class': fl_ev['flare_class'] if fl_ev is not None else '',
                'target_flare_id': fl_ev['flare_id'] if fl_ev is not None else '',
                'target_flare_start_utc': fl_ev['start_time'] if fl_ev is not None else '',
                'target_flare_ar': fl_ev['active_region'] if fl_ev is not None else '',
                'obs_id': obs_id,
                'roi_id': roi_id,
                'filter': 'NB04',
                'shape_y': shape_y,
                'shape_x': shape_x,
                'is_full_disk': int(shape_y >= 2048 and shape_x >= 2048),
                'hpc_tx_center_arcsec': tx_c,
                'hpc_ty_center_arcsec': ty_c,
                'frame_count': HISTORY_WINDOW_FRAMES,
                'start_utc': t_start.isoformat(sep=' '),
                'cutoff_t0_utc': t0.isoformat(sep=' '),
                'target_time_utc': target_time.isoformat(sep=' '),
                'history_duration_minutes': round(history_span_min, 3),
                'target_horizon_seconds': round(target_delta_s, 2),
                'max_internal_gap_seconds': round(max_gap, 2),
                'target_frame_id': target_frame['frame_id'],
                'target_frame_path': target_frame['path']
            }
            sequences.append(seq_meta)
            
            # Member records
            for order, frame in enumerate(window):
                members.append({
                    'sequence_id': seq_id,
                    'frame_order': order,
                    'frame_id': frame['frame_id'],
                    'timestamp_utc': frame['timestamp_utc'],
                    'path': frame['path']
                })
                
            # Extract feature matrix (T x D)
            mat = np.zeros((HISTORY_WINDOW_FRAMES, len(feature_cols)), dtype=np.float32)
            for f_idx, frame in enumerate(window):
                for c_idx, c_name in enumerate(feature_cols):
                    val = frame.get(c_name, 0.0)
                    mat[f_idx, c_idx] = float(val) if pd.notnull(val) else 0.0
                    
            feature_matrices.append(mat)
            sequence_labels.append(is_positive)
            sequence_ids.append(seq_id)
            
    df_seq = pd.DataFrame(sequences)
    df_mem = pd.DataFrame(members)
    
    # Save CSVs in Sequences folder and root
    df_seq.to_csv(SEQUENCES_DIR / 'suit2024_sequences_metadata.csv', index=False)
    df_mem.to_csv(SEQUENCES_DIR / 'suit2024_sequence_members.csv', index=False)
    
    # Also update root CSVs for project compatibility
    df_seq.to_csv(ROOT / 'suit2024_sequences_labelled.csv', index=False)
    df_mem.to_csv(ROOT / 'suit2024_sequence_members.csv', index=False)
    
    # Save feature matrices as .npz
    X = np.stack(feature_matrices, axis=0) # shape (N, T, D)
    y = np.array(sequence_labels, dtype=np.int64)
    seq_ids_arr = np.array(sequence_ids)
    
    np.savez_compressed(
        SEQUENCES_DIR / 'suit2024_sequences_features.npz',
        X=X,
        y=y,
        sequence_ids=seq_ids_arr,
        feature_cols=np.array(feature_cols)
    )
    
    pos_count = int(y.sum())
    neg_count = int(len(y) - pos_count)
    pos_pct = (pos_count / len(y)) * 100 if len(y) > 0 else 0.0
    neg_pct = (neg_count / len(y)) * 100 if len(y) > 0 else 0.0
    
    print("\n================ Sequence Construction Summary ================")
    print(f"Total valid sequences: {len(df_seq)}")
    print(f"Positive samples (flare = 1): {pos_count} ({pos_pct:.2f}%)")
    print(f"Negative samples (flare = 0): {neg_count} ({neg_pct:.2f}%)")
    print(f"Feature tensor shape: {X.shape}")
    print(f"Features: {feature_cols}")
    print(f"Saved metadata to {SEQUENCES_DIR / 'suit2024_sequences_metadata.csv'}")
    print(f"Saved members to {SEQUENCES_DIR / 'suit2024_sequence_members.csv'}")
    print(f"Saved feature tensor to {SEQUENCES_DIR / 'suit2024_sequences_features.npz'}")

if __name__ == '__main__':
    main()
