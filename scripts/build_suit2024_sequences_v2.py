"""Construct temporal sequences from the complete SUIT 2024 dataset (2,652 files) with WCS spatial grouping and flare labeling."""
from pathlib import Path
import csv, json, os, re
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FRAMES_CSV = ROOT / 'suit2024_frames_features_v2.csv'
GOES_2024_CSV = ROOT / 'labels' / 'goes_xrs_flare_report_2024.csv'
GOES_2026_CSV = ROOT / 'labels' / 'goes_xrs_flare_report_2026.csv'
SEQUENCES_DIR = ROOT / 'Sequences'
SEQUENCES_CSV = ROOT / 'suit2024_sequences_labelled_v2.csv'
MEMBERS_CSV = ROOT / 'suit2024_sequence_members_v2.csv'

def parse_time(value):
    val_str = str(value).replace('Z', '')
    if '.' in val_str:
        prefix, fraction = val_str.split('.', 1)
        val_str = prefix + '.' + fraction[:6]
    return datetime.fromisoformat(val_str)

def main():
    SEQUENCES_DIR.mkdir(exist_ok=True)
    
    # 1. Load frame features
    df_frames = pd.read_csv(FRAMES_CSV)
    print(f"Loaded {len(df_frames)} frames from {FRAMES_CSV.name}")
    
    # Filter strictly to target channel: NB04 (Mg II h, 2803 Å)
    df_nb04 = df_frames[df_frames['filter'] == 'NB04'].copy()
    df_nb04['dt'] = pd.to_datetime(df_nb04['timestamp_utc'])
    df_nb04 = df_nb04.sort_values('dt').reset_index(drop=True)
    print(f"Total NB04 frames: {len(df_nb04)}")
    
    # 2. Load GOES flare catalogs
    goes_2024 = pd.read_csv(GOES_2024_CSV)
    goes_2026 = pd.read_csv(GOES_2026_CSV)
    goes_all = pd.concat([goes_2024, goes_2026], ignore_index=True)
    goes_all['start_dt'] = pd.to_datetime(goes_all['start_time'])
    goes_all['end_dt'] = pd.to_datetime(goes_all['end_time'])
    goes_all['peak_dt'] = pd.to_datetime(goes_all['time'])
    print(f"Loaded {len(goes_all)} GOES solar flares from 2024 and 2026 catalogs.")
    
    # 3. Spatial Grouping using WCS and physical solar coordinates:
    # A) Full-Disk Solar Stream (2048x2048): Complete solar disk monitoring
    # B) Active Region AR 3590 High-Resolution ROI Stream (704x704): Continuous chromospheric crop of AR 3590
    # C) Active Region AR 3590 Commissioning ROI 15 Crop (512x572)
    # D) Active Region ROI 77 Crop (560x560)
    # E) Active Region ROI 100 Crop (704x704) from T24_0549_000277
    # F) Full-Disk Stream from T24_0549_000277 (2048x2048)
    # G) Active Region ROI 104 Crop (640x640) from T24_0728_000381
    
    tracks = {}
    
    # Track 1: T24 Full-Disk (2048x2048) - 486 frames
    t24_fd = df_nb04[(df_nb04['obs_id'] == 'T24_0503_000268') & (df_nb04['shape_y'] == 2048)].copy()
    tracks['T24_0503_000268_FullDisk_2048x2048'] = {
        'df': t24_fd,
        'spatial_region': 'Full Solar Disk (Continuous Chromosphere Monitoring)',
        'target_ar': 'AR 3590 (Disk East Limb / Center)',
        'is_full_disk': 1,
        'plate_scale': 1.396
    }
    
    # Track 2: T24 AR 3590 High-Res ROI (704x704) across ROI 17, 19, 100 - 362 frames
    t24_ar = df_nb04[(df_nb04['obs_id'] == 'T24_0503_000268') & (df_nb04['shape_y'] == 704)].copy()
    tracks['T24_0503_000268_AR3590_704x704'] = {
        'df': t24_ar,
        'spatial_region': 'Active Region AR 3590 (High-Resolution ROI Tracking)',
        'target_ar': 'AR 3590',
        'is_full_disk': 0,
        'plate_scale': 0.698
    }
    
    # Track 3: T24 ROI 15 Early Crop (512x572) - 95 frames
    t24_roi15 = df_nb04[(df_nb04['obs_id'] == 'T24_0503_000268') & (df_nb04['roi_id'].astype(str).str.startswith('15')) & (df_nb04['shape_y'] == 512)].copy()
    tracks['T24_0503_000268_ROI15_512x572'] = {
        'df': t24_roi15,
        'spatial_region': 'Active Region AR 3590 (Commissioning Crop)',
        'target_ar': 'AR 3590',
        'is_full_disk': 0,
        'plate_scale': 0.698
    }
    
    # Track 4: T25 ROI 77 Active Region (560x560) - 95 frames
    t25_roi77 = df_nb04[(df_nb04['obs_id'] == 'T25_1126_001252') & (df_nb04['shape_y'] == 560)].copy()
    tracks['T25_1126_001252_ROI77_560x560'] = {
        'df': t25_roi77,
        'spatial_region': 'Active Region Flare Monitoring Track',
        'target_ar': 'Solar Active Region',
        'is_full_disk': 0,
        'plate_scale': 0.698
    }
    
    # Also include the individual ROI track versions for completeness:
    # ROI 17 (704x704): 100 frames
    t24_roi17 = df_nb04[(df_nb04['obs_id'] == 'T24_0503_000268') & (df_nb04['roi_id'] == '17') & (df_nb04['shape_y'] == 704)].copy()
    tracks['T24_0503_000268_ROI17_704x704'] = {
        'df': t24_roi17,
        'spatial_region': 'Active Region AR 3590 (ROI 17 Segment)',
        'target_ar': 'AR 3590',
        'is_full_disk': 0,
        'plate_scale': 0.698
    }
    
    # ROI 19 (704x704): 99 frames
    t24_roi19 = df_nb04[(df_nb04['obs_id'] == 'T24_0503_000268') & (df_nb04['roi_id'] == '19') & (df_nb04['shape_y'] == 704)].copy()
    tracks['T24_0503_000268_ROI19_704x704'] = {
        'df': t24_roi19,
        'spatial_region': 'Active Region AR 3590 (ROI 19 Segment)',
        'target_ar': 'AR 3590',
        'is_full_disk': 0,
        'plate_scale': 0.698
    }
    
    # ROI 100 (704x704): 163 frames
    t24_roi100 = df_nb04[(df_nb04['obs_id'] == 'T24_0503_000268') & (df_nb04['roi_id'] == '100') & (df_nb04['shape_y'] == 704)].copy()
    tracks['T24_0503_000268_ROI100_704x704'] = {
        'df': t24_roi100,
        'spatial_region': 'Active Region AR 3590 (ROI 100 Segment)',
        'target_ar': 'AR 3590',
        'is_full_disk': 0,
        'plate_scale': 0.698
    }

    # Sequence Generation Parameters:
    HISTORY_WINDOW_FRAMES = 80 # ~95.8 min to 119.0 min nominal history (2-3 hour range)
    MAX_HISTORY_HOURS = 3.0    # Strict maximum history cap (180.0 minutes)
    
    feature_cols = [
        'mean_dn', 'std_dn', 'min_dn', 'max_dn',
        'p01_dn', 'p10_dn', 'p50_dn', 'p90_dn', 'p95_dn', 'p99_dn', 'p999_dn',
        'skew_dn', 'kurt_dn', 'enhancement_ratio', 'bright_pix_frac',
        'quality_percent', 'exposure_s', 'hpc_tx_center_arcsec', 'hpc_ty_center_arcsec'
    ]
    
    # We will construct sequences for the primary continuous WCS tracks:
    # 1. Full-Disk 2048x2048 (406 sequences, 7 positive)
    # 2. AR 3590 High-Res 704x704 (282 sequences, 7 positive)
    # 3. ROI 15 (16 sequences, 0 positive)
    # 4. ROI 77 (16 sequences, 0 positive)
    primary_track_keys = [
        'T24_0503_000268_FullDisk_2048x2048',
        'T24_0503_000268_AR3590_704x704',
        'T24_0503_000268_ROI15_512x572',
        'T25_1126_001252_ROI77_560x560'
    ]
    
    sequences = []
    members = []
    feature_matrices = []
    
    print("\n--- Constructing Sequences ---")
    for tkey in primary_track_keys:
        tinfo = tracks[tkey]
        grp = tinfo['df'].sort_values('dt').reset_index(drop=True)
        times = grp['dt'].tolist()
        records = grp.to_dict('records')
        n_frames = len(grp)
        
        tx_c = float(grp['hpc_tx_center_arcsec'].mean())
        ty_c = float(grp['hpc_ty_center_arcsec'].mean())
        
        print(f"Track: {tkey} ({n_frames} frames)...")
        
        for end_idx in range(HISTORY_WINDOW_FRAMES - 1, n_frames - 1):
            start_idx = end_idx - (HISTORY_WINDOW_FRAMES - 1)
            target_idx = end_idx + 1
            
            window = records[start_idx:end_idx + 1]
            window_times = times[start_idx:end_idx + 1]
            
            t_start = window_times[0]
            t0 = window_times[-1]
            target_frame = records[target_idx]
            target_time = times[target_idx]
            
            history_span_min = (t0 - t_start).total_seconds() / 60.0
            history_span_hrs = history_span_min / 60.0
            target_delta_s = (target_time - t0).total_seconds()
            
            # Enforce 3.0-hour maximum history cap
            if history_span_hrs > MAX_HISTORY_HOURS:
                continue
                
            # Internal cadence gaps
            gaps = [(r - l).total_seconds() for l, r in zip(window_times[:-1], window_times[1:])]
            max_gap = max(gaps) if gaps else 0.0
            
            # Target Flare Labeling:
            # Flare onset occurs in the target observation window (t0, target_time + 15s]
            # Checked against NOAA GOES XRS confirmed flares
            fl_hits = goes_all[
                (goes_all['start_dt'] > t0) &
                (goes_all['start_dt'] <= target_time + timedelta(seconds=15))
            ]
            
            is_positive = int(len(fl_hits) > 0)
            fl_ev = fl_hits.iloc[0] if len(fl_hits) > 0 else None
            
            seq_id = f"{tkey}_{start_idx:05d}_{end_idx:05d}"
            
            seq_meta = {
                'sequence_id': seq_id,
                'label': is_positive,
                'label_definition': '1 = Confirmed solar flare onset in next ~90s observation window; 0 = no flare onset',
                'target_flare_class': fl_ev['flare_class'] if fl_ev is not None else '',
                'target_flare_id': fl_ev['flare_id'] if fl_ev is not None else '',
                'target_flare_start_utc': fl_ev['start_time'] if fl_ev is not None else '',
                'target_flare_peak_utc': fl_ev['time'] if fl_ev is not None else '',
                'target_flare_ar': fl_ev['active_region'] if fl_ev is not None else '',
                'filter': 'NB04',
                'spatial_group': tkey,
                'solar_region_observed': tinfo['spatial_region'],
                'target_active_region': tinfo['target_ar'],
                'shape_y': int(target_frame['shape_y']),
                'shape_x': int(target_frame['shape_x']),
                'is_full_disk': tinfo['is_full_disk'],
                'hpc_tx_center_arcsec': tx_c,
                'hpc_ty_center_arcsec': ty_c,
                'frame_count': HISTORY_WINDOW_FRAMES,
                'start_utc': t_start.isoformat(sep=' '),
                'cutoff_t0_utc': t0.isoformat(sep=' '),
                'target_time_utc': target_time.isoformat(sep=' '),
                'history_duration_minutes': round(history_span_min, 3),
                'target_horizon_seconds': round(target_delta_s, 2),
                'max_internal_gap_seconds': round(max_gap, 2),
                'gap_quality_flag': 'continuous' if max_gap <= 150 else 'irregular_cadence',
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

    df_seqs = pd.DataFrame(sequences)
    df_members = pd.DataFrame(members)
    
    # Save CSVs
    df_seqs.to_csv(SEQUENCES_CSV, index=False)
    print(f"Saved {len(df_seqs)} sequences to {SEQUENCES_CSV.name}")
    df_members.to_csv(MEMBERS_CSV, index=False)
    print(f"Saved {len(df_members)} sequence member rows to {MEMBERS_CSV.name}")
    
    # Save matrices tensor
    X = np.stack(feature_matrices, axis=0) # (N, 80, 19)
    y = df_seqs['label'].values.astype(np.int64)
    np.save(SEQUENCES_DIR / 'suit2024_X_features_v2.npy', X)
    np.save(SEQUENCES_DIR / 'suit2024_y_labels_v2.npy', y)
    print(f"Saved feature tensor X: {X.shape} and labels y: {y.shape} to {SEQUENCES_DIR.name}/")
    
    pos_count = int(y.sum())
    neg_count = len(y) - pos_count
    pos_pct = pos_count / len(y) * 100.0
    neg_pct = neg_count / len(y) * 100.0
    
    print("\n=======================================================")
    print("SUIT 2024 Sequence Construction Summary (Version 2)")
    print(f"Dataset:                  Aditya-L1 SUIT Level-1 Archive (2,652 FITS files)")
    print(f"Target filter:            NB04 (Mg II h, 280.3 nm chromospheric channel)")
    print(f"Valid sequences:          {len(y)}")
    print(f"Positive samples:         {pos_count}")
    print(f"Negative samples:         {neg_count}")
    print(f"Positive percentage:      {pos_pct:.2f}%")
    print(f"Negative percentage:      {neg_pct:.2f}%")
    print(f"Typical sequence length:  {HISTORY_WINDOW_FRAMES} frames (~94.2 to 119.0 minutes)")
    print(f"Maximum history:          {df_seqs['history_duration_minutes'].max():.2f} minutes (< 180.0 min)")
    print(f"Target horizon:           {df_seqs['target_horizon_seconds'].min():.1f} to {df_seqs['target_horizon_seconds'].max():.1f} s")
    print("=======================================================")
    
    print("\nPositive Flare Sequences Found:")
    for _, r in df_seqs[df_seqs['label'] == 1].iterrows():
        print(f"  {r['sequence_id']} | Flare: {r['target_flare_class']} at {r['target_flare_start_utc']} (Peak {r['target_flare_peak_utc']}) | Horizon: {r['target_horizon_seconds']}s | Target: {r['target_time_utc']}")

if __name__ == '__main__':
    main()
