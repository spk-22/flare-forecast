"""
scripts/rebuild_step2_build_dataset.py
Rebuild the complete feature table and sequence tensors from scratch
using authentic SUIT 2024 NB04 chromospheric FITS files and NOAA GOES catalogs.
"""

import sys, os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path
import csv, json, re, time
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.stats import skew, kurtosis

ROOT = Path(__file__).resolve().parent.parent
INVENTORY_CSV = ROOT / "data" / "suit2024_nb04_inventory.csv"
V2_FRAMES_CSV = ROOT / "suit2024_frames_features_v2.csv"
OUTPUT_FRAMES_V4 = ROOT / "data" / "suit2024_frames_features_v4.csv"
GOES_2024_CSV = ROOT / "labels" / "goes_xrs_flare_report_2024.csv"
GOES_2026_CSV = ROOT / "labels" / "goes_xrs_flare_report_2026.csv"
SEQUENCES_DIR = ROOT / "Sequences"
OUTPUT_SEQ_CSV = ROOT / "data" / "suit2024_sequences_labelled_v4.csv"
OUTPUT_X_NPY = SEQUENCES_DIR / "suit2024_X_features_v4.npy"
OUTPUT_Y_NPY = SEQUENCES_DIR / "suit2024_y_labels_v4.npy"
OUTPUT_MEMBERS_CSV = ROOT / "data" / "suit2024_sequence_members_v4.csv"

FEATURE_COLS = [
    'mean_dn', 'std_dn', 'min_dn', 'max_dn',
    'p01_dn', 'p10_dn', 'p50_dn', 'p90_dn', 'p95_dn', 'p99_dn', 'p999_dn',
    'skew_dn', 'kurt_dn', 'enhancement_ratio', 'bright_pix_frac',
    'quality_percent', 'exposure_s', 'hpc_tx_center_arcsec', 'hpc_ty_center_arcsec'
]

def parse_time(value):
    val_str = str(value).replace('Z', '')
    if '.' in val_str:
        prefix, fraction = val_str.split('.', 1)
        val_str = prefix + '.' + fraction[:6]
    return datetime.fromisoformat(val_str)

def header_value(header, *keys, default=''):
    for key in keys:
        if key in header:
            val = header.get(key)
            if val not in (None, '', 'NA', 'NaN'):
                return val
    return default

def number(value, default=np.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def extract_single_fits_features(fpath):
    try:
        with fits.open(fpath, memmap=False) as hdul:
            header = hdul[0].header
            data = hdul[0].data
            
            if data is not None:
                image = np.asarray(data, dtype=np.float32)
                good = image[np.isfinite(image)]
            else:
                good = np.array([], dtype=np.float32)
                
            shape_y = header.get('NAXIS2', 0)
            shape_x = header.get('NAXIS1', 0)
            
            if good.size > 0:
                mean_val = float(np.mean(good))
                std_val = float(np.std(good))
                min_val = float(np.min(good))
                max_val = float(np.max(good))
                p01, p10, p50, p90, p95, p99, p999 = np.percentile(good, [1, 10, 50, 90, 95, 99, 99.9])
                skew_val = float(skew(good)) if good.size > 10 else 0.0
                kurt_val = float(kurtosis(good)) if good.size > 10 else 0.0
                enhancement_ratio = float((p99 - p50) / (std_val + 1e-5))
                bright_pix_frac = float(np.mean(good > (p50 + 2 * std_val)))
            else:
                mean_val = std_val = min_val = max_val = p01 = p10 = p50 = p90 = p95 = p99 = p999 = np.nan
                skew_val = kurt_val = enhancement_ratio = bright_pix_frac = np.nan
                
            crpix1 = number(header_value(header, 'CRPIX1'), default=0.0)
            crpix2 = number(header_value(header, 'CRPIX2'), default=0.0)
            cdelt1 = number(header_value(header, 'CDELT1', 'CDELTA1'), default=0.698)
            cdelt2 = number(header_value(header, 'CDELT2', 'CDELTA2'), default=0.698)
            crota2 = number(header_value(header, 'CROTA2'), default=0.0)
            
            px_c = (shape_x + 1.0) / 2.0
            py_c = (shape_y + 1.0) / 2.0
            rad = np.radians(crota2)
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            dx = px_c - crpix1
            dy = py_c - crpix2
            tx_center = float((dx * cos_a - dy * sin_a) * cdelt1)
            ty_center = float((dx * sin_a + dy * cos_a) * cdelt2)
            
            date_obs_raw = header_value(header, 'DATE-OBS', 'DATE_OBS')
            ts_utc = parse_time(date_obs_raw).isoformat(sep=' ')
            
            product = re.search(r'_(\d{4})[NB|BB].*\.fits$', fpath.name)
            product_code = product.group(1) if product else ''
            
            is_full_disk = int(shape_y >= 2048 and shape_x >= 2048)
            
            return {
                'frame_id': fpath.stem,
                'path': str(fpath.relative_to(ROOT)).replace('\\', '/') if ROOT in fpath.parents else str(fpath),
                'filename': fpath.name,
                'timestamp_utc': ts_utc,
                'filter': header_value(header, 'FTR_NAME', 'FILTER'),
                'obs_id': header_value(header, 'OBS_ID'),
                'roi_id': str(header_value(header, 'ROI_ID')),
                'product_code': product_code,
                'shape_y': int(shape_y),
                'shape_x': int(shape_x),
                'x1': header_value(header, 'X1'),
                'x2': header_value(header, 'X2'),
                'y1': header_value(header, 'Y1'),
                'y2': header_value(header, 'Y2'),
                'is_full_disk': is_full_disk,
                'quality_percent': number(header_value(header, 'QVAL'), default=100.0),
                'exposure_s': number(header_value(header, 'EXPTIME', 'EXPOSURE'), default=np.nan),
                'ctype1': header_value(header, 'CTYPE1'),
                'ctype2': header_value(header, 'CTYPE2'),
                'cunit1': header_value(header, 'CUNIT1'),
                'cunit2': header_value(header, 'CUNIT2'),
                'crpix1': crpix1,
                'crpix2': crpix2,
                'crval1': number(header_value(header, 'CRVAL1'), default=0.0),
                'crval2': number(header_value(header, 'CRVAL2'), default=0.0),
                'cdelt1_arcsec': cdelt1,
                'cdelt2_arcsec': cdelt2,
                'crota2_deg': crota2,
                'hpc_tx_center_arcsec': round(tx_center, 3),
                'hpc_ty_center_arcsec': round(ty_center, 3),
                'fov_x_arcsec': round(shape_x * cdelt1, 3),
                'fov_y_arcsec': round(shape_y * cdelt2, 3),
                'rsun_obs_arcsec': number(header_value(header, 'RSUN_OBS'), default=np.nan),
                'dsun_obs_m': number(header_value(header, 'DSUN_OBS'), default=np.nan),
                'hglon_obs_deg': number(header_value(header, 'HGLN_OBS'), default=np.nan),
                'hglat_obs_deg': number(header_value(header, 'HGLT_OBS'), default=np.nan),
                'mean_dn': round(mean_val, 4),
                'std_dn': round(std_val, 4),
                'min_dn': round(min_val, 4),
                'max_dn': round(max_val, 4),
                'p01_dn': round(float(p01), 4),
                'p10_dn': round(float(p10), 4),
                'p50_dn': round(float(p50), 4),
                'p90_dn': round(float(p90), 4),
                'p95_dn': round(float(p95), 4),
                'p99_dn': round(float(p99), 4),
                'p999_dn': round(float(p999), 4),
                'skew_dn': round(skew_val, 4),
                'kurt_dn': round(kurt_val, 4),
                'enhancement_ratio': round(enhancement_ratio, 4),
                'bright_pix_frac': round(bright_pix_frac, 6),
                'nan_count': int(image.size - good.size) if data is not None else 0,
                'has_basic_wcs': int(all(k in header for k in ('CTYPE1','CTYPE2','CRVAL1','CRVAL2','CRPIX1','CRPIX2')))
            }
    except Exception as e:
        print(f"Error processing {fpath.name}: {e}")
        return None

def main():
    print("=== Step 2: Rebuilding Solar Flare Features and Sequences ===")
    t_script_start = time.time()
    
    # 1. Load Inventory
    df_inv = pd.read_csv(INVENTORY_CSV)
    print(f"Loaded {len(df_inv)} NB04 observations from inventory.")
    
    # Check existing features in v2
    existing_records = []
    already_filenames = set()
    if V2_FRAMES_CSV.exists():
        df_v2 = pd.read_csv(V2_FRAMES_CSV)
        df_v2_nb04 = df_v2[df_v2['filter'] == 'NB04'].copy()
        already_filenames = set(df_v2_nb04['filename'])
        existing_records = df_v2_nb04.to_dict('records')
        print(f"Reusing {len(existing_records)} already-extracted NB04 records from v2.")
        
    # Find remaining to extract
    needed_df = df_inv[~df_inv['filename'].isin(already_filenames)].copy()
    print(f"Extracting features for {len(needed_df)} newly scanned FITS files...")
    
    needed_paths = [Path(p) for p in needed_df['filepath']]
    new_records = []
    
    if len(needed_paths) > 0:
        with ThreadPoolExecutor(max_workers=12) as executor:
            for idx, rec in enumerate(executor.map(extract_single_fits_features, needed_paths)):
                if rec is not None:
                    new_records.append(rec)
                if (idx + 1) % 500 == 0 or (idx + 1) == len(needed_paths):
                    print(f"  Processed {idx + 1}/{len(needed_paths)} files...")
                    
    all_records = existing_records + new_records
    df_features = pd.DataFrame(all_records)
    df_features['timestamp_utc'] = df_features['timestamp_utc'].astype(str)
    df_features['dt'] = pd.to_datetime(df_features['timestamp_utc'])
    df_features = df_features.sort_values('dt').reset_index(drop=True)
    
    # Fill any NA in numeric columns with 0.0 or median
    for col in FEATURE_COLS:
        if col in df_features.columns:
            df_features[col] = pd.to_numeric(df_features[col], errors='coerce').fillna(0.0)
            
    df_features.to_csv(OUTPUT_FRAMES_V4, index=False)
    print(f"Saved {len(df_features)} frame feature records to {OUTPUT_FRAMES_V4.name}")
    
    # 2. Load GOES Catalogues
    g24 = pd.read_csv(GOES_2024_CSV)
    g26 = pd.read_csv(GOES_2026_CSV)
    goes_all = pd.concat([g24, g26], ignore_index=True)
    goes_all['start_dt'] = pd.to_datetime(goes_all['start_time'])
    goes_all['end_dt'] = pd.to_datetime(goes_all['end_time'])
    goes_all['peak_dt'] = pd.to_datetime(goes_all['time'])
    print(f"Loaded {len(goes_all)} GOES solar flares from 2024 and 2026 catalogues.")
    
    # 3. Spatial Grouping & Sequence Construction
    # Track grouping: (obs_date, dim_str, track_roi)
    df_features['dim_str'] = df_features['shape_x'].astype(str) + 'x' + df_features['shape_y'].astype(str)
    df_features['track_roi'] = df_features.apply(
        lambda r: 'FULL_DISK' if (r['shape_x'] == 2048 and r['shape_y'] == 2048) else str(r['roi_id']),
        axis=1
    )
    
    TARGET_SEQ_LEN = 80
    MIN_FRAMES_IN_TRACK = 35
    MAX_HISTORY_HOURS = 3.0
    
    sequences = []
    feature_matrices = []
    members = []
    pos_count = 0
    neg_count = 0
    
    print("\n--- Constructing Temporal Sequences (Target Length = 80 timesteps) ---")
    track_groups = df_features.groupby(['obs_id', 'dim_str', 'track_roi'], sort=False)
    
    for (obs_id, dim, roi), grp in track_groups:
        grp = grp.sort_values('dt').reset_index(drop=True)
        times = grp['dt'].tolist()
        records = grp.to_dict('records')
        feat_vals = grp[FEATURE_COLS].values
        n_frames = len(grp)
        
        if n_frames < MIN_FRAMES_IN_TRACK:
            continue
            
        track_name = f"{obs_id}_{dim}_ROI{roi}"
        
        for target_idx in range(MIN_FRAMES_IN_TRACK, n_frames):
            t0 = times[target_idx - 1]
            target_time = times[target_idx]
            delta_s = (target_time - t0).total_seconds()
            
            if delta_s > 180.0:
                continue
                
            start_idx = max(0, target_idx - TARGET_SEQ_LEN)
            t_start = times[start_idx]
            dur_min = (t0 - t_start).total_seconds() / 60.0
            
            if (dur_min / 60.0) > MAX_HISTORY_HOURS:
                continue
                
            raw_window_feats = feat_vals[start_idx:target_idx] # shape: (actual_len, 19)
            actual_len = len(raw_window_feats)
            
            # Pad on left with earliest frame if actual_len < 80
            if actual_len < TARGET_SEQ_LEN:
                pad_size = TARGET_SEQ_LEN - actual_len
                pad_slice = np.repeat(raw_window_feats[0:1], pad_size, axis=0)
                final_feat_matrix = np.concatenate([pad_slice, raw_window_feats], axis=0)
            else:
                final_feat_matrix = raw_window_feats
                
            # Check flare onset in prediction horizon (t0, target_time + 15s]
            hits = goes_all[(goes_all['start_dt'] > t0) & (goes_all['start_dt'] <= target_time + timedelta(seconds=15))]
            is_positive = int(len(hits) > 0)
            fl_ev = hits.iloc[0] if is_positive else None
            
            if is_positive:
                pos_count += 1
            else:
                neg_count += 1
                
            seq_id = f"{track_name}_{start_idx:05d}_{target_idx:05d}"
            
            seq_meta = {
                'sequence_id': seq_id,
                'label': is_positive,
                'target_flare_class': fl_ev['flare_class'] if is_positive else '',
                'target_flare_id': fl_ev['flare_id'] if is_positive else '',
                'target_flare_start_utc': fl_ev['start_time'] if is_positive else '',
                'target_flare_peak_utc': fl_ev['time'] if is_positive else '',
                'target_flare_ar': fl_ev['active_region'] if is_positive else '',
                'obs_id': obs_id,
                'spatial_group': track_name,
                'dim_str': dim,
                'roi_id': roi,
                'start_utc': times[start_idx].isoformat(),
                't0_utc': t0.isoformat(),
                'target_utc': target_time.isoformat(),
                'horizon_seconds': round(delta_s, 2),
                'history_duration_min': round(dur_min, 2),
                'actual_frames_used': actual_len,
                'padded_frames': TARGET_SEQ_LEN - actual_len,
                't0_mean_dn': round(float(records[target_idx - 1]['mean_dn']), 4),
                't0_enhancement_ratio': round(float(records[target_idx - 1]['enhancement_ratio']), 4),
                't0_bright_pix_frac': round(float(records[target_idx - 1]['bright_pix_frac']), 6),
                't0_hpc_tx': round(float(records[target_idx - 1]['hpc_tx_center_arcsec']), 2),
                't0_hpc_ty': round(float(records[target_idx - 1]['hpc_ty_center_arcsec']), 2)
            }
            sequences.append(seq_meta)
            feature_matrices.append(final_feat_matrix)
            
    df_seq = pd.DataFrame(sequences)
    X_arr = np.array(feature_matrices, dtype=np.float32)
    y_arr = df_seq['label'].values.astype(np.int64)
    
    print("\n--- Summary of Rebuilt Dataset ---")
    print(f"Total sequences: {len(df_seq)}")
    print(f"Positive pre-flare sequences: {pos_count}")
    print(f"Negative non-flare sequences: {neg_count}")
    print(f"Feature tensor shape: {X_arr.shape}")
    print(f"Labels array shape: {y_arr.shape}")
    
    pos_flares = df_seq[df_seq['label'] == 1]
    print(f"Unique flares captured: {pos_flares['target_flare_start_utc'].nunique()}")
    print("Flare class distribution:")
    print(pos_flares['target_flare_class'].apply(lambda x: x[0] if isinstance(x, str) and len(x) > 0 else 'Unknown').value_counts().to_string())
    
    # Save outputs
    SEQUENCES_DIR.mkdir(exist_ok=True)
    df_seq.to_csv(OUTPUT_SEQ_CSV, index=False)
    np.save(OUTPUT_X_NPY, X_arr)
    np.save(OUTPUT_Y_NPY, y_arr)
    
    t_end = time.time()
    print(f"\nSaved sequences to {OUTPUT_SEQ_CSV}")
    print(f"Saved X tensor to {OUTPUT_X_NPY}")
    print(f"Saved y array to {OUTPUT_Y_NPY}")
    print(f"Rebuild completed successfully in {t_end - t_script_start:.2f} seconds.")

if __name__ == '__main__':
    main()
