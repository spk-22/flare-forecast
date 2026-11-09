"""
scripts/audit_sequence_header_flags.py
Extract and audit FITS header flare indicators for each sequence in suit2024:
  - FLR_TRIG (External Flare Trigger time in ms)
  - HL1OSFLG (HEL1OS flare flag)
  - SX1FLG (SoLEXS detector 1 flare flag)
  - SX2FLG (SoLEXS detector 2 flare flag)
  - NRMFLG / NORM_FLR
  - PRMFLG / PROM_FLR
Inspect the last valid FITS frame (t0) for each of the 2,690 sequences,
record their values and combinations, and report positive/negative distributions.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

def normalize_ts(ts_str):
    return str(ts_str).replace('T', ' ').strip()

def parse_num_flag(val):
    if val is None or str(val).strip() in ('NA', 'NaN', '', 'None'):
        return None
    try:
        return float(val)
    except (ValueError, TypeError):
        return None

def run_audit():
    print("Loading sequence definitions and frame feature table...")
    df_seq = pd.read_csv(DATA / "suit2024_sequences_labelled_v4.csv")
    df_frames = pd.read_csv(DATA / "suit2024_frames_features_v4.csv")

    df_frames['norm_ts'] = df_frames['timestamp_utc'].apply(normalize_ts)
    df_seq['norm_t0'] = df_seq['t0_utc'].apply(normalize_ts)

    # Build mapping from norm_ts to frame path
    ts_to_path = dict(zip(df_frames['norm_ts'], df_frames['path']))

    # Cache FITS header indicators for each unique frame path
    unique_paths = df_frames['path'].unique()
    print(f"Auditing FITS headers across {len(unique_paths)} unique frames...")

    header_cache = {}
    sample_comments = {}

    for p in unique_paths:
        fpath = Path(p)
        if not fpath.exists():
            fpath = ROOT / p
        if not fpath.exists():
            # Try suit2024 root
            fpath = ROOT / "suit2024" / Path(p).name

        if not fpath.exists():
            continue

        try:
            with fits.open(fpath, memmap=True) as hdul:
                hdr = hdul[0].header
                flr_trig = hdr.get('FLR_TRIG', None)
                hl1os = hdr.get('HL1OSFLG', None)
                sx1 = hdr.get('SX1FLG', None)
                sx2 = hdr.get('SX2FLG', None)
                nrmflg = hdr.get('NRMFLG', None)
                norm_flr = hdr.get('NORM_FLR', None)
                prmflg = hdr.get('PRMFLG', None)
                prom_flr = hdr.get('PROM_FLR', None)

                if not sample_comments and 'FLR_TRIG' in hdr:
                    sample_comments['FLR_TRIG'] = hdr.comments['FLR_TRIG']
                if 'HL1OSFLG' not in sample_comments and 'HL1OSFLG' in hdr:
                    sample_comments['HL1OSFLG'] = hdr.comments['HL1OSFLG']
                if 'SX1FLG' not in sample_comments and 'SX1FLG' in hdr:
                    sample_comments['SX1FLG'] = hdr.comments['SX1FLG']
                if 'SX2FLG' not in sample_comments and 'SX2FLG' in hdr:
                    sample_comments['SX2FLG'] = hdr.comments['SX2FLG']

                header_cache[str(p)] = {
                    'FLR_TRIG': flr_trig,
                    'HL1OSFLG': hl1os,
                    'SX1FLG': sx1,
                    'SX2FLG': sx2,
                    'NRMFLG': nrmflg,
                    'NORM_FLR': norm_flr,
                    'PRMFLG': prmflg,
                    'PROM_FLR': prom_flr,
                }
        except Exception as e:
            continue

    print(f"Cached header metadata for {len(header_cache)} frames.")
    print("Header Comments Documented in FITS Cards:")
    for k, v in sample_comments.items():
        print(f"  {k}: '{v}'")

    # Match each sequence's last valid frame (t0)
    seq_records = []
    for idx, row in df_seq.iterrows():
        seq_id = row['sequence_id']
        t0 = row['norm_t0']
        frame_path = ts_to_path.get(t0, None)
        meta = header_cache.get(str(frame_path), {}) if frame_path else {}

        flr_trig = meta.get('FLR_TRIG', None)
        hl1os = meta.get('HL1OSFLG', None)
        sx1 = meta.get('SX1FLG', None)
        sx2 = meta.get('SX2FLG', None)
        norm_flr = meta.get('NORM_FLR', None)

        # Investigate triggers:
        # Documented meaning: FLR_TRIG is ms of last telecommand
        # True detector triggers: HL1OSFLG == 1.0 or SX1FLG == 1.0 or SX2FLG == 1.0 or NORM_FLR == 1
        flr_trig_val = parse_num_flag(flr_trig)
        hl1os_val = parse_num_flag(hl1os)
        sx1_val = parse_num_flag(sx1)
        sx2_val = parse_num_flag(sx2)
        norm_flr_val = parse_num_flag(norm_flr)

        trig_raw_positive = (flr_trig_val is not None and flr_trig_val > 0)
        true_detector_trigger = (
            (hl1os_val is not None and hl1os_val == 1.0) or
            (sx1_val is not None and sx1_val == 1.0) or
            (sx2_val is not None and sx2_val == 1.0) or
            (norm_flr_val is not None and norm_flr_val == 1.0)
        )

        seq_records.append({
            'sequence_id': seq_id,
            'existing_label': row['label'],
            'last_frame_path': frame_path,
            'last_frame_t0': row['t0_utc'],
            'FLR_TRIG': flr_trig,
            'HL1OSFLG': hl1os,
            'SX1FLG': sx1,
            'SX2FLG': sx2,
            'NORM_FLR': norm_flr,
            'flr_trig_active_raw': int(trig_raw_positive),
            'true_detector_trigger': int(true_detector_trigger),
        })

    df_out = pd.DataFrame(seq_records)
    out_csv = DATA / "suit2024_sequence_header_flare_audit.csv"
    df_out.to_csv(out_csv, index=False)
    print(f"\nSaved sequence header audit to: {out_csv}")

    # Summary Statistics
    total_seqs = len(df_out)
    print("\n" + "=" * 70)
    print("  SEQUENCE-LEVEL FITS HEADER FLARE INDICATOR AUDIT (N = 2,690)")
    print("=" * 70)

    print("\n1. FLR_TRIG Distribution (Raw Onboard Trigger Timestamp Field):")
    flr_trig_num = pd.to_numeric(df_out['FLR_TRIG'], errors='coerce')
    print(flr_trig_num.describe())
    n_flr_trig_pos = (flr_trig_num > 0).sum()
    n_flr_trig_zero = (flr_trig_num == 0).sum()
    n_flr_trig_na = flr_trig_num.isna().sum()
    print(f"   FLR_TRIG > 0:    {n_flr_trig_pos} ({n_flr_trig_pos / total_seqs * 100:.2f}%)")
    print(f"   FLR_TRIG == 0:   {n_flr_trig_zero} ({n_flr_trig_zero / total_seqs * 100:.2f}%)")
    print(f"   FLR_TRIG is NaN: {n_flr_trig_na} ({n_flr_trig_na / total_seqs * 100:.2f}%)")

    print("\n2. Detector Source Flags (HEL1OS and SoLEXS):")
    for col in ['HL1OSFLG', 'SX1FLG', 'SX2FLG', 'NORM_FLR']:
        val_counts = df_out[col].value_counts(dropna=False).to_dict()
        print(f"   {col}: {val_counts}")

    print("\n3. Observed Header Flag Combinations at Last Sequence Frame:")
    combo_counts = df_out.groupby(['flr_trig_active_raw', 'HL1OSFLG', 'SX1FLG', 'SX2FLG', 'NORM_FLR']).size().reset_index(name='count')
    print(combo_counts.to_string(index=False))

    print("\n4. Comparison with Existing Ground-Truth Sequence Labels:")
    ct = pd.crosstab(df_out['existing_label'], df_out['true_detector_trigger'], margins=True)
    print(ct)
    print("=" * 70)

if __name__ == "__main__":
    run_audit()

