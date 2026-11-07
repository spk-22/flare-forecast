"""Test sequence generation under various spatial grouping and window parameters."""
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

# Focus on NB04
df_nb04 = df_inv[df_inv['filter'] == 'NB04'].copy().reset_index(drop=True)

# Define spatial groups:
# 1. By (obs_id, roi_id, shape)
# Let's inspect spatial grouping based on WCS vs ROI/OBS
print("Evaluating candidate spatial tracks in NB04:")
tracks = {}
for (obs_id, roi_id, shape), grp in df_nb04.groupby(['obs_id', 'roi_id', 'shape']):
    grp = grp.sort_values('dt').reset_index(drop=True)
    grp_name = f"{obs_id}_ROI{roi_id}_{shape}"
    tracks[grp_name] = grp
    times = grp['dt'].tolist()
    span_m = (times[-1] - times[0]).total_seconds() / 60.0 if len(times) > 1 else 0
    print(f"Track: {grp_name:35s} | Frames: {len(grp):4d} | Span: {span_m:6.1f} min ({span_m/60:.2f} hrs)")

print("\n--- Flare events in GOES catalog during these periods ---")
for _, ev in goes_all.iterrows():
    st = ev['start_dt']
    # Check if inside any track window
    for tname, grp in tracks.items():
        t0, t1 = grp['dt'].min(), grp['dt'].max()
        if t0 - timedelta(minutes=5) <= st <= t1 + timedelta(minutes=5):
            print(f"Flare {ev['flare_id']} ({ev['flare_class']}) at {st} inside track {tname}")
