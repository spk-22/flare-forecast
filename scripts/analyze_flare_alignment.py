"""Analyze all flares and temporal alignment with SUIT frames."""
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

ROOT = Path(__file__).resolve().parent.parent
df_inv = pd.read_csv(ROOT / 'all_fits_inventory.csv')
goes_2024 = pd.read_csv(ROOT / 'labels' / 'goes_xrs_flare_report_2024.csv')
goes_2026 = pd.read_csv(ROOT / 'labels' / 'goes_xrs_flare_report_2026.csv')
goes_all = pd.concat([goes_2024, goes_2026], ignore_index=True)
goes_all['start_dt'] = pd.to_datetime(goes_all['start_time'])
goes_all['end_dt'] = pd.to_datetime(goes_all['end_time'])

# Sort inventory by observation time
df_inv['dt'] = pd.to_datetime(df_inv['date_obs'])
df_inv = df_inv.sort_values('dt').reset_index(drop=True)

print(f"Total inventory frames: {len(df_inv)}")
print(f"Filter distribution:\n{df_inv['filter'].value_counts()}")

# Filter to NB04
df_nb04 = df_inv[df_inv['filter'] == 'NB04'].copy().reset_index(drop=True)
print(f"\nTotal NB04 frames: {len(df_nb04)}")

# Check each contiguous stream / group
# Group by (obs_id, roi_id, shape)
for (obs_id, roi_id, shape), grp in df_nb04.groupby(['obs_id', 'roi_id', 'shape']):
    grp = grp.sort_values('dt').reset_index(drop=True)
    times = grp['dt'].tolist()
    gaps = [(t2 - t1).total_seconds() for t1, t2 in zip(times[:-1], times[1:])]
    med_cadence = pd.Series(gaps).median() if gaps else 0
    max_gap = max(gaps) if gaps else 0
    span_hours = (times[-1] - times[0]).total_seconds() / 3600.0 if len(times) > 1 else 0
    print(f"\nGroup: OBS={obs_id}, ROI={roi_id}, Shape={shape} | Count={len(grp)} | Span={span_hours:.2f}h | MedCadence={med_cadence:.1f}s | MaxGap={max_gap:.1f}s")
    print(f"   Start: {times[0]} | End: {times[-1]}")
    
    # Check flares during this group's time window
    grp_flares = goes_all[(goes_all['start_dt'] >= times[0] - timedelta(minutes=5)) & (goes_all['start_dt'] <= times[-1] + timedelta(minutes=5))]
    for _, fl in grp_flares.iterrows():
        print(f"   -> Flare {fl['flare_class']} at {fl['start_time']} (AR: {fl['active_region']})")
