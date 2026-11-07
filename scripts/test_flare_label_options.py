import pandas as pd
from datetime import datetime, timedelta
import numpy as np

# Load inventory
df = pd.read_csv('all_fits_inventory_v2.csv')
df['dt'] = pd.to_datetime(df['date_obs'])
df = df.sort_values('dt').reset_index(drop=True)

# Load GOES catalogs
g24 = pd.read_csv('labels/goes_xrs_flare_report_2024.csv')
g26 = pd.read_csv('labels/goes_xrs_flare_report_2026.csv')
goes = pd.concat([g24, g26], ignore_index=True)
goes['start_dt'] = pd.to_datetime(goes['start_time'])
goes['end_dt'] = pd.to_datetime(goes['end_time'])
goes['peak_dt'] = pd.to_datetime(goes['time'])

# Filter to NB04
df_nb04 = df[df['filter'] == 'NB04'].copy().reset_index(drop=True)

print(f"Total NB04 frames: {len(df_nb04)}")

# We will test two grouping approaches:
# Grouping A: (obs_id, roi_id, shape) - the previous track grouping
# Grouping B: (obs_id, spatial_cluster, shape) - WCS physical region grouping

# Let's inspect Grouping A first
tracks_A = {}
for (obs_id, roi_id, ny, nx), grp in df_nb04.groupby(['obs_id', 'roi_id', 'ny', 'nx']):
    grp = grp.sort_values('dt').reset_index(drop=True)
    tkey = f"{obs_id}_ROI{roi_id}_{ny}x{nx}"
    tracks_A[tkey] = grp

print(f"\nGrouping A tracks: {len(tracks_A)}")

def test_sequence_construction(tracks, min_hist_min=90.0, max_hist_min=180.0, flare_classes=('C', 'M', 'X')):
    sequences = []
    for tname, grp in tracks.items():
        if len(grp) < 10:
            continue
        times = grp['dt'].tolist()
        n = len(grp)
        
        # Test sliding window based on time or frames
        # Each sequence has history between min_hist_min and max_hist_min minutes
        # In previous code, fixed 80 frames was used (which for ~70s cadence is ~93-95 min)
        # Let's test sliding window with 80 frames first, then variable
        for end_idx in range(len(grp) - 1):
            t0 = times[end_idx]
            target_idx = end_idx + 1
            t_target = times[target_idx]
            delta_target_s = (t_target - t0).total_seconds()
            
            # Find earliest frame in [t0 - max_hist_min, t0 - min_hist_min]
            # If using fixed 80 frames:
            if end_idx >= 79:
                start_idx = end_idx - 79
                t_start = times[start_idx]
                dur_min = (t0 - t_start).total_seconds() / 60.0
                if dur_min <= max_hist_min:
                    # Check flares in target window (t0, t_target]
                    hits = goes[
                        (goes['flare_class'].str.startswith(flare_classes, na=False)) &
                        (goes['start_dt'] > t0) &
                        (goes['start_dt'] <= t_target + timedelta(seconds=15))
                    ]
                    
                    # Also check ongoing flare: flare active at target time
                    ongoing_hits = goes[
                        (goes['flare_class'].str.startswith(flare_classes, na=False)) &
                        (goes['start_dt'] <= t_target) &
                        (goes['end_dt'] >= t_target)
                    ]
                    
                    sequences.append({
                        'track': tname,
                        'seq_id': f"{tname}_{start_idx:04d}_{end_idx:04d}",
                        't_start': t_start,
                        't0': t0,
                        't_target': t_target,
                        'delta_target_s': delta_target_s,
                        'dur_min': dur_min,
                        'onset_hits': len(hits),
                        'onset_flares': [f"{r['flare_class']}@{r['start_time'][11:19]}" for _, r in hits.iterrows()],
                        'ongoing_hits': len(ongoing_hits),
                        'ongoing_flares': [f"{r['flare_class']}@{r['start_time'][11:19]}" for _, r in ongoing_hits.iterrows()]
                    })
    return pd.DataFrame(sequences)

# Test 1: Fixed 80 frames, M/X only (previous)
res1 = test_sequence_construction(tracks_A, flare_classes=('M', 'X'))
print("\n--- Test 1: Fixed 80 frames, M/X only ---")
print(f"Total sequences: {len(res1)}")
print(f"Onset Positive: {(res1['onset_hits'] > 0).sum()}")
print(f"Ongoing Positive: {(res1['ongoing_hits'] > 0).sum()}")

# Test 2: Fixed 80 frames, C/M/X (All confirmed flares)
res2 = test_sequence_construction(tracks_A, flare_classes=('C', 'M', 'X'))
print("\n--- Test 2: Fixed 80 frames, C/M/X ---")
print(f"Total sequences: {len(res2)}")
print(f"Onset Positive: {(res2['onset_hits'] > 0).sum()}")
if (res2['onset_hits'] > 0).sum() > 0:
    pos2 = res2[res2['onset_hits'] > 0]
    for _, r in pos2.iterrows():
        print(f"   {r['seq_id']} | t0: {r['t0']} | target: {r['t_target']} | flares: {r['onset_flares']}")
print(f"Ongoing Positive: {(res2['ongoing_hits'] > 0).sum()}")

# Test 3: What if history window is 2-3 hours (e.g. >= 90 frames or flexible window)?
