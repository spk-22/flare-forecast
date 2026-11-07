import pandas as pd
from datetime import datetime, timedelta

df = pd.read_csv('all_fits_inventory_v2.csv')
df['dt'] = pd.to_datetime(df['date_obs'])
df = df.sort_values('dt').reset_index(drop=True)

g24 = pd.read_csv('labels/goes_xrs_flare_report_2024.csv')
g26 = pd.read_csv('labels/goes_xrs_flare_report_2026.csv')
goes = pd.concat([g24, g26], ignore_index=True)
goes['start_dt'] = pd.to_datetime(goes['start_time'])
goes['end_dt'] = pd.to_datetime(goes['end_time'])
goes['peak_dt'] = pd.to_datetime(goes['time'])

df_nb04 = df[df['filter'] == 'NB04'].copy()

# Look at 2048x2048 in T24_0503_000268
fd_2048 = df_nb04[(df_nb04['obs_id'] == 'T24_0503_000268') & (df_nb04['ny'] == 2048)].sort_values('dt').reset_index(drop=True)
print(f"Full-Disk 2048x2048 in T24: {len(fd_2048)} frames from {fd_2048['dt'].min()} to {fd_2048['dt'].max()}")

# Look at 704x704 AR 3590 in T24_0503_000268
roi_704 = df_nb04[(df_nb04['obs_id'] == 'T24_0503_000268') & (df_nb04['ny'] == 704)].sort_values('dt').reset_index(drop=True)
print(f"AR 3590 704x704 in T24: {len(roi_704)} frames from {roi_704['dt'].min()} to {roi_704['dt'].max()}")

def count_flares_in_stream(stream, name, max_hist_min=180.0, min_hist_min=90.0):
    times = stream['dt'].tolist()
    sequences = []
    for end_idx in range(len(stream) - 1):
        t0 = times[end_idx]
        target_time = times[end_idx + 1]
        
        # Check window: between 2 and 3 hours of history (or fixed ~80 frames = ~95 min)
        # Let's test fixed 80 frames:
        if end_idx >= 79:
            start_idx = end_idx - 79
            t_start = times[start_idx]
            dur_m = (t0 - t_start).total_seconds() / 60.0
            if dur_m <= max_hist_min:
                # Check GOES flare onsets in target window (t0, target_time + 15s]
                # Test M/X
                mx_hits = goes[
                    (goes['flare_class'].str.startswith(('M', 'X'), na=False)) &
                    (goes['start_dt'] > t0) &
                    (goes['start_dt'] <= target_time + timedelta(seconds=15))
                ]
                
                # Test All (C/M/X)
                all_hits = goes[
                    (goes['start_dt'] > t0) &
                    (goes['start_dt'] <= target_time + timedelta(seconds=15))
                ]
                
                sequences.append({
                    'stream': name,
                    't0': t0,
                    'target_time': target_time,
                    'dur_m': dur_m,
                    'mx_count': len(mx_hits),
                    'all_count': len(all_hits),
                    'mx_flares': [f"{r['flare_class']}@{r['start_time']}" for _, r in mx_hits.iterrows()],
                    'all_flares': [f"{r['flare_class']}@{r['start_time']}" for _, r in all_hits.iterrows()]
                })
    df_seq = pd.DataFrame(sequences)
    print(f"\nStream '{name}':")
    print(f"  Total valid sequences: {len(df_seq)}")
    print(f"  M/X flare onset positive: {(df_seq['mx_count'] > 0).sum()}")
    print(f"  All (C/M/X) flare onset positive: {(df_seq['all_count'] > 0).sum()}")
    if (df_seq['all_count'] > 0).sum() > 0:
        for _, r in df_seq[df_seq['all_count'] > 0].iterrows():
            print(f"    -> {r['target_time']} | flares: {r['all_flares']}")
    return df_seq

s1 = count_flares_in_stream(fd_2048, "Full-Disk 2048x2048")
s2 = count_flares_in_stream(roi_704, "AR 3590 704x704")
