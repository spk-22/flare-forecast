import pandas as pd
from datetime import datetime, timedelta

df = pd.read_csv('all_fits_inventory_v2.csv')
df['dt'] = pd.to_datetime(df['date_obs'])

g24 = pd.read_csv('labels/goes_xrs_flare_report_2024.csv')
g24['start_dt'] = pd.to_datetime(g24['start_time'])
g24['end_dt'] = pd.to_datetime(g24['end_time'])
g24['peak_dt'] = pd.to_datetime(g24['time'])

g26 = pd.read_csv('labels/goes_xrs_flare_report_2026.csv')
g26['start_dt'] = pd.to_datetime(g26['start_time'])
g26['end_dt'] = pd.to_datetime(g26['end_time'])
g26['peak_dt'] = pd.to_datetime(g26['time'])

goes = pd.concat([g24, g26], ignore_index=True)

print("=== ALL TRACKS (Grouped by OBS_ID, ROI_ID, Filter, Shape) ===")
tracks_summary = []
for (obs_id, roi_id, ftr, ny, nx), grp in df.groupby(['obs_id', 'roi_id', 'filter', 'ny', 'nx']):
    grp = grp.sort_values('dt').reset_index(drop=True)
    min_t = grp['dt'].min()
    max_t = grp['dt'].max()
    dur_m = (max_t - min_t).total_seconds() / 60.0
    cadences = [(r - l).total_seconds() for l, r in zip(grp['dt'][:-1], grp['dt'][1:])]
    med_cadence = pd.Series(cadences).median() if cadences else 0.0
    
    # Matching flares during this track window
    # 1) Flares whose start time falls inside the track
    flares_during = goes[(goes['start_dt'] >= min_t) & (goes['start_dt'] <= max_t)]
    # 2) M/X flares
    mx_during = flares_during[flares_during['flare_class'].str.startswith(('M', 'X'), na=False)]
    
    tracks_summary.append({
        'obs_id': obs_id,
        'roi_id': str(roi_id),
        'filter': ftr,
        'shape': f"{ny}x{nx}",
        'frames': len(grp),
        'min_t': min_t,
        'max_t': max_t,
        'dur_m': round(dur_m, 1),
        'median_cadence_s': round(med_cadence, 1),
        'flares_count': len(flares_during),
        'mx_count': len(mx_during),
        'flares_list': [f"{r['flare_class']}@{r['start_time'][11:16]}" for _, r in flares_during.iterrows()]
    })

df_tracks = pd.DataFrame(tracks_summary)
for _, r in df_tracks.iterrows():
    if r['frames'] >= 10:
        print(f"OBS: {r['obs_id']:15s} ROI {r['roi_id']:4s} | {r['filter']:5s} | {r['shape']:9s} | {r['frames']:4d} frames | {r['min_t'].strftime('%Y-%m-%d %H:%M')} -> {r['max_t'].strftime('%H:%M')} ({r['dur_m']}m) | Cad: {r['median_cadence_s']}s | Flares: {r['flares_list']}")
