"""Find all flares during SUIT observation time windows."""
from pathlib import Path
import pandas as pd
from datetime import datetime, timedelta
import json

ROOT = Path(__file__).resolve().parent.parent
df_inv = pd.read_csv(ROOT / 'all_fits_inventory.csv')

# Load GOES 2024 and 2026
goes_2024 = pd.read_csv(ROOT / 'labels' / 'goes_xrs_flare_report_2024.csv')
goes_2026 = pd.read_csv(ROOT / 'labels' / 'goes_xrs_flare_report_2026.csv')

# Combine GOES reports
goes_all = pd.concat([goes_2024, goes_2026], ignore_index=True)
goes_all['start_dt'] = pd.to_datetime(goes_all['start_time'])
goes_all['end_dt'] = pd.to_datetime(goes_all['end_time'])
if 'peak_time' in goes_all.columns:
    goes_all['peak_dt'] = pd.to_datetime(goes_all['peak_time'])

# Load HEK if any
hek_flares = []
with open(ROOT / 'labels' / 'hek_flares_2026-09-10_to_2026-09-13.json') as f:
    hek_data = json.load(f)
    for res in hek_data.get('result', []):
        st = res.get('event_starttime')
        et = res.get('event_endtime')
        if st and et:
            hek_flares.append({
                'start_dt': pd.to_datetime(st),
                'end_dt': pd.to_datetime(et),
                'flare_class': res.get('fl_goescls', ''),
                'active_region': res.get('ar_noaanum', ''),
                'source': 'HEK'
            })
df_hek = pd.DataFrame(hek_flares)

print("Checking flare activity during each SUIT observation block:")

# Get time range for each obs_id
for obs_id, group in df_inv.groupby('obs_id'):
    min_t = pd.to_datetime(group['date_obs'].min())
    max_t = pd.to_datetime(group['date_obs'].max())
    print(f"\n=======================================================")
    print(f"OBS_ID: {obs_id}")
    print(f"Time span: {min_t} to {max_t} (Duration: {(max_t - min_t).total_seconds()/3600:.2f} hours)")
    print(f"Total files: {len(group)}, NB04 files: {len(group[group['filter']=='NB04'])}")
    
    # Check flares in window [-3 hours before min_t, +3 hours after max_t]
    search_start = min_t - timedelta(hours=3)
    search_end = max_t + timedelta(hours=3)
    
    flares_in_range = goes_all[(goes_all['start_dt'] >= search_start) & (goes_all['start_dt'] <= search_end)]
    print(f"GOES flares in expanded range ({search_start} to {search_end}): {len(flares_in_range)}")
    for _, fl in flares_in_range.iterrows():
        print(f"   Flare {fl['flare_id']} | Class: {fl['flare_class']} | Start: {fl['start_time']} | End: {fl['end_time']} | AR: {fl['active_region']} | Loc: ({fl['flare_loc_xrs_hpc_x']}, {fl['flare_loc_xrs_hpc_y']})")
