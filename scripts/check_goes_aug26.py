"""Analyze GOES flares on 2026-08-26 and compare against the SUIT observation windows."""
import pandas as pd

df = pd.read_csv('labels/goes_xrs_flare_report_2026.csv')
df['start_dt'] = pd.to_datetime(df['start_time'])
df['end_dt'] = pd.to_datetime(df['end_time'])

aug26 = df[(df['start_dt'] >= '2026-08-26 00:00:00') & (df['start_dt'] <= '2026-08-26 23:59:59')].sort_values('start_dt')
print(f"Total GOES flares on 2026-08-26: {len(aug26)}")
for _, r in aug26.iterrows():
    print(f"Flare ID: {r['flare_id']} | Class: {r['flare_class']} | Start: {r['start_time']} | Peak: {r['time']} | End: {r['end_time']} | AR: {r['active_region']} | Loc: ({r['flare_loc_xrs_hpc_x']}, {r['flare_loc_xrs_hpc_y']})")
