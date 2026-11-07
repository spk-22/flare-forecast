import pandas as pd
from datetime import datetime, timedelta

# Load GOES 2024 and 2026
g24 = pd.read_csv('labels/goes_xrs_flare_report_2024.csv')
g26 = pd.read_csv('labels/goes_xrs_flare_report_2026.csv')

print(f"GOES 2024 flares: {len(g24)}")
print(f"GOES 2026 flares: {len(g26)}")

g24['start_dt'] = pd.to_datetime(g24['start_time'])
g24['end_dt'] = pd.to_datetime(g24['end_time'])
g24['peak_dt'] = pd.to_datetime(g24['time'])

feb22 = g24[(g24['start_dt'] >= '2024-02-22 12:00:00') & (g24['start_dt'] <= '2024-02-23 00:00:00')]
print("\n--- Flares on 2024-02-22 (12:00 to 24:00 UTC) ---")
for _, r in feb22.iterrows():
    print(f"  {r['flare_id']} | Class: {r['flare_class']} | Start: {r['start_time']} | Peak: {r['time']} | End: {r['end_time']} | AR: {r['active_region']}")

feb28 = g24[(g24['start_dt'] >= '2024-02-28 21:00:00') & (g24['start_dt'] <= '2024-02-29 01:00:00')]
print("\n--- Flares on 2024-02-28 ---")
for _, r in feb28.iterrows():
    print(f"  {r['flare_id']} | Class: {r['flare_class']} | Start: {r['start_time']} | Peak: {r['time']}")

may17 = g24[(g24['start_dt'] >= '2024-05-17 08:00:00') & (g24['start_dt'] <= '2024-05-17 11:00:00')]
print("\n--- Flares on 2024-05-17 ---")
for _, r in may17.iterrows():
    print(f"  {r['flare_id']} | Class: {r['flare_class']} | Start: {r['start_time']} | Peak: {r['time']}")

g26['start_dt'] = pd.to_datetime(g26['start_time'])
for d in ['2026-03-29', '2026-09-26', '2026-10-07', '2026-10-08', '2026-10-09', '2026-10-10']:
    sub = g26[g26['start_dt'].dt.strftime('%Y-%m-%d') == d]
    print(f"\n--- Flares on {d} (total {len(sub)}) ---")
    for _, r in sub.iterrows():
        print(f"  {r['flare_id']} | Class: {r['flare_class']} | Start: {r['start_time']} | Peak: {r['time']}")
