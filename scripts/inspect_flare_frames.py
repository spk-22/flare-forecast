"""Inspect frames around M4.8 and X6.3 flares."""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
df_inv = pd.read_csv(ROOT / 'all_fits_inventory.csv')
df_inv['dt'] = pd.to_datetime(df_inv['date_obs'])

# Look at ROI 100 704x704 frames between 20:20 and 20:40, and 22:00 and 22:15
df_roi100 = df_inv[(df_inv['obs_id']=='T24_0503_000268') & (df_inv['shape']=='704x704') & (df_inv['filter']=='NB04')].sort_values('dt')

print("Frames around M4.8 flare (20:29:00):")
print(df_roi100[(df_roi100['dt'] >= '2024-02-22 20:25:00') & (df_roi100['dt'] <= '2024-02-22 20:35:00')][['filename', 'dt']])

print("\nFrames around X6.3 flare (22:08:00):")
print(df_roi100[(df_roi100['dt'] >= '2024-02-22 22:05:00') & (df_roi100['dt'] <= '2024-02-22 22:12:00')][['filename', 'dt']])
