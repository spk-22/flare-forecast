"""Inspect all sequences covering the M4.8 and X6.3 flare events."""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
df_inv = pd.read_csv(ROOT / 'all_fits_inventory.csv')
df_inv['dt'] = pd.to_datetime(df_inv['date_obs'])

# 2048x2048 stream
df_2048 = df_inv[(df_inv['obs_id']=='T24_0503_000268') & (df_inv['shape']=='2048x2048') & (df_inv['filter']=='NB04')].sort_values('dt').reset_index(drop=True)
print(f"2048x2048 total frames: {len(df_2048)}")
print(f"Start: {df_2048['dt'].min()} | End: {df_2048['dt'].max()}")

# Find index of frame around 20:28:25
print("Frames in 2048x2048 around 20:29:00:")
print(df_2048[(df_2048['dt'] >= '2024-02-22 20:25:00') & (df_2048['dt'] <= '2024-02-22 20:35:00')][['filename', 'dt', 'roi_id']])

print("\nFrames in 2048x2048 around 22:08:00:")
print(df_2048[(df_2048['dt'] >= '2024-02-22 22:05:00') & (df_2048['dt'] <= '2024-02-22 22:12:00')][['filename', 'dt', 'roi_id']])
