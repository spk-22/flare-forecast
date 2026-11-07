"""Print full headers of representative FITS files from 2024, 2025, and 2026."""
from pathlib import Path
import pandas as pd
from astropy.io import fits
import json

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'suit2024'
df_inv = pd.read_csv(ROOT / 'all_fits_inventory.csv')

reps = [
    df_inv[df_inv['obs_id'] == 'T24_0503_000268'].iloc[0]['filename'],
    df_inv[(df_inv['obs_id'] == 'T24_0503_000268') & (df_inv['shape'] == '704x704')].iloc[0]['filename'],
    df_inv[df_inv['obs_id'] == 'T25_1126_001252'].iloc[0]['filename'],
    df_inv[df_inv['obs_id'] == 'T26_1546_002547'].iloc[0]['filename']
]

for r in reps:
    print(f"================ HEADER FOR {r} ================")
    with fits.open(DATA_DIR / r) as hdul:
        h = hdul[0].header
        for k in h.keys():
            print(f"{k:12s} = {h[k]}  /  {h.comments[k]}")
