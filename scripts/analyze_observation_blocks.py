"""Analyze the observation blocks, cadence, filter interleaving, and solar pointing on 2026-08-26."""
import pandas as pd
from datetime import datetime
from astropy.io import fits
from pathlib import Path
import numpy as np

extract_dir = Path('d:/flare-forecast/suit_flares_extracted')
df = pd.read_csv('d:/flare-forecast/suit_flares_extracted_summary.csv')
df['dt'] = pd.to_datetime(df['date_obs'])
df = df.sort_values('dt').reset_index(drop=True)

print("Unique OBS_STRT blocks:")
for strt, grp in df.groupby('obs_strt'):
    t_min = grp['dt'].min()
    t_max = grp['dt'].max()
    duration_min = (t_max - t_min).total_seconds() / 60.0
    nb03_count = len(grp[grp['filter'] == 'NB03'])
    nb05_count = len(grp[grp['filter'] == 'NB05'])
    
    # Check WCS pointing
    f_sample = extract_dir / grp.iloc[0]['filename']
    with fits.open(f_sample) as hdul:
        h = hdul[0].header
        crpix1, crpix2 = h.get('CRPIX1', 0), h.get('CRPIX2', 0)
        sun_cx, sun_cy = h.get('SUN_CX', 0), h.get('SUN_CY', 0)
        cdelt1 = h.get('CDELT1', 0.698)
        crota2 = h.get('CROTA2', 0)
        flr_trig = h.get('FLR_TRIG')
        hl1os = h.get('HL1OSFLG')
        flr_ed = h.get('FLR_ED')
    
    print(f"\n=======================================================")
    print(f"OBS_STRT: {strt}")
    print(f"Observation window: {t_min} to {t_max} (Duration: {duration_min:.2f} min)")
    print(f"Total frames: {len(grp)} | NB03 (Mg II k): {nb03_count} | NB05 (283 nm): {nb05_count}")
    print(f"FLR_TRIG: {flr_trig} | HL1OSFLG: {hl1os} | FLR_ED: {flr_ed}")
    print(f"SUN_CX, CY: ({sun_cx}, {sun_cy}) | CRPIX1, 2: ({crpix1}, {crpix2})")
