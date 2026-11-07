"""Compute precise solar coordinates and compare against GOES flare locations."""
from pathlib import Path
from astropy.io import fits
import pandas as pd
import numpy as np

extract_dir = Path('d:/flare-forecast/suit_flares_extracted')
df = pd.read_csv('d:/flare-forecast/suit_flares_extracted_summary.csv')

for trig in [23461599, 43877639]:
    grp = df[df['flr_trig'] == trig]
    sample = extract_dir / grp.iloc[0]['filename']
    with fits.open(sample) as hdul:
        h = hdul[0].header
        nx, ny = h.get('NAXIS1'), h.get('NAXIS2')
        crpix1, crpix2 = h.get('CRPIX1'), h.get('CRPIX2')
        cdelt1, cdelt2 = h.get('CDELT1', 0.698), h.get('CDELT2', 0.698)
        crota2 = h.get('CROTA2', 0)
        
        px = (nx + 1) / 2.0
        py = (ny + 1) / 2.0
        rad = np.radians(crota2)
        dx = px - crpix1
        dy = py - crpix2
        tx = (dx * np.cos(rad) - dy * np.sin(rad)) * cdelt1
        ty = (dx * np.sin(rad) + dy * np.cos(rad)) * cdelt2
        print(f"Trigger {trig} ({len(grp)} files): Center = ({tx:.2f}\", {ty:.2f}\") | FOV = ({nx*cdelt1:.1f}\" x {ny*cdelt2:.1f}\")")
        print(f"  First frame: {grp['date_obs'].min()} | Last frame: {grp['date_obs'].max()}")
