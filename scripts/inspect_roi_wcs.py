"""Examine ROI geometry and WCS coordinate calculations across files."""
import pandas as pd
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'suit2024'

df_inv = pd.read_csv(ROOT / 'all_fits_inventory.csv')

print("Sample headers inspection across groups:")
for obs_id in df_inv['obs_id'].unique():
    subset = df_inv[df_inv['obs_id'] == obs_id]
    for (roi_id, shape, ftr), group in subset.groupby(['roi_id', 'shape', 'filter']):
        fname = group.iloc[0]['filename']
        with fits.open(DATA_DIR / fname) as hdul:
            h = hdul[0].header
            print(f"--- OBS: {obs_id} | ROI: {roi_id} | Filter: {ftr} | Shape: {shape} ({len(group)} files) ---")
            print(f"    DATE-OBS: {h.get('DATE-OBS')}")
            print(f"    X1: {h.get('X1')}, X2: {h.get('X2')}, Y1: {h.get('Y1')}, Y2: {h.get('Y2')}")
            print(f"    CRVAL1: {h.get('CRVAL1')}, CRVAL2: {h.get('CRVAL2')}")
            print(f"    CRPIX1: {h.get('CRPIX1')}, CRPIX2: {h.get('CRPIX2')}")
            print(f"    CDELT1: {h.get('CDELT1', h.get('CDELTA1'))}, CDELT2: {h.get('CDELT2', h.get('CDELTA2'))}")
            print(f"    CROTA2: {h.get('CROTA2')}, CTYPE1: {h.get('CTYPE1')}, CTYPE2: {h.get('CTYPE2')}")
            print(f"    RSUN_OBS: {h.get('RSUN_OBS')}, DSUN_OBS: {h.get('DSUN_OBS')}")
            print(f"    HGLN_OBS: {h.get('HGLN_OBS')}, HGLT_OBS: {h.get('HGLT_OBS')}")
            print(f"    EXPTIME: {h.get('EXPTIME')}, QVAL: {h.get('QVAL')}")
