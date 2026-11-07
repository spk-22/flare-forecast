"""Inspect all FITS files in suit2024 and produce detailed breakdown."""
from pathlib import Path
from collections import Counter
import pandas as pd
from astropy.io import fits
import json

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'suit2024'
fits_files = sorted(DATA_DIR.glob('*.fits'))

records = []
errors = []

for idx, p in enumerate(fits_files):
    try:
        with fits.open(p, memmap=False) as hdul:
            h = hdul[0].header
            records.append({
                'filename': p.name,
                'filter': h.get('FTR_NAME', h.get('FILTER', 'UNKNOWN')),
                'obs_id': str(h.get('OBS_ID', 'UNKNOWN')),
                'roi_id': str(h.get('ROI_ID', 'UNKNOWN')),
                'date_obs': str(h.get('DATE-OBS', h.get('DATE_OBS', 'UNKNOWN'))),
                'shape': f"{h.get('NAXIS2',0)}x{h.get('NAXIS1',0)}",
                'naxis1': h.get('NAXIS1', 0),
                'naxis2': h.get('NAXIS2', 0),
                'qval': h.get('QVAL', 'UNKNOWN'),
                'exptime': h.get('EXPTIME', h.get('EXPOSURE', None)),
                'crval1': h.get('CRVAL1', None),
                'crval2': h.get('CRVAL2', None),
                'crpix1': h.get('CRPIX1', None),
                'crpix2': h.get('CRPIX2', None),
                'cdelt1': h.get('CDELT1', h.get('CDELTA1', None)),
                'cdelt2': h.get('CDELT2', h.get('CDELTA2', None)),
                'crota2': h.get('CROTA2', None),
                'x1': h.get('X1', None),
                'x2': h.get('X2', None),
                'y1': h.get('Y1', None),
                'y2': h.get('Y2', None),
                'rsun_obs': h.get('RSUN_OBS', None),
                'dsun_obs': h.get('DSUN_OBS', None),
                'hgln_obs': h.get('HGLN_OBS', None),
                'hglt_obs': h.get('HGLT_OBS', None)
            })
    except Exception as e:
        errors.append((p.name, str(e)))

df = pd.DataFrame(records)
print(f"Total FITS files parsed: {len(df)}, Errors: {len(errors)}")
print("\n--- Filter Distribution ---")
print(df['filter'].value_counts())

print("\n--- OBS_ID breakdown ---")
print(df.groupby(['obs_id', 'filter', 'roi_id', 'shape'])['date_obs'].agg(['count', 'min', 'max']))

df.to_csv(ROOT / 'all_fits_inventory.csv', index=False)
print("\nSaved all_fits_inventory.csv")
