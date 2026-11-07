"""Inspect detailed WCS, coordinate systems, and spatial groupings across the dataset."""
from pathlib import Path
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'suit2024'
df_inv = pd.read_csv(ROOT / 'all_fits_inventory.csv')

print(f"Total files: {len(df_inv)}")
print("\nUnique filters:", df_inv['filter'].unique())

# Detailed WCS check
sample_records = []
for obs_id in df_inv['obs_id'].unique():
    subset = df_inv[df_inv['obs_id'] == obs_id]
    for (roi_id, shape, ftr), group in subset.groupby(['roi_id', 'shape', 'filter']):
        fpath = DATA_DIR / group.iloc[0]['filename']
        with fits.open(fpath, memmap=False) as hdul:
            h = hdul[0].header
            w = WCS(h)
            # Center of the image in pixel coordinates
            ny, nx = h.get('NAXIS2', 0), h.get('NAXIS1', 0)
            center_x, center_y = nx / 2.0, ny / 2.0
            
            # World coordinates at center and corners
            try:
                center_world = w.pixel_to_world(center_x, center_y)
                center_coord_str = f"({center_world.Tx.arcsec:.1f}\", {center_world.Ty.arcsec:.1f}\")" if hasattr(center_world, 'Tx') else str(center_world)
            except Exception as e:
                center_coord_str = f"WCS err: {e}"
                
            sample_records.append({
                'obs_id': obs_id,
                'roi_id': roi_id,
                'filter': ftr,
                'shape': shape,
                'count': len(group),
                'min_time': group['date_obs'].min(),
                'max_time': group['date_obs'].max(),
                'crval1': h.get('CRVAL1'),
                'crval2': h.get('CRVAL2'),
                'crpix1': h.get('CRPIX1'),
                'crpix2': h.get('CRPIX2'),
                'cdelt1': h.get('CDELT1', h.get('CDELTA1')),
                'cdelt2': h.get('CDELT2', h.get('CDELTA2')),
                'ctype1': h.get('CTYPE1'),
                'ctype2': h.get('CTYPE2'),
                'crota2': h.get('CROTA2'),
                'center_coord': center_coord_str,
                'x1': h.get('X1'), 'x2': h.get('X2'),
                'y1': h.get('Y1'), 'y2': h.get('Y2'),
                'rsun_obs': h.get('RSUN_OBS'),
                'hgln_obs': h.get('HGLN_OBS'),
                'hglt_obs': h.get('HGLT_OBS')
            })

df_samples = pd.DataFrame(sample_records)
print(df_samples[['obs_id', 'roi_id', 'filter', 'shape', 'count', 'min_time', 'center_coord', 'crval1', 'crval2', 'cdelt1']])
df_samples.to_csv(ROOT / 'spatial_wcs_summary.csv', index=False)
