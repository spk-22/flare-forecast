"""Verify ROI solar coordinates formula against WCS."""
import pandas as pd
from pathlib import Path
from astropy.io import fits
from astropy.wcs import WCS
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'suit2024'
df_inv = pd.read_csv(ROOT / 'all_fits_inventory.csv')

def get_hpc_center(header):
    nx, ny = header.get('NAXIS1', 0), header.get('NAXIS2', 0)
    crpix1 = float(header.get('CRPIX1', 0))
    crpix2 = float(header.get('CRPIX2', 0))
    cdelt1 = float(header.get('CDELT1', header.get('CDELTA1', 0.698)))
    cdelt2 = float(header.get('CDELT2', header.get('CDELTA2', 0.698)))
    crota2 = float(header.get('CROTA2', 0))
    
    # center pixel (1-based: (nx+1)/2, (ny+1)/2)
    px = (nx + 1) / 2.0
    py = (ny + 1) / 2.0
    
    rad = np.radians(crota2)
    cos_a = np.cos(rad)
    sin_a = np.sin(rad)
    
    dx = px - crpix1
    dy = py - crpix2
    
    # Solar HPC coordinates in arcsec
    tx = (dx * cos_a - dy * sin_a) * cdelt1
    ty = (dx * sin_a + dy * cos_a) * cdelt2
    
    return tx, ty, (nx * cdelt1), (ny * cdelt2)

print("Comparing HPC center calculation across all ROI groups in dataset:")
groups = df_inv.groupby(['obs_id', 'roi_id', 'shape', 'filter'])
summary = []
for (obs_id, roi_id, shape, ftr), grp in groups:
    fname = grp.iloc[0]['filename']
    with fits.open(DATA_DIR / fname) as hdul:
        h = hdul[0].header
        tx, ty, fov_x, fov_y = get_hpc_center(h)
        summary.append({
            'obs_id': obs_id,
            'roi_id': roi_id,
            'shape': shape,
            'filter': ftr,
            'count': len(grp),
            'min_time': grp['date_obs'].min(),
            'max_time': grp['date_obs'].max(),
            'tx_center_arcsec': round(tx, 2),
            'ty_center_arcsec': round(ty, 2),
            'fov_x_arcsec': round(fov_x, 2),
            'fov_y_arcsec': round(fov_y, 2),
            'is_full_disk': 1 if '4096' in shape or '2048' in shape else 0
        })

df_sum = pd.DataFrame(summary)
print(df_sum.to_string())
