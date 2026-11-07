"""Calculate accurate solar coordinates (Helioprojective Tx, Ty arcsec) for each FITS image."""
from pathlib import Path
import pandas as pd
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'suit2024'
df_inv = pd.read_csv(ROOT / 'all_fits_inventory.csv')

def compute_solar_bbox(row):
    fpath = DATA_DIR / row['filename']
    with fits.open(fpath, memmap=False) as hdul:
        h = hdul[0].header
        ny, nx = h.get('NAXIS2', 0), h.get('NAXIS1', 0)
        
        # We can use WCS directly
        # If CDELTA1 is present instead of CDELT1, astropy handles or we fix
        w = WCS(h)
        
        # Calculate corners in pixel coordinates (0-indexed: [0,0], [0,ny], [nx,0], [nx,ny], [nx/2, ny/2])
        pix_coords = np.array([
            [0, 0],
            [nx - 1, 0],
            [nx - 1, ny - 1],
            [0, ny - 1],
            [(nx - 1) / 2.0, (ny - 1) / 2.0]
        ])
        
        # World coordinates
        world_coords = w.all_pix2world(pix_coords, 0)
        cunit1 = str(h.get('CUNIT1', 'arcsec')).lower()
        cunit2 = str(h.get('CUNIT2', 'arcsec')).lower()
        
        # Scale to arcsec if in deg
        scale_x = 3600.0 if 'deg' in cunit1 else 1.0
        scale_y = 3600.0 if 'deg' in cunit2 else 1.0
        
        tx_vals = world_coords[:, 0] * scale_x
        ty_vals = world_coords[:, 1] * scale_y
        
        return {
            'tx_center': float(tx_vals[4]),
            'ty_center': float(ty_vals[4]),
            'tx_min': float(np.min(tx_vals[:4])),
            'tx_max': float(np.max(tx_vals[:4])),
            'ty_min': float(np.min(ty_vals[:4])),
            'ty_max': float(np.max(ty_vals[:4])),
            'fov_x_arcsec': float(np.max(tx_vals[:4]) - np.min(tx_vals[:4])),
            'fov_y_arcsec': float(np.max(ty_vals[:4]) - np.min(ty_vals[:4])),
            'crpix1': float(h.get('CRPIX1', 0)),
            'crpix2': float(h.get('CRPIX2', 0)),
            'crval1': float(h.get('CRVAL1', 0)),
            'crval2': float(h.get('CRVAL2', 0)),
            'cdelt1': float(h.get('CDELT1', h.get('CDELTA1', 0))),
            'cdelt2': float(h.get('CDELT2', h.get('CDELTA2', 0))),
            'crota2': float(h.get('CROTA2', 0))
        }

print("Testing WCS on sample files...")
test_results = []
for obs_id in df_inv['obs_id'].unique():
    subset = df_inv[df_inv['obs_id'] == obs_id]
    for (roi_id, shape, ftr), group in subset.groupby(['roi_id', 'shape', 'filter']):
        res = compute_solar_bbox(group.iloc[0])
        res['obs_id'] = obs_id
        res['roi_id'] = roi_id
        res['shape'] = shape
        res['filter'] = ftr
        res['count'] = len(group)
        test_results.append(res)

df_test = pd.DataFrame(test_results)
print(df_test[['obs_id', 'roi_id', 'filter', 'shape', 'count', 'tx_center', 'ty_center', 'fov_x_arcsec', 'fov_y_arcsec']])
