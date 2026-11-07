"""Test proper HPC coordinate un-wrapping."""
import numpy as np
from pathlib import Path
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'suit2024'
df_inv = pd.read_csv(ROOT / 'all_fits_inventory.csv')

def parse_wcs_solar_bbox(fpath):
    with fits.open(fpath, memmap=False) as hdul:
        h = hdul[0].header
        ny, nx = h.get('NAXIS2', 0), h.get('NAXIS1', 0)
        w = WCS(h)
        
        pix_corners = np.array([
            [0, 0],
            [nx - 1, 0],
            [nx - 1, ny - 1],
            [0, ny - 1],
            [(nx - 1) / 2.0, (ny - 1) / 2.0]
        ])
        
        world = w.all_pix2world(pix_corners, 0)
        cunit1 = str(h.get('CUNIT1', 'arcsec')).lower()
        cunit2 = str(h.get('CUNIT2', 'arcsec')).lower()
        
        tx_deg = world[:, 0]
        ty_deg = world[:, 1]
        
        # Unwrap angles: if in degrees, unwrap around [-180, 180]
        tx_deg_unwrapped = (tx_deg + 180.0) % 360.0 - 180.0
        ty_deg_unwrapped = (ty_deg + 180.0) % 360.0 - 180.0
        
        # Convert to arcseconds
        tx_arcsec = tx_deg_unwrapped * 3600.0 if 'deg' in cunit1 else world[:, 0]
        ty_arcsec = ty_deg_unwrapped * 3600.0 if 'deg' in cunit2 else world[:, 1]
        
        return {
            'tx_center_arcsec': round(float(tx_arcsec[4]), 2),
            'ty_center_arcsec': round(float(ty_arcsec[4]), 2),
            'tx_min_arcsec': round(float(np.min(tx_arcsec[:4])), 2),
            'tx_max_arcsec': round(float(np.max(tx_arcsec[:4])), 2),
            'ty_min_arcsec': round(float(np.min(ty_arcsec[:4])), 2),
            'ty_max_arcsec': round(float(np.max(ty_arcsec[:4])), 2),
            'fov_x_arcsec': round(float(np.max(tx_arcsec[:4]) - np.min(tx_arcsec[:4])), 2),
            'fov_y_arcsec': round(float(np.max(ty_arcsec[:4]) - np.min(ty_arcsec[:4])), 2),
            'x1': h.get('X1'), 'x2': h.get('X2'), 'y1': h.get('Y1'), 'y2': h.get('Y2'),
            'crpix1': h.get('CRPIX1'), 'crpix2': h.get('CRPIX2'),
            'crval1': h.get('CRVAL1'), 'crval2': h.get('CRVAL2'),
            'cdelt1': h.get('CDELT1', h.get('CDELTA1')), 'cdelt2': h.get('CDELT2', h.get('CDELTA2')),
            'crota2': h.get('CROTA2')
        }

records = []
for obs_id in df_inv['obs_id'].unique():
    subset = df_inv[df_inv['obs_id'] == obs_id]
    for (roi_id, shape, ftr), group in subset.groupby(['roi_id', 'shape', 'filter']):
        fname = group.iloc[0]['filename']
        res = parse_wcs_solar_bbox(DATA_DIR / fname)
        res['obs_id'] = obs_id
        res['roi_id'] = roi_id
        res['shape'] = shape
        res['filter'] = ftr
        res['count'] = len(group)
        records.append(res)

df_wcs = pd.DataFrame(records)
print(df_wcs[['obs_id', 'roi_id', 'filter', 'shape', 'count', 'tx_center_arcsec', 'ty_center_arcsec', 'fov_x_arcsec', 'fov_y_arcsec']])
df_wcs.to_csv(ROOT / 'wcs_unwrapped_summary.csv', index=False)
