"""Extract comprehensive FITS header metadata, WCS solar coordinates, and image intensity features for all 1940 files."""
from pathlib import Path
import csv
import json
import re
from datetime import datetime
import numpy as np
import pandas as pd
from astropy.io import fits
from scipy.stats import skew, kurtosis

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'suit2024'
OUTPUT_FRAMES = ROOT / 'suit2024_frames_features.csv'

def parse_time(value):
    val_str = str(value).replace('Z', '')
    if '.' in val_str:
        prefix, fraction = val_str.split('.', 1)
        val_str = prefix + '.' + fraction[:6]
    return datetime.fromisoformat(val_str)

def header_value(header, *keys, default=''):
    for key in keys:
        if key in header:
            val = header.get(key)
            if val not in (None, '', 'NA', 'NaN'):
                return val
    return default

def number(value, default=np.nan):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default

def process_frame(fpath):
    with fits.open(fpath, memmap=False) as hdul:
        header = hdul[0].header
        data = hdul[0].data
        
        # Calculate image features if data is present
        if data is not None:
            image = np.asarray(data, dtype=np.float32)
            good = image[np.isfinite(image)]
        else:
            good = np.array([], dtype=np.float32)
            
        shape_y = header.get('NAXIS2', 0)
        shape_x = header.get('NAXIS1', 0)
        
        if good.size > 0:
            mean_val = float(np.mean(good))
            std_val = float(np.std(good))
            min_val = float(np.min(good))
            max_val = float(np.max(good))
            p01, p10, p50, p90, p95, p99, p999 = np.percentile(good, [1, 10, 50, 90, 95, 99, 99.9])
            skew_val = float(skew(good)) if good.size > 10 else 0.0
            kurt_val = float(kurtosis(good)) if good.size > 10 else 0.0
            enhancement_ratio = float((p99 - p50) / (std_val + 1e-5))
            bright_pix_frac = float(np.mean(good > (p50 + 2 * std_val)))
        else:
            mean_val = std_val = min_val = max_val = p01 = p10 = p50 = p90 = p95 = p99 = p999 = np.nan
            skew_val = kurt_val = enhancement_ratio = bright_pix_frac = np.nan
            
        # Solar HPC coordinates via WCS header fields
        crpix1 = number(header_value(header, 'CRPIX1'), default=0.0)
        crpix2 = number(header_value(header, 'CRPIX2'), default=0.0)
        cdelt1 = number(header_value(header, 'CDELT1', 'CDELTA1'), default=0.698)
        cdelt2 = number(header_value(header, 'CDELT2', 'CDELTA2'), default=0.698)
        crota2 = number(header_value(header, 'CROTA2'), default=0.0)
        
        px_c = (shape_x + 1.0) / 2.0
        py_c = (shape_y + 1.0) / 2.0
        rad = np.radians(crota2)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        dx = px_c - crpix1
        dy = py_c - crpix2
        tx_center = float((dx * cos_a - dy * sin_a) * cdelt1)
        ty_center = float((dx * sin_a + dy * cos_a) * cdelt2)
        
        date_obs_raw = header_value(header, 'DATE-OBS', 'DATE_OBS')
        ts_utc = parse_time(date_obs_raw).isoformat(sep=' ')
        
        product = re.search(r'_(\d{4})[NB|BB].*\.fits$', fpath.name)
        product_code = product.group(1) if product else ''
        
        is_full_disk = int(shape_y >= 2048 and shape_x >= 2048)
        
        return {
            'frame_id': fpath.stem,
            'path': str(fpath.relative_to(ROOT)).replace('\\', '/'),
            'filename': fpath.name,
            'timestamp_utc': ts_utc,
            'filter': header_value(header, 'FTR_NAME', 'FILTER'),
            'obs_id': header_value(header, 'OBS_ID'),
            'roi_id': str(header_value(header, 'ROI_ID')),
            'product_code': product_code,
            'shape_y': int(shape_y),
            'shape_x': int(shape_x),
            'x1': header_value(header, 'X1'),
            'x2': header_value(header, 'X2'),
            'y1': header_value(header, 'Y1'),
            'y2': header_value(header, 'Y2'),
            'is_full_disk': is_full_disk,
            'quality_percent': number(header_value(header, 'QVAL'), default=100.0),
            'exposure_s': number(header_value(header, 'EXPTIME', 'EXPOSURE'), default=np.nan),
            'ctype1': header_value(header, 'CTYPE1'),
            'ctype2': header_value(header, 'CTYPE2'),
            'cunit1': header_value(header, 'CUNIT1'),
            'cunit2': header_value(header, 'CUNIT2'),
            'crpix1': crpix1,
            'crpix2': crpix2,
            'crval1': number(header_value(header, 'CRVAL1'), default=0.0),
            'crval2': number(header_value(header, 'CRVAL2'), default=0.0),
            'cdelt1_arcsec': cdelt1,
            'cdelt2_arcsec': cdelt2,
            'crota2_deg': crota2,
            'hpc_tx_center_arcsec': round(tx_center, 3),
            'hpc_ty_center_arcsec': round(ty_center, 3),
            'fov_x_arcsec': round(shape_x * cdelt1, 3),
            'fov_y_arcsec': round(shape_y * cdelt2, 3),
            'rsun_obs_arcsec': number(header_value(header, 'RSUN_OBS'), default=np.nan),
            'dsun_obs_m': number(header_value(header, 'DSUN_OBS'), default=np.nan),
            'hglon_obs_deg': number(header_value(header, 'HGLN_OBS'), default=np.nan),
            'hglat_obs_deg': number(header_value(header, 'HGLT_OBS'), default=np.nan),
            'mean_dn': round(mean_val, 4),
            'std_dn': round(std_val, 4),
            'min_dn': round(min_val, 4),
            'max_dn': round(max_val, 4),
            'p01_dn': round(float(p01), 4),
            'p10_dn': round(float(p10), 4),
            'p50_dn': round(float(p50), 4),
            'p90_dn': round(float(p90), 4),
            'p95_dn': round(float(p95), 4),
            'p99_dn': round(float(p99), 4),
            'p999_dn': round(float(p999), 4),
            'skew_dn': round(skew_val, 4),
            'kurt_dn': round(kurt_val, 4),
            'enhancement_ratio': round(enhancement_ratio, 4),
            'bright_pix_frac': round(bright_pix_frac, 6),
            'nan_count': int(image.size - good.size) if data is not None else 0,
            'has_basic_wcs': int(all(k in header for k in ('CTYPE1','CTYPE2','CRVAL1','CRVAL2','CRPIX1','CRPIX2')))
        }

def main():
    fits_files = sorted(DATA_DIR.glob('*.fits'))
    print(f"Processing {len(fits_files)} FITS files...")
    
    records = []
    for idx, fpath in enumerate(fits_files):
        if (idx + 1) % 200 == 0 or idx == len(fits_files) - 1:
            print(f"  Processed {idx + 1}/{len(fits_files)} files...")
        rec = process_frame(fpath)
        records.append(rec)
        
    records.sort(key=lambda r: r['timestamp_utc'])
    df_out = pd.DataFrame(records)
    df_out.to_csv(OUTPUT_FRAMES, index=False)
    print(f"Successfully saved {len(df_out)} frame records to {OUTPUT_FRAMES}")

if __name__ == '__main__':
    main()
