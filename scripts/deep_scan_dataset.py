import os, glob
import pandas as pd
from astropy.io import fits
from datetime import datetime, timedelta

suit_dir = r'D:\flare-forecast\suit2024'
files = glob.glob(os.path.join(suit_dir, '*.fits'))

# Load GOES reports
g24 = pd.read_csv('labels/goes_xrs_flare_report_2024.csv')
g26 = pd.read_csv('labels/goes_xrs_flare_report_2026.csv')
goes = pd.concat([g24, g26], ignore_index=True)
goes['start_dt'] = pd.to_datetime(goes['start_time'])
goes['end_dt'] = pd.to_datetime(goes['end_time'])
goes['peak_dt'] = pd.to_datetime(goes['time'])

# Parse all 2652 headers
headers = []
for f in files:
    h = fits.getheader(f, 0)
    d_str = h.get('DATE-OBS') or h.get('DATE_OBS') or ''
    headers.append({
        'path': f,
        'filename': os.path.basename(f),
        'filter': h.get('FTR_NAME') or h.get('FILTER') or '',
        'obs_id': h.get('OBS_ID', ''),
        'roi_id': str(h.get('ROI_ID', '')),
        'date_obs': d_str,
        'ny': h.get('NAXIS2', 0),
        'nx': h.get('NAXIS1', 0),
        'crval1': h.get('CRVAL1', 0),
        'crval2': h.get('CRVAL2', 0),
        'crpix1': h.get('CRPIX1', 0),
        'crpix2': h.get('CRPIX2', 0),
        'cdelt1': h.get('CDELT1', 0),
        'cdelt2': h.get('CDELT2', 0),
        'crota2': h.get('CROTA2', 0),
        'flr_trig': h.get('FLR_TRIG'),
        'norm_flr': h.get('NORM_FLR'),
    })

df = pd.DataFrame(headers)
df['dt'] = pd.to_datetime(df['date_obs'])
df = df.sort_values('dt').reset_index(drop=True)
df.to_csv('all_fits_inventory_v2.csv', index=False)
print(f"Saved all_fits_inventory_v2.csv ({len(df)} files)")

print("\n--- Detailed Scan of OBS_ID x Filter x Spatial Track ---")
for (obs_id, ftr), grp in df.groupby(['obs_id', 'filter']):
    min_t = grp['dt'].min()
    max_t = grp['dt'].max()
    dur_h = (max_t - min_t).total_seconds() / 3600.0
    dur_m = (max_t - min_t).total_seconds() / 60.0
    
    # Check any flare during [min_t - 15min, max_t + 15min]
    fls = goes[(goes['start_dt'] >= min_t - timedelta(minutes=15)) & (goes['start_dt'] <= max_t + timedelta(minutes=15))]
    print(f"\nOBS: {obs_id:15s} | Filter: {ftr:5s} | Frames: {len(grp):4d} | Span: {min_t} -> {max_t} ({dur_m:.1f} m)")
    rois = grp['roi_id'].unique().tolist()
    shapes = grp.apply(lambda r: f"{r.ny}x{r.nx}", axis=1).unique().tolist()
    print(f"   ROIs: {rois} | Shapes: {shapes}")
    if len(fls) > 0:
        for _, fl in fls.iterrows():
            print(f"   >>> FLARE: {fl['flare_id']} | Class: {fl['flare_class']} | Start: {fl['start_time']} | Peak: {fl['time']} | End: {fl['end_time']} | AR: {fl['active_region']}")
