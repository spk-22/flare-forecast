"""Scan all extracted FITS files, inspect headers, and summarize observation campaigns and ROIs."""
from pathlib import Path
from astropy.io import fits
from collections import Counter
import pandas as pd

extract_dir = Path('d:/flare-forecast/suit_flares_extracted')
fits_files = sorted(extract_dir.glob('*.fits'))

records = []
errors = []

for idx, f in enumerate(fits_files):
    try:
        with fits.open(f, memmap=False) as hdul:
            hdr = hdul[0].header
            records.append({
                'filename': f.name,
                'path': str(f.relative_to(extract_dir)),
                'obs_id': str(hdr.get('OBS_ID', '')),
                'roi_id': hdr.get('ROI_ID', hdr.get('HKROIID', '')),
                'date_obs': str(hdr.get('DATE-OBS', hdr.get('T_OBS', ''))),
                'obs_strt': str(hdr.get('OBS_STRT', '')),
                'obs_stp': str(hdr.get('OBS_STP', '')),
                'filter': hdr.get('FTR_NAME', hdr.get('FILTER', '')),
                'wavelnth': hdr.get('WAVELNTH', ''),
                'waveunit': hdr.get('WAVEUNIT', ''),
                'exptime': hdr.get('EXPTIME', ''),
                'qval': hdr.get('QVAL', ''),
                'qdesc': hdr.get('QDESC', ''),
                'flr_ed': hdr.get('FLR_ED', ''),
                'flr_trig': hdr.get('FLR_TRIG', ''),
                'hl1osflg': hdr.get('HL1OSFLG', ''),
                'heliostr': hdr.get('HELIOSTR', ''),
                'nrmflg': hdr.get('NRMFLG', ''),
                'prmflg': hdr.get('PRMFLG', ''),
                'sx1flg': hdr.get('SX1FLG', ''),
                'sx2flg': hdr.get('SX2FLG', ''),
                'shape_x': hdr.get('NAXIS1', ''),
                'shape_y': hdr.get('NAXIS2', '')
            })
    except Exception as e:
        errors.append((f.name, str(e)))

df = pd.DataFrame(records)
print(f"Total files parsed: {len(df)}, Errors: {len(errors)}")
print("\nROI counts:")
print(df['roi_id'].value_counts())
print("\nOBS_ID counts:")
print(df['obs_id'].value_counts())
print("\nFilter counts:")
print(df['filter'].value_counts())
print("\nUnique observation start times (OBS_STRT):")
print(df['obs_strt'].value_counts())
print("\nMin and Max DATE-OBS:")
print("Min:", df['date_obs'].min(), "Max:", df['date_obs'].max())

df.to_csv('d:/flare-forecast/suit_flares_extracted_summary.csv', index=False)
print("Saved summary to d:/flare-forecast/suit_flares_extracted_summary.csv")
