"""Inspect header keywords and flare flags in the extracted FITS files."""
from pathlib import Path
from astropy.io import fits

sample_path = next(Path('d:/flare-forecast/suit_flares_extracted').glob('*.fits'))
print('Inspecting:', sample_path.name)
with fits.open(sample_path) as hdul:
    hdul.info()
    hdr = hdul[0].header
    for k in sorted(hdr.keys()):
        k_lower = k.lower()
        if any(w in k_lower for w in ['flr', 'flare', 'roi', 'date', 'time', 'obs', 'seq', 'qval', 'qdesc', 'filter', 'ftr', 'wave', 'onset', 'evnt', 'event', 'solx', 'hl1os']):
            cmt = hdr.comments[k] if k in hdr.comments else ''
            print(f"{k:12s} = {hdr.get(k)}  /  {cmt}")
