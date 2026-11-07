import pandas as pd
from astropy.io import fits
import numpy as np

df = pd.read_csv('all_fits_inventory_v2.csv')
t24 = df[df['obs_id'] == 'T24_0503_000268']

print("--- ROI Crops in T24_0503_000268 ---")
for roi in ['15', '17', '19', '100']:
    sub = t24[(t24['roi_id'].astype(str) == roi) & (t24['ny'] < 1000)]
    f0 = sub.iloc[0]['path']
    h = fits.getheader(f0)
    
    # Calculate physical coordinates on full detector (4096 x 4096)
    # X1, X2, Y1, Y2 are the pixel boundaries of the ROI crop on the full 4k detector
    x1, x2 = h.get('X1', 0), h.get('X2', 0)
    y1, y2 = h.get('Y1', 0), h.get('Y2', 0)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    
    # In full detector coordinates, Sun center is at CRPIX1_FD ~ 2480, CRPIX2_FD ~ 1200
    # Let's inspect full disk header to get exact full-disk sun center
    print(f"ROI {roi:4s} | Frames: {len(sub):3d} | X1-X2: [{x1:4d}, {x2:4d}] | Y1-Y2: [{y1:4d}, {y2:4d}] | Center pixel on full CCD: ({cx:.1f}, {cy:.1f})")

# Check full-disk frame
fd = t24[t24['ny'] == 2048].iloc[0]
h_fd = fits.getheader(fd['path'])
print(f"\nFull-Disk 2048x2048: CRPIX1={h_fd.get('CRPIX1')}, CRPIX2={h_fd.get('CRPIX2')}, CDELT1={h_fd.get('CDELT1')}, CROTA2={h_fd.get('CROTA2')}")
