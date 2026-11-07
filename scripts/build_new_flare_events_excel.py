"""
Extract SUIT Flare Events from the new_suit_flares_only folder.
Identifies distinct candidate solar flare events and creates 'New Flare Events.xlsx'.
Uses the same methodology as the previous suit_flares_only extraction.
"""
from pathlib import Path
from collections import defaultdict
from datetime import datetime, timedelta
import warnings
import numpy as np
import pandas as pd
from astropy.io import fits

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / 'new_suit_flares_only'
OUTPUT_FILE = ROOT / 'New Flare Events.xlsx'

# Wavelength lookup for SUIT filters
FILTER_WAVELENGTHS = {
    'NB01': '2140 Angstrom',
    'NB02': '2770 Angstrom',
    'NB03': '2796 Angstrom (Mg II k)',
    'NB04': '2803 Angstrom (Mg II h)',
    'NB05': '2832 Angstrom',
    'NB06': '3000 Angstrom',
    'NB07': '3880 Angstrom',
    'NB08': '3965 Angstrom (Ca II H)',
    'BB01': '2000-4000 Angstrom (Broadband)',
    'BB02': '3000-5000 Angstrom (Broadband)',
    'BB03': '3500-5500 Angstrom (Broadband)',
}


def parse_fits_headers(data_dir: Path) -> pd.DataFrame:
    """Parse all FITS file headers and return a DataFrame with key metadata."""
    fits_files = sorted(data_dir.glob('*.fits'))
    print(f"Found {len(fits_files)} FITS files in {data_dir}")

    records = []
    errors = []
    for i, fpath in enumerate(fits_files):
        if (i + 1) % 100 == 0:
            print(f"  Parsed {i + 1}/{len(fits_files)} files...")
        try:
            with fits.open(fpath, memmap=False) as hdul:
                h = hdul[0].header
                
                # Extract filter name from filename
                fname = fpath.stem
                ftr_name = h.get('FTR_NAME', '')
                if not ftr_name:
                    # Try to extract from filename
                    parts = fname.split('_')
                    if parts:
                        last = parts[-1]
                        for prefix in ['NB', 'BB']:
                            idx = last.find(prefix)
                            if idx >= 0:
                                ftr_name = last[idx:idx+4]
                                break

                # Parse observation time
                date_obs = h.get('DATE-OBS', h.get('T_OBS', ''))
                
                rec = {
                    'filename': fpath.name,
                    'filepath': str(fpath),
                    'obs_id': h.get('OBS_ID', ''),
                    'prop_id': h.get('PROP_ID', ''),
                    'date_obs': str(date_obs),
                    'obs_start': h.get('OBS_STRT', ''),
                    'obs_stop': h.get('OBS_STP', ''),
                    'filter_name': ftr_name,
                    'wavelnth': h.get('WAVELNTH', 0),
                    'roi_id': h.get('ROI_ID', -1),
                    'roi_ff': h.get('ROI_FF', ''),
                    'obs_mode': h.get('OBS_MODE', -1),
                    'flr_trig': h.get('FLR_TRIG', 0),
                    'norm_flr': h.get('NORM_FLR', 0),
                    'prom_flr': h.get('PROM_FLR', 0),
                    'flr_ed': h.get('FLR_ED', 0),
                    'naxis1': h.get('NAXIS1', 0),
                    'naxis2': h.get('NAXIS2', 0),
                    'exptime': h.get('EXPTIME', 0.0),
                    'crpix1': h.get('CRPIX1', 0.0),
                    'crpix2': h.get('CRPIX2', 0.0),
                    'crval1': h.get('CRVAL1', 0.0),
                    'crval2': h.get('CRVAL2', 0.0),
                    'cdelt1': h.get('CDELT1', 0.0),
                    'cdelt2': h.get('CDELT2', 0.0),
                    'crota2': h.get('CROTA2', 0.0),
                    'r_sun': h.get('R_SUN', 0.0),
                    'rsun_obs': h.get('RSUN_OBS', 0.0),
                    'p_angle': h.get('P_ANGLE', 0.0),
                    'dsun_obs': h.get('DSUN_OBS', 0.0),
                    'hglt_obs': h.get('HGLT_OBS', 0.0),
                    'hgln_obs': h.get('HGLN_OBS', 0.0),
                    'sun_cx': h.get('SUN_CX', 0.0),
                    'sun_cy': h.get('SUN_CY', 0.0),
                    'x1': h.get('X1', 0),
                    'y1': h.get('Y1', 0),
                    'x2': h.get('X2', 0),
                    'y2': h.get('Y2', 0),
                    'img_type': h.get('IMG_TYPE', ''),
                    'qval': h.get('QVAL', 0.0),
                    'qdesc': h.get('QDESC', ''),
                    'frm_cnt': h.get('FRM_CNT', 0),
                }
                records.append(rec)
        except Exception as e:
            errors.append({'filename': fpath.name, 'error': str(e)})
            print(f"  ERROR reading {fpath.name}: {e}")

    df = pd.DataFrame(records)
    print(f"\nSuccessfully parsed {len(df)} files, {len(errors)} errors")
    return df, errors


def compute_helioprojective_coords(row):
    """Compute center-of-ROI helioprojective coordinates in arcsec."""
    nx, ny = row['naxis1'], row['naxis2']
    cx_pix = (nx + 1) / 2.0
    cy_pix = (ny + 1) / 2.0
    
    crpix1, crpix2 = row['crpix1'], row['crpix2']
    cdelt1, cdelt2 = row['cdelt1'], row['cdelt2']
    crval1, crval2 = row['crval1'], row['crval2']
    
    tx = crval1 + (cx_pix - crpix1) * cdelt1
    ty = crval2 + (cy_pix - crpix2) * cdelt2
    return tx, ty


def identify_events(df: pd.DataFrame, gap_minutes: float = 30.0) -> pd.DataFrame:
    """
    Group FITS frames into distinct flare events based on:
    1. Same OBS_ID
    2. Same FLR_TRIG value
    3. Temporal clustering (gaps > gap_minutes create new events)
    """
    # Parse observation timestamps
    df['dt'] = pd.to_datetime(df['date_obs'], errors='coerce')
    df = df.sort_values('dt').reset_index(drop=True)
    
    # Group by OBS_ID and FLR_TRIG
    groups = df.groupby(['obs_id', 'flr_trig'])
    
    events = []
    event_counter = 0
    
    for (obs_id, flr_trig), group in groups:
        group = group.sort_values('dt').reset_index(drop=True)
        
        if len(group) == 0:
            continue
        
        # Sub-divide by temporal gaps
        sub_events = []
        current_sub = [0]
        
        for i in range(1, len(group)):
            dt_diff = (group.loc[i, 'dt'] - group.loc[i - 1, 'dt']).total_seconds() / 60.0
            if dt_diff > gap_minutes:
                sub_events.append(current_sub)
                current_sub = [i]
            else:
                current_sub.append(i)
        sub_events.append(current_sub)
        
        for sub_idx in sub_events:
            sub_group = group.iloc[sub_idx]
            event_counter += 1
            
            first_dt = sub_group['dt'].min()
            last_dt = sub_group['dt'].max()
            date_str = first_dt.strftime('%Y%m%d')
            
            event_id = f"SUIT_FLR_{date_str}_{event_counter:02d}"
            
            for _, row in sub_group.iterrows():
                events.append({
                    'event_id': event_id,
                    'obs_id': obs_id,
                    'flr_trig': flr_trig,
                    **row.to_dict()
                })
    
    return pd.DataFrame(events)


def build_flare_events_excel(df_events: pd.DataFrame, errors: list, data_dir: Path, output_file: Path):
    """Build the Excel workbook with Flare Events and Processing Log sheets."""
    
    # --- Sheet 1: Flare Events ---
    rows = []
    for _, row in df_events.iterrows():
        dt = row['dt']
        obs_date = dt.strftime('%Y-%m-%d') if pd.notna(dt) else ''
        obs_time = dt.strftime('%H:%M:%S.%f')[:-3] if pd.notna(dt) else ''
        
        # Compute helioprojective coords
        tx, ty = compute_helioprojective_coords(row)
        
        filter_desc = FILTER_WAVELENGTHS.get(row['filter_name'], f"{row['filter_name']} ({row['wavelnth']} Angstrom)")
        
        rows.append({
            'Event_ID': row['event_id'],
            'File_ID': Path(row['filename']).stem,
            'Filename': row['filename'],
            'ROI_ID': row['roi_id'],
            'Observation_Date_UTC': obs_date,
            'Observation_Time_UTC': obs_time,
            'Flare_Onset_Date_UTC': obs_date,  # Will be refined below
            'Flare_Onset_Time_UTC': '',  # To be filled via cross-reference
            'Onset_Time_Source': f"SUIT Payload Flag (FLR_TRIG={row['flr_trig']})",
            'Wavelength_or_Filter': f"{row['filter_name']} ({row['wavelnth']} Angstrom)",
            'Observation_ID': row['obs_id'],
            'Event_Grouping_Method': (
                f"Payload Flag Grouping (FLR_TRIG={row['flr_trig']}, "
                f"OBS_ID={row['obs_id']}, ROI_ID={row['roi_id']})"
            ),
            'Confidence_Status': 'Candidate Flare Observation (Payload-flagged)',
            'NORM_FLR': row['norm_flr'],
            'PROM_FLR': row['prom_flr'],
            'FLR_ED': row['flr_ed'],
            'Image_Size': f"{row['naxis1']}x{row['naxis2']}",
            'Exposure_Time_ms': row['exptime'],
            'HPC_Tx_arcsec': round(tx, 2),
            'HPC_Ty_arcsec': round(ty, 2),
            'Solar_Radius_arcsec': round(row['rsun_obs'], 2),
            'P_Angle_deg': round(row['p_angle'], 3),
            'Quality_Value': row['qval'],
            'Quality_Desc': row['qdesc'],
            'OBS_Start_UTC': row['obs_start'],
            'OBS_Stop_UTC': row['obs_stop'],
        })
    
    df_out = pd.DataFrame(rows)
    
    # Attempt GOES cross-reference for onset times
    try:
        df_out = cross_reference_goes(df_out, data_dir.parent)
    except Exception as e:
        print(f"  GOES cross-reference skipped: {e}")
    
    # --- Sheet 2: Processing Log ---
    fits_files = sorted(data_dir.glob('*.fits'))
    total_size = sum(f.stat().st_size for f in fits_files)
    
    unique_events = df_out['Event_ID'].nunique()
    unique_obs_ids = df_events['obs_id'].nunique()
    unique_flr_trigs = df_events['flr_trig'].nunique()
    unique_filters = df_events['filter_name'].nunique()
    date_range = f"{df_events['dt'].min().strftime('%Y-%m-%d %H:%M')} to {df_events['dt'].max().strftime('%Y-%m-%d %H:%M')}"
    
    log_rows = [
        ('ARCHIVE & FILE PARSING AUDIT', ''),
        ('Source Directory', str(data_dir)),
        ('Target Observation Campaigns', ', '.join(df_events['obs_id'].unique())),
        ('Total Files Scanned', str(len(fits_files))),
        ('FITS Files Successfully Parsed', str(len(df_events))),
        ('Parse Errors', str(len(errors))),
        ('Total Data Size', f"{total_size:,} bytes / {total_size / (1024**3):.2f} GB"),
        ('', ''),
        ('EVENT IDENTIFICATION', ''),
        ('Distinct Flare Events Identified', str(unique_events)),
        ('Unique OBS_IDs', str(unique_obs_ids)),
        ('Unique FLR_TRIG Values', str(unique_flr_trigs)),
        ('Unique Filters Used', str(unique_filters)),
        ('Observation Date Range', date_range),
        ('Grouping Method', 'OBS_ID + FLR_TRIG + Temporal Gap (30 min)'),
        ('', ''),
        ('ROI BREAKDOWN', ''),
    ]
    
    roi_counts = df_events.groupby('roi_id').size()
    for roi, count in roi_counts.items():
        log_rows.append((f'  ROI {roi} Files', str(count)))
    
    log_rows.append(('', ''))
    log_rows.append(('FILTER BREAKDOWN', ''))
    filter_counts = df_events.groupby('filter_name').size().sort_index()
    for filt, count in filter_counts.items():
        wav_desc = FILTER_WAVELENGTHS.get(filt, '')
        log_rows.append((f'  {filt} ({wav_desc})', str(count)))
    
    log_rows.append(('', ''))
    log_rows.append(('OBS_ID BREAKDOWN', ''))
    for obs_id in sorted(df_events['obs_id'].unique()):
        sub = df_events[df_events['obs_id'] == obs_id]
        dt_min = sub['dt'].min().strftime('%Y-%m-%d %H:%M:%S')
        dt_max = sub['dt'].max().strftime('%Y-%m-%d %H:%M:%S')
        log_rows.append((f'  {obs_id}', f"{len(sub)} files, {dt_min} to {dt_max}"))
    
    log_rows.append(('', ''))
    log_rows.append(('EVENT DETAILS', ''))
    for eid in sorted(df_out['Event_ID'].unique()):
        sub = df_out[df_out['Event_ID'] == eid]
        log_rows.append((
            f'  {eid}',
            f"{len(sub)} frames, {sub['Observation_Date_UTC'].iloc[0]} "
            f"{sub['Observation_Time_UTC'].iloc[0]} to {sub['Observation_Time_UTC'].iloc[-1]}"
        ))
    
    if errors:
        log_rows.append(('', ''))
        log_rows.append(('PARSING ERRORS', ''))
        for err in errors:
            log_rows.append((err['filename'], err['error']))
    
    log_rows.append(('', ''))
    log_rows.append(('PROCESSING METADATA', ''))
    log_rows.append(('Script', 'build_new_flare_events_excel.py'))
    log_rows.append(('Generated At', datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')))
    log_rows.append(('Python Libraries', 'astropy, pandas, openpyxl'))
    
    df_log = pd.DataFrame(log_rows, columns=['ARCHIVE & FILE PARSING AUDIT', 'Details'])
    
    # --- Write Excel ---
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        df_out.to_excel(writer, sheet_name='Flare Events', index=False)
        df_log.to_excel(writer, sheet_name='Processing Log', index=False)
    
    print(f"\n[OK] Excel workbook saved to: {output_file}")
    print(f"  Sheet 'Flare Events': {len(df_out)} rows, {len(df_out.columns)} columns")
    print(f"  Sheet 'Processing Log': {len(df_log)} rows")
    print(f"  Distinct events: {unique_events}")


def cross_reference_goes(df_out: pd.DataFrame, root_dir: Path) -> pd.DataFrame:
    """Try to cross-reference with GOES flare catalogs in the labels directory."""
    labels_dir = root_dir / 'labels'
    if not labels_dir.exists():
        return df_out
    
    # Look for GOES CSV files covering Dec 2025
    goes_files = list(labels_dir.glob('goes_*.csv'))
    hek_files = list(labels_dir.glob('hek_*.json'))
    
    print(f"\n  Found {len(goes_files)} GOES CSV files, {len(hek_files)} HEK JSON files")
    
    # Try to load GOES data
    goes_flares = []
    for gf in goes_files:
        try:
            gdf = pd.read_csv(gf)
            goes_flares.append(gdf)
        except Exception:
            pass
    
    if not goes_flares:
        print("  No GOES flare data loaded for cross-reference.")
        return df_out
    
    goes_all = pd.concat(goes_flares, ignore_index=True)
    print(f"  Loaded {len(goes_all)} GOES flare records for cross-referencing")
    
    # Try common column names for start time
    time_col = None
    for col in ['start_time', 'event_starttime', 'begin_time', 'fl_goess']:
        if col in goes_all.columns:
            time_col = col
            break
    
    if time_col:
        goes_all['goes_dt'] = pd.to_datetime(goes_all[time_col], errors='coerce')
        
        # For each event, find nearest GOES flare
        for event_id in df_out['Event_ID'].unique():
            mask = df_out['Event_ID'] == event_id
            event_rows = df_out[mask]
            event_start = pd.to_datetime(
                event_rows['Observation_Date_UTC'].iloc[0] + ' ' + event_rows['Observation_Time_UTC'].iloc[0]
            )
            
            # Find GOES flares within ±3 hours
            time_diffs = (goes_all['goes_dt'] - event_start).abs()
            close_mask = time_diffs < timedelta(hours=3)
            
            if close_mask.any():
                nearest_idx = time_diffs[close_mask].idxmin()
                nearest = goes_all.loc[nearest_idx]
                
                onset_dt = nearest['goes_dt']
                class_col = None
                for c in ['fl_goescls', 'fl_class', 'goes_class', 'class']:
                    if c in nearest.index:
                        class_col = c
                        break
                
                class_str = nearest.get(class_col, 'Unknown') if class_col else 'Unknown'
                
                df_out.loc[mask, 'Flare_Onset_Date_UTC'] = onset_dt.strftime('%Y-%m-%d')
                df_out.loc[mask, 'Flare_Onset_Time_UTC'] = onset_dt.strftime('%H:%M:%S')
                df_out.loc[mask, 'Onset_Time_Source'] = (
                    f"GOES XRS Cross-reference ({class_str}), "
                    + df_out.loc[mask, 'Onset_Time_Source'].iloc[0]
                )
                df_out.loc[mask, 'Confidence_Status'] = 'Confirmed Flare Observation (GOES cross-referenced)'
                
                print(f"    {event_id} matched GOES flare: {class_str} at {onset_dt}")
    
    return df_out


def main():
    print("=" * 70)
    print("SUIT Flare Event Extraction - new_suit_flares_only")
    print("=" * 70)
    
    if not DATA_DIR.exists():
        print(f"ERROR: Data directory not found: {DATA_DIR}")
        return
    
    # Step 1: Parse all FITS headers
    print("\n[1/3] Parsing FITS file headers...")
    df, errors = parse_fits_headers(DATA_DIR)
    
    # Step 2: Identify distinct flare events
    print("\n[2/3] Identifying distinct flare events...")
    df_events = identify_events(df)
    
    # Summary
    print(f"\n  OBS_IDs found: {sorted(df_events['obs_id'].unique())}")
    print(f"  FLR_TRIG values: {sorted(df_events['flr_trig'].unique())}")
    print(f"  Date range: {df_events['dt'].min()} to {df_events['dt'].max()}")
    print(f"  Filters: {sorted(df_events['filter_name'].unique())}")
    print(f"  ROI IDs: {sorted(df_events['roi_id'].unique())}")
    print(f"  Event IDs: {sorted(df_events['event_id'].unique())}")
    
    # Step 3: Build Excel workbook
    print("\n[3/3] Building Excel workbook...")
    build_flare_events_excel(df_events, errors, DATA_DIR, OUTPUT_FILE)
    
    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == '__main__':
    main()
