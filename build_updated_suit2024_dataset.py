"""Build the updated NB04 sequence dataset from suit2024 FITS files."""
import csv, json, os, re
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
from astropy.io import fits

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'suit2024'
GOES = ROOT / 'labels' / 'goes_xrs_flare_report_2024.csv'
FRAMES = ROOT / 'suit2024_frames_features.csv'
SEQUENCES = ROOT / 'suit2024_sequences_labelled.csv'
MEMBERS = ROOT / 'suit2024_sequence_members.csv'
SEQUENCE_DIR = DATA / 'sequences'

def parse_time(value):
    value = str(value).replace('Z', '')
    if '.' in value:
        prefix, fraction = value.split('.', 1)
        value = prefix + '.' + fraction[:6]
    return datetime.fromisoformat(value)

def header_value(header, *keys):
    for key in keys:
        value = header.get(key)
        if value not in (None, '', 'NA'):
            return value
    return ''

def number(value):
    try: return float(value)
    except (TypeError, ValueError): return ''

def frame_record(path):
    with fits.open(path, memmap=False) as hdul:
        header = hdul[0].header
        image = np.asarray(hdul[0].data)
        good = image[np.isfinite(image)]
        percentiles = np.percentile(good, [1, 50, 99, 99.9]) if good.size else [np.nan] * 4
        shape_y, shape_x = image.shape[-2:]
        product = re.search(r'_(\d{4})NB04\.fits$', path.name)
        return {
            'frame_id': path.stem, 'path': str(path.relative_to(ROOT)),
            'timestamp_utc': parse_time(header_value(header, 'DATE-OBS', 'DATE_OBS')).isoformat(sep=' '),
            'filter': header_value(header, 'FTR_NAME', 'FILTER'), 'obs_id': header_value(header, 'OBS_ID'),
            'roi_id': header_value(header, 'ROI_ID'), 'product_code': product.group(1) if product else '',
            'shape_y': shape_y, 'shape_x': shape_x, 'x1': header_value(header, 'X1'), 'x2': header_value(header, 'X2'),
            'y1': header_value(header, 'Y1'), 'y2': header_value(header, 'Y2'),
            'is_full_disk': int(shape_y >= 4000 and shape_x >= 4000), 'quality_percent': number(header_value(header, 'QVAL')),
            'exposure_s': number(header_value(header, 'EXPTIME', 'EXPOSURE')),
            'ctype1': header_value(header, 'CTYPE1'), 'ctype2': header_value(header, 'CTYPE2'),
            'cunit1': header_value(header, 'CUNIT1'), 'cunit2': header_value(header, 'CUNIT2'),
            'crpix1': number(header_value(header, 'CRPIX1')), 'crpix2': number(header_value(header, 'CRPIX2')),
            'crval1': number(header_value(header, 'CRVAL1')), 'crval2': number(header_value(header, 'CRVAL2')),
            'cdelt1_arcsec': number(header_value(header, 'CDELT1', 'CDELTA1')), 'cdelt2_arcsec': number(header_value(header, 'CDELT2', 'CDELTA2')),
            'crota2_deg': number(header_value(header, 'CROTA2')), 'rsun_obs_arcsec': number(header_value(header, 'RSUN_OBS')),
            'dsun_obs_m': number(header_value(header, 'DSUN_OBS')), 'hglon_obs_deg': number(header_value(header, 'HGLN_OBS')),
            'hglat_obs_deg': number(header_value(header, 'HGLT_OBS')), 'mean_dn': float(np.mean(good)), 'std_dn': float(np.std(good)),
            'p01_dn': float(percentiles[0]), 'p50_dn': float(percentiles[1]), 'p99_dn': float(percentiles[2]), 'p999_dn': float(percentiles[3]),
            'nan_count': int(image.size - good.size), 'has_basic_wcs': int(all(k in header for k in ('CTYPE1','CTYPE2','CRVAL1','CRVAL2','CRPIX1','CRPIX2'))),
            'header_json': json.dumps({key: str(value) for key, value in header.items()}, sort_keys=True)
        }

def main():
    files = sorted(DATA.rglob('*NB04*.fits'))
    records = [frame_record(path) for path in files]
    records.sort(key=lambda row: row['timestamp_utc'])
    with FRAMES.open('w', newline='', encoding='utf-8') as out:
        writer = csv.DictWriter(out, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)

    with GOES.open(newline='', encoding='utf-8') as source:
        events = list(csv.DictReader(source))
    tracks = defaultdict(list)
    for row in records:
        # Product code is intentionally excluded: it is part of the same NB04 stream.
        tracks[(row['obs_id'], row['roi_id'], row['shape_y'], row['shape_x'])].append(row)

    sequences, members = [], []
    for key, track in tracks.items():
        track.sort(key=lambda row: row['timestamp_utc'])
        times = [parse_time(row['timestamp_utc']) for row in track]
        for end in range(79, len(track)):
            start = end - 79; window = track[start:end + 1]; window_times = times[start:end + 1]
            gaps = [(right - left).total_seconds() for left, right in zip(window_times, window_times[1:])]
            cutoff = window_times[-1]
            hits = [event for event in events if event['flare_class'].startswith(('M','X')) and cutoff < datetime.strptime(event['start_time'], '%Y-%m-%d %H:%M:%S') <= cutoff + timedelta(seconds=90)]
            event = hits[0] if hits else {}
            sequence_id = f'{key[0]}_ROI{key[1]}_{key[2]}x{key[3]}_{start:05d}_{end:05d}'
            sequences.append({
                'sequence_id': sequence_id, 'label': int(bool(hits)),
                'label_definition': '1=M/X GOES onset in next 90 s; 0=no M/X onset in next 90 s',
                'obs_id': key[0], 'roi_id': key[1], 'shape_y': key[2], 'shape_x': key[3], 'frame_count': 80,
                'start_utc': window_times[0].isoformat(sep=' '), 'end_utc': cutoff.isoformat(sep=' '),
                'history_span_minutes': round((cutoff-window_times[0]).total_seconds()/60, 3),
                'max_internal_gap_seconds': round(max(gaps), 3), 'gap_quality_flag': 'review' if max(gaps) > 135 else 'continuous',
                'goes_flare_id': event.get('flare_id',''), 'goes_flare_class': event.get('flare_class',''),
                'goes_active_region': event.get('active_region',''), 'goes_hpc_x_arcsec': event.get('flare_loc_xrs_hpc_x',''),
                'goes_hpc_y_arcsec': event.get('flare_loc_xrs_hpc_y',''), 'hel1os_status': 'not_joined_no_catalogue'
            })
            for order, frame in enumerate(window):
                members.append({'sequence_id': sequence_id, 'frame_order': order, 'frame_id': frame['frame_id'], 'timestamp_utc': frame['timestamp_utc'], 'source_path': frame['path']})

    with SEQUENCES.open('w', newline='', encoding='utf-8') as out:
        writer = csv.DictWriter(out, fieldnames=list(sequences[0])); writer.writeheader(); writer.writerows(sequences)
    with MEMBERS.open('w', newline='', encoding='utf-8') as out:
        writer = csv.DictWriter(out, fieldnames=list(members[0])); writer.writeheader(); writer.writerows(members)

    SEQUENCE_DIR.mkdir(exist_ok=True)
    for sequence in sequences:
        folder = SEQUENCE_DIR / sequence['sequence_id']; folder.mkdir(exist_ok=True)
    for member in members:
        source = ROOT / member['source_path']; target = SEQUENCE_DIR / member['sequence_id'] / source.name
        if not target.exists(): os.link(source, target)
    print(f'frames={len(records)} sequences={len(sequences)} positive={sum(x["label"] for x in sequences)} negative={sum(not x["label"] for x in sequences)}')

if __name__ == '__main__': main()
