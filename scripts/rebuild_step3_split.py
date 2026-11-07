"""
scripts/rebuild_step3_split.py
Partition rebuilt dataset into event-aware, strictly chronological Train, Val, and Test splits.
Zero cross-event leakage.
"""

import sys, os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SEQ_CSV = ROOT / "data" / "suit2024_sequences_labelled_v4.csv"
OUTPUT_SPLIT_JSON = ROOT / "data" / "dataset_splits_v4.json"

def main():
    df = pd.read_csv(SEQ_CSV)
    print(f"Total sequences loaded: {len(df)}")
    
    df['date_str'] = pd.to_datetime(df['t0_utc']).dt.strftime('%Y-%m-%d')
    df['hour'] = pd.to_datetime(df['t0_utc']).dt.hour
    
    # Chronological, event-aware partition
    train_mask = df['date_str'] <= '2024-12-29'
    val_mask = (df['date_str'] == '2024-12-30') & (df['hour'] < 14)
    test_mask = ((df['date_str'] == '2024-12-30') & (df['hour'] >= 14)) | (df['date_str'] >= '2024-12-31')
    
    train_indices = np.where(train_mask)[0].tolist()
    val_indices = np.where(val_mask)[0].tolist()
    test_indices = np.where(test_mask)[0].tolist()
    
    # Sanity checks
    assert len(set(train_indices).intersection(set(val_indices))) == 0
    assert len(set(train_indices).intersection(set(test_indices))) == 0
    assert len(set(val_indices).intersection(set(test_indices))) == 0
    assert len(train_indices) + len(val_indices) + len(test_indices) == len(df)
    
    splits = {
        'train_indices': train_indices,
        'val_indices': val_indices,
        'test_indices': test_indices,
        'stats': {
            'train': {
                'total': len(train_indices),
                'positives': int(df.iloc[train_indices]['label'].sum()),
                'negatives': int((df.iloc[train_indices]['label'] == 0).sum()),
                'unique_flares': int(df.iloc[train_indices][df.iloc[train_indices]['label'] == 1]['target_flare_start_utc'].nunique())
            },
            'val': {
                'total': len(val_indices),
                'positives': int(df.iloc[val_indices]['label'].sum()),
                'negatives': int((df.iloc[val_indices]['label'] == 0).sum()),
                'unique_flares': int(df.iloc[val_indices][df.iloc[val_indices]['label'] == 1]['target_flare_start_utc'].nunique())
            },
            'test': {
                'total': len(test_indices),
                'positives': int(df.iloc[test_indices]['label'].sum()),
                'negatives': int((df.iloc[test_indices]['label'] == 0).sum()),
                'unique_flares': int(df.iloc[test_indices][df.iloc[test_indices]['label'] == 1]['target_flare_start_utc'].nunique())
            }
        }
    }
    
    with open(OUTPUT_SPLIT_JSON, 'w', encoding='utf-8') as f:
        json.dump(splits, f, indent=2)
        
    print(f"Splits saved to {OUTPUT_SPLIT_JSON}")
    print("Partition statistics:")
    for split_name, st in splits['stats'].items():
        pct = (st['total'] / len(df)) * 100
        pos_pct = (st['positives'] / st['total']) * 100
        print(f"  {split_name.upper():5s}: {st['total']:4d} sequences ({pct:.1f}%) | {st['positives']:2d} positives ({pos_pct:.2f}%) | {st['unique_flares']} independent flares")

if __name__ == '__main__':
    main()

