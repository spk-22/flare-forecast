"""Analyze chronological distribution of sequences for train/test split."""
import pandas as pd
import numpy as np

df = pd.read_csv('Sequences/suit2024_sequences_metadata.csv')
df['t0'] = pd.to_datetime(df['cutoff_t0_utc'])
df = df.sort_values('t0').reset_index(drop=True)

print("Sequences sorted by t0:")
print(f"Total sequences: {len(df)}")
print(f"Time span: {df['t0'].min()} to {df['t0'].max()}")

pos_indices = df[df['label'] == 1].index.tolist()
print(f"Positive sample indices (in chronologically sorted list): {pos_indices}")
for idx in pos_indices:
    row = df.iloc[idx]
    print(f"  Pos sample index {idx}: {row['sequence_id']} | t0 = {row['cutoff_t0_utc']} | target = {row['target_time_utc']}")

# Check split point:
# If we do e.g. 70/30 or 75/25 split:
# Total = 310.
# 70% is index 217. Indices 218..309 are in test set.
# The positive samples are around indices 220..250!
# Let's see how many samples and positives fall in train vs test for different cutoff points.
for split_pct in [0.65, 0.70, 0.75, 0.80]:
    split_idx = int(len(df) * split_pct)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    print(f"\nSplit at {split_pct*100:.0f}% (Train < index {split_idx}, Test >= index {split_idx}):")
    print(f"  Train: {len(train_df)} samples, {train_df['label'].sum()} pos ({train_df['label'].mean()*100:.2f}%) | {train_df['t0'].min()} to {train_df['t0'].max()}")
    print(f"  Test:  {len(test_df)} samples, {test_df['label'].sum()} pos ({test_df['label'].mean()*100:.2f}%) | {test_df['t0'].min()} to {test_df['t0'].max()}")
