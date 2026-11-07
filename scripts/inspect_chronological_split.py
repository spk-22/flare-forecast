import pandas as pd
from datetime import datetime

df = pd.read_csv('suit2024_sequences_labelled_v2.csv')
df['t0'] = pd.to_datetime(df['cutoff_t0_utc'])
df = df.sort_values('t0').reset_index(drop=True)

print(f"Total sequences: {len(df)}")
pos = df[df['label'] == 1]
print(f"Total positive sequences: {len(pos)}")

for idx, r in pos.iterrows():
    print(f"Index: {idx:3d} | t0: {r['cutoff_t0_utc']} | Flare: {r['target_flare_class']} | Track: {r['spatial_group']}")

# Let's inspect potential split points:
# We want both Train and Test to have positive samples and a strict chronological cutoff.
# For example, split around 70% (e.g. index 500) or around 2024-02-22 21:00:00
for split_pct in [0.60, 0.65, 0.70, 0.75]:
    s_idx = int(len(df) * split_pct)
    train_pos = (df.iloc[:s_idx]['label'] == 1).sum()
    test_pos = (df.iloc[s_idx:]['label'] == 1).sum()
    t_cutoff = df.iloc[s_idx]['cutoff_t0_utc']
    print(f"\nSplit at {split_pct*100:.0f}% (idx {s_idx}, Cutoff: {t_cutoff}):")
    print(f"   Train: {s_idx} samples ({train_pos} positive, {s_idx - train_pos} negative)")
    print(f"   Test:  {len(df) - s_idx} samples ({test_pos} positive, {len(df) - s_idx - test_pos} negative)")
