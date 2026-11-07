"""Inspect previous sequences and features."""
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

df_seq = pd.read_csv(ROOT / 'suit2024_sequences_labelled.csv')
print("Previous sequences count:", len(df_seq))
print("Label counts:\n", df_seq['label'].value_counts())
print("\nSample sequence columns:", list(df_seq.columns))
print("\nSample rows:")
print(df_seq[['sequence_id', 'label', 'obs_id', 'roi_id', 'shape_y', 'shape_x', 'frame_count', 'history_span_minutes', 'start_utc', 'end_utc', 'goes_flare_class']].head(10))

if df_seq['label'].sum() > 0:
    print("\nPositive sequences:")
    print(df_seq[df_seq['label'] == 1][['sequence_id', 'label', 'start_utc', 'end_utc', 'goes_flare_class', 'goes_flare_id']])

df_feat = pd.read_csv(ROOT / 'suit2024_frames_features.csv')
print("\nPrevious frame features count:", len(df_feat))
print("Features columns:", list(df_feat.columns))
