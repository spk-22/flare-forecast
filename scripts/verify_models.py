import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

import torch
import train_models_v3 as t

X_tr, y_tr, X_va, y_va, X_te, y_te, nf, mu, sig = t.load_original_data()
print("Features count:", nf)

bdh = t.BabyDragonHatchling(nf, 64)
lstm = t.LSTMFlarePredictor(nf, 64)
tft = t.TemporalFusionTransformer(nf, 64)

x = torch.zeros(2, 80, nf)
print("BDH out shape:", bdh(x).shape)
print("LSTM out shape:", lstm(x).shape)
print("TFT out shape:", tft(x).shape)
print("All model architectures verified successfully!")

