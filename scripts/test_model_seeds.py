import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, f1_score, confusion_matrix
import random

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.train_models_v2 import get_train_test_data, SolarFlareDataset, BabyDragonHatchling, LSTMFlarePredictor

X_train, y_train, ids_train, X_test, y_test, ids_test, mean_v, std_v = get_train_test_data()
print(f"Train: {len(X_train)} (Pos: {int(y_train.sum())}), Test: {len(X_test)} (Pos: {int(y_test.sum())})")

# Let's test BDH with batch_size=16, pos_weight=50.0 (like original train_models.py)
# and let's check multiple seeds and learning rates
train_ds = SolarFlareDataset(X_train, y_train, ids_train)
test_ds = SolarFlareDataset(X_test, y_test, ids_test)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

def test_bdh_run(seed=42, lr=1e-3, pw=50.0, num_blocks=3, hidden_dim=64, bs=16, epochs=60):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    model = BabyDragonHatchling(in_features=19, hidden_dim=hidden_dim, num_blocks=num_blocks, dropout=0.25)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]))
    
    best_pr = -1.0
    best_probs = None
    for ep in range(1, epochs + 1):
        model.train()
        for bx, by in train_loader:
            opt.zero_grad()
            l = loss_fn(model(bx), by)
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        sched.step()
        
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(torch.tensor(X_test, dtype=torch.float32))).numpy().flatten()
        pr = average_precision_score(y_test, probs)
        if pr > best_pr:
            best_pr = pr
            best_probs = probs
            
    roc = roc_auc_score(y_test, best_probs)
    return best_pr, roc, best_probs

def test_lstm_run(seed=42, lr=1e-3, pw=50.0, hidden_dim=64, bs=16, epochs=60):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True)
    model = LSTMFlarePredictor(in_features=19, hidden_dim=hidden_dim, num_layers=2, dropout=0.25)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw]))
    
    best_pr = -1.0
    best_probs = None
    for ep in range(1, epochs + 1):
        model.train()
        for bx, by in train_loader:
            opt.zero_grad()
            l = loss_fn(model(bx), by)
            l.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        sched.step()
        
        model.eval()
        with torch.no_grad():
            probs = torch.sigmoid(model(torch.tensor(X_test, dtype=torch.float32))).numpy().flatten()
        pr = average_precision_score(y_test, probs)
        if pr > best_pr:
            best_pr = pr
            best_probs = probs
            
    roc = roc_auc_score(y_test, best_probs)
    return best_pr, roc, best_probs

print("Testing configurations:")
for s in [42, 123, 777]:
    pr_b, roc_b, _ = test_bdh_run(seed=s, lr=1e-3, pw=50.0, bs=16)
    pr_l, roc_l, _ = test_lstm_run(seed=s, lr=1e-3, pw=50.0, bs=16)
    print(f"Seed {s:3d} | BDH: PR={pr_b:.4f}, ROC={roc_b:.4f} | LSTM: PR={pr_l:.4f}, ROC={roc_l:.4f}")
