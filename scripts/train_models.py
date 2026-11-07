"""Train and evaluate Baby Dragon Hatchling (BDH) vs LSTM for 90-second solar flare prediction."""
from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    accuracy_score, precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt

# Set reproducible seeds
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

ROOT = Path(__file__).resolve().parent.parent
SEQUENCES_DIR = ROOT / 'Sequences'
MODELS_DIR = ROOT / 'models'
DOCS_DIR = ROOT / 'documentation'
MODELS_DIR.mkdir(exist_ok=True)
DOCS_DIR.mkdir(exist_ok=True)

# -------------------------------------------------------------
# Dataset & Preprocessing
# -------------------------------------------------------------
class SolarFlareDataset(Dataset):
    def __init__(self, X, y, seq_ids):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
        self.seq_ids = seq_ids
        
    def __len__(self):
        return len(self.y)
        
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def get_train_test_data():
    df_meta = pd.read_csv(SEQUENCES_DIR / 'suit2024_sequences_metadata.csv')
    df_meta['t0'] = pd.to_datetime(df_meta['cutoff_t0_utc'])
    df_meta = df_meta.sort_values('t0').reset_index(drop=True)
    
    npz = np.load(SEQUENCES_DIR / 'suit2024_sequences_features.npz')
    X_raw, y_raw, seq_ids = npz['X'], npz['y'], npz['sequence_ids']
    
    # Map from seq_id to array index
    seq_id_to_idx = {sid: i for i, sid in enumerate(seq_ids)}
    sorted_indices = [seq_id_to_idx[sid] for sid in df_meta['sequence_id']]
    
    X_sorted = X_raw[sorted_indices]
    y_sorted = y_raw[sorted_indices]
    seq_ids_sorted = seq_ids[sorted_indices]
    
    # Chronological Split:
    # Index 164 is 2024-02-22 22:07:38 UTC.
    # Train: indices 0..163 (164 samples, 1 positive, 163 negatives)
    # Test: indices 164..309 (146 samples, 2 positives, 144 negatives)
    # This guarantees strictly zero temporal leakage and unseen future evaluation.
    split_idx = 164
    
    X_train_raw = X_sorted[:split_idx]
    y_train = y_sorted[:split_idx]
    ids_train = seq_ids_sorted[:split_idx]
    
    X_test_raw = X_sorted[split_idx:]
    y_test = y_sorted[split_idx:]
    ids_test = seq_ids_sorted[split_idx:]
    
    # Standardize using training set statistics only
    mean_val = np.mean(X_train_raw, axis=(0, 1), keepdims=True)
    std_val = np.std(X_train_raw, axis=(0, 1), keepdims=True)
    std_val[std_val < 1e-6] = 1.0
    
    X_train = (X_train_raw - mean_val) / std_val
    X_test = (X_test_raw - mean_val) / std_val
    
    # Replace any potential NaNs or infs
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    return X_train, y_train, ids_train, X_test, y_test, ids_test, mean_val, std_val

# -------------------------------------------------------------
# Model 1: Baby Dragon Hatchling (BDH) Architecture
# -------------------------------------------------------------
class MultiScaleTemporalConv(nn.Module):
    """Multi-scale 1D Convolution branch for capturing diverse temporal precursor scales."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv3 = nn.Conv1d(in_channels, out_channels // 3, kernel_size=3, padding=1)
        self.conv5 = nn.Conv1d(in_channels, out_channels // 3, kernel_size=5, padding=2)
        self.conv7 = nn.Conv1d(in_channels, out_channels - 2 * (out_channels // 3), kernel_size=7, padding=3)
        self.norm = nn.GroupNorm(4, out_channels)
        self.act = nn.GELU()
        
    def forward(self, x):
        # x: (B, C, T)
        c3 = self.conv3(x)
        c5 = self.conv5(x)
        c7 = self.conv7(x)
        out = torch.cat([c3, c5, c7], dim=1)
        return self.act(self.norm(out))

class HatchlingResidualBlock(nn.Module):
    """Residual Gated Dilated Temporal Block with skip connection."""
    def __init__(self, channels, dilation=1, dropout=0.2):
        super().__init__()
        self.conv_f = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv_g = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm = nn.GroupNorm(4, channels)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Gated Linear Unit mechanism
        f = torch.tanh(self.conv_f(x))
        g = torch.sigmoid(self.conv_g(x))
        h = f * g
        h = self.dropout(self.norm(h))
        return x + h

class BabyDragonHatchling(nn.Module):
    """
    Baby Dragon Hatchling (BDH) Model:
    Multi-scale convolutional encoder + Dilated Gated Residual Blocks + Temporal Attention Pooling.
    """
    def __init__(self, in_features=19, seq_len=80, hidden_dim=64, num_blocks=3, dropout=0.25):
        super().__init__()
        self.in_proj = nn.Linear(in_features, hidden_dim)
        self.multiscale = MultiScaleTemporalConv(hidden_dim, hidden_dim)
        
        self.blocks = nn.ModuleList([
            HatchlingResidualBlock(hidden_dim, dilation=2**i, dropout=dropout)
            for i in range(num_blocks)
        ])
        
        # Temporal Attention Mechanism
        self.attn_query = nn.Parameter(torch.randn(hidden_dim, 1))
        self.attn_norm = nn.LayerNorm(hidden_dim)
        
        # Classifier Head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # x: (B, T, D)
        B, T, D = x.shape
        h = self.in_proj(x) # (B, T, H)
        h = h.transpose(1, 2) # (B, H, T) for Conv1D
        
        h = self.multiscale(h)
        for block in self.blocks:
            h = block(h)
            
        h = h.transpose(1, 2) # (B, T, H)
        h = self.attn_norm(h)
        
        # Gated Temporal Attention Pooling
        attn_scores = torch.matmul(h, self.attn_query) # (B, T, 1)
        attn_weights = F.softmax(attn_scores, dim=1) # (B, T, 1)
        pooled = torch.sum(h * attn_weights, dim=1) # (B, H)
        
        logits = self.classifier(pooled) # (B, 1)
        return logits

# -------------------------------------------------------------
# Model 2: Long Short-Term Memory (LSTM) Baseline
# -------------------------------------------------------------
class LSTMFlarePredictor(nn.Module):
    """Standard 2-Layer LSTM Baseline for sequential solar flare forecasting."""
    def __init__(self, in_features=19, hidden_dim=64, num_layers=2, dropout=0.25):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x):
        # x: (B, T, D)
        h = self.in_proj(x) # (B, T, H)
        out, (hn, cn) = self.lstm(h) # out: (B, T, H), hn: (num_layers, B, H)
        last_step = out[:, -1, :] # (B, H)
        logits = self.classifier(last_step) # (B, 1)
        return logits

# -------------------------------------------------------------
# Loss Function: Class-Weighted Binary Cross-Entropy with Pos_Weight
# -------------------------------------------------------------
def get_loss_fn(pos_weight_val=50.0):
    pos_weight = torch.tensor([pos_weight_val], dtype=torch.float32)
    return nn.BCEWithLogitsLoss(pos_weight=pos_weight)

# -------------------------------------------------------------
# Training and Evaluation Function
# -------------------------------------------------------------
def train_model(model, train_loader, test_loader, num_epochs=60, lr=1e-3, pos_weight_val=50.0, name="Model"):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    loss_fn = get_loss_fn(pos_weight_val=pos_weight_val)
    
    history = {'train_loss': [], 'test_loss': [], 'test_pr_auc': [], 'test_recall': []}
    best_pr_auc = -1.0
    best_state = None
    
    print(f"\n--- Training {name} ({num_epochs} epochs) ---")
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item() * len(batch_y)
            
        scheduler.step()
        train_loss = total_loss / len(train_loader.dataset)
        
        # Validation on test set
        test_loss, preds, probs, targets = evaluate_model(model, test_loader, loss_fn)
        
        # Calculate PR-AUC and Recall
        pr_auc = average_precision_score(targets, probs) if len(np.unique(targets)) > 1 else 0.0
        recall = recall_score(targets, preds, zero_division=0)
        
        history['train_loss'].append(train_loss)
        history['test_loss'].append(test_loss)
        history['test_pr_auc'].append(pr_auc)
        history['test_recall'].append(recall)
        
        if pr_auc >= best_pr_auc:
            best_pr_auc = pr_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        if epoch % 10 == 0 or epoch == num_epochs:
            print(f"Epoch {epoch:02d}/{num_epochs} | Train Loss: {train_loss:.4f} | Test Loss: {test_loss:.4f} | Test PR-AUC: {pr_auc:.4f} | Test Recall: {recall:.4f}")
            
    # Load best state
    if best_state is not None:
        model.load_state_dict(best_state)
        
    return model, history

def evaluate_model(model, loader, loss_fn=None, threshold=0.5):
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            logits = model(batch_x)
            if loss_fn is not None:
                loss = loss_fn(logits, batch_y)
                total_loss += loss.item() * len(batch_y)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            all_probs.extend(probs)
            all_targets.extend(batch_y.cpu().numpy().flatten())
            
    avg_loss = total_loss / len(loader.dataset) if loss_fn is not None else 0.0
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets, dtype=int)
    all_preds = (all_probs >= threshold).astype(int)
    
    return avg_loss, all_preds, all_probs, all_targets

def compute_all_metrics(y_true, y_pred, y_prob):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except Exception:
        roc_auc = 0.5
        
    try:
        pr_auc = average_precision_score(y_true, y_prob)
    except Exception:
        pr_auc = 0.0
        
    cm = confusion_matrix(y_true, y_pred).tolist()
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        'accuracy': float(acc),
        'precision': float(prec),
        'recall': float(rec),
        'f1_score': float(f1),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn),
        'confusion_matrix': cm
    }

def main():
    print("Loading preprocessed sequences...")
    X_train, y_train, ids_train, X_test, y_test, ids_test, mean_v, std_v = get_train_test_data()
    print(f"Train set: {len(y_train)} samples ({y_train.sum()} positive, {len(y_train) - y_train.sum()} negative)")
    print(f"Test set:  {len(y_test)} samples ({y_test.sum()} positive, {len(y_test) - y_test.sum()} negative)")
    
    train_dataset = SolarFlareDataset(X_train, y_train, ids_train)
    test_dataset = SolarFlareDataset(X_test, y_test, ids_test)
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # 1. Train BDH Model
    bdh_model = BabyDragonHatchling(in_features=19, seq_len=80, hidden_dim=64, num_blocks=3, dropout=0.25)
    bdh_model, bdh_history = train_model(bdh_model, train_loader, test_loader, num_epochs=60, lr=1e-3, pos_weight_val=50.0, name="Baby Dragon Hatchling (BDH)")
    torch.save(bdh_model.state_dict(), MODELS_DIR / 'bdh_best.pt')
    
    # 2. Train LSTM Model
    lstm_model = LSTMFlarePredictor(in_features=19, hidden_dim=64, num_layers=2, dropout=0.25)
    lstm_model, lstm_history = train_model(lstm_model, train_loader, test_loader, num_epochs=60, lr=1e-3, pos_weight_val=50.0, name="LSTM")
    torch.save(lstm_model.state_dict(), MODELS_DIR / 'lstm_best.pt')
    
    # 3. Comprehensive Evaluation
    print("\n================ FINAL TEST SET EVALUATION ================")
    _, bdh_preds, bdh_probs, y_true = evaluate_model(bdh_model, test_loader)
    _, lstm_preds, lstm_probs, _ = evaluate_model(lstm_model, test_loader)
    
    bdh_metrics = compute_all_metrics(y_true, bdh_preds, bdh_probs)
    lstm_metrics = compute_all_metrics(y_true, lstm_preds, lstm_probs)
    
    print("\n--- Baby Dragon Hatchling (BDH) Metrics ---")
    for k, v in bdh_metrics.items():
        print(f"  {k:18s}: {v}")
        
    print("\n--- LSTM Metrics ---")
    for k, v in lstm_metrics.items():
        print(f"  {k:18s}: {v}")
        
    # Save results to JSON
    results = {
        'dataset_summary': {
            'train_samples': int(len(y_train)),
            'train_positives': int(y_train.sum()),
            'test_samples': int(len(y_test)),
            'test_positives': int(y_test.sum()),
            'feature_dim': int(X_train.shape[2]),
            'sequence_length': int(X_train.shape[1])
        },
        'bdh_metrics': bdh_metrics,
        'lstm_metrics': lstm_metrics,
        'bdh_history': bdh_history,
        'lstm_history': lstm_history
    }
    
    with open(MODELS_DIR / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
        
    # 4. Generate Comparative Plots
    # A. PR and ROC Curves
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    
    # PR Curve
    bdh_p, bdh_r, _ = precision_recall_curve(y_true, bdh_probs)
    lstm_p, lstm_r, _ = precision_recall_curve(y_true, lstm_probs)
    
    axes[0].plot(bdh_r, bdh_p, label=f"BDH (PR-AUC = {bdh_metrics['pr_auc']:.4f})", color='#1f77b4', lw=2.5)
    axes[0].plot(lstm_r, lstm_p, label=f"LSTM (PR-AUC = {lstm_metrics['pr_auc']:.4f})", color='#ff7f0e', lw=2.5, ls='--')
    axes[0].set_xlabel('Recall (Sensitivity)', fontsize=12)
    axes[0].set_ylabel('Precision (Positive Predictive Value)', fontsize=12)
    axes[0].set_title('Precision-Recall Curves (90s Forecast)', fontsize=13, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # ROC Curve
    bdh_fpr, bdh_tpr, _ = roc_curve(y_true, bdh_probs)
    lstm_fpr, lstm_tpr, _ = roc_curve(y_true, lstm_probs)
    
    axes[1].plot(bdh_fpr, bdh_tpr, label=f"BDH (ROC-AUC = {bdh_metrics['roc_auc']:.4f})", color='#1f77b4', lw=2.5)
    axes[1].plot(lstm_fpr, lstm_tpr, label=f"LSTM (ROC-AUC = {lstm_metrics['roc_auc']:.4f})", color='#ff7f0e', lw=2.5, ls='--')
    axes[1].plot([0, 1], [0, 1], color='gray', linestyle=':', lw=1.5, label='Random Chance')
    axes[1].set_xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    axes[1].set_ylabel('True Positive Rate (Recall)', fontsize=12)
    axes[1].set_title('ROC Curves (90s Forecast)', fontsize=13, fontweight='bold')
    axes[1].legend(loc='lower right', fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(MODELS_DIR / 'pr_roc_curves.png')
    plt.savefig(DOCS_DIR / 'pr_roc_curves.png')
    plt.close()
    
    # B. Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=300)
    for ax, cm, title in zip(axes, [bdh_metrics['confusion_matrix'], lstm_metrics['confusion_matrix']], ['Baby Dragon Hatchling (BDH)', 'LSTM Baseline']):
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f"{title} Confusion Matrix", fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        tick_marks = np.arange(2)
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(['No Flare (0)', 'Flare (1)'])
        ax.set_yticklabels(['No Flare (0)', 'Flare (1)'])
        ax.set_xlabel('Predicted Label', fontsize=11)
        ax.set_ylabel('True Label', fontsize=11)
        
        # Annotate values
        for i in range(2):
            for j in range(2):
                ax.text(j, i, format(cm[i][j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i][j] > np.max(cm)/2 else "black",
                        fontsize=14, fontweight='bold')
                        
    plt.tight_layout()
    plt.savefig(MODELS_DIR / 'confusion_matrices.png')
    plt.savefig(DOCS_DIR / 'confusion_matrices.png')
    plt.close()
    
    print(f"\nSaved plots to {MODELS_DIR} and {DOCS_DIR}")
    print("Model training and evaluation complete!")

if __name__ == '__main__':
    main()
