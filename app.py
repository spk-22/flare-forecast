"""
app.py — Flask inference server for Solar Flare Forecasting v3 dashboard.

Two-page dashboard:
  GET  /                    -> Page 1: Solar Flare Prediction (upload ZIP)
  POST /predict             -> Run all 3 models on uploaded ZIP
  GET  /results             -> Page 2: Model Training Results
  GET  /api/results_data    -> JSON for results page
  GET  /static/<path>       -> Static files (plots)

Usage:
  python app.py
  Then open http://localhost:5000
"""

import os, sys, json, zipfile, io, time, traceback, base64, tempfile
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import matplotlib.cm as cm
from flask import Flask, request, jsonify, render_template_string, send_from_directory

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).resolve().parent
MODELS    = ROOT / "models"
DOCS      = ROOT / "documentation"
SEQ_DIR   = ROOT / "Sequences"

# ── Model Architectures (v4 Authentic SUIT & Legacy v3 Fallback) ─────────────
from scripts.rebuild_step4_train_models import (
    BabyDragonHatchlingV4,
    LSTMFlarePredictor as LSTMFlarePredictorV4,
    TemporalFusionTransformer as TemporalFusionTransformerV4
)

# Legacy v3 model architectures (kept for backward compatibility with v3 checkpoints)
class MultiScaleTemporalConvV3(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        third       = out_ch // 3
        rem         = out_ch - 2 * third
        self.conv3  = nn.Conv1d(in_ch, third, 3,  padding=1)
        self.conv7  = nn.Conv1d(in_ch, third, 7,  padding=3)
        self.conv15 = nn.Conv1d(in_ch, rem,   15, padding=7)
        self.norm   = nn.GroupNorm(min(8, out_ch), out_ch)
        self.act    = nn.GELU()
    def forward(self, x):
        return self.act(self.norm(
            torch.cat([self.conv3(x), self.conv7(x), self.conv15(x)], dim=1)))

class GatedResidualBlockV3(nn.Module):
    def __init__(self, ch, dilation=1, dropout=0.30):
        super().__init__()
        self.conv_f = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.conv_g = nn.Conv1d(ch, ch, 3, padding=dilation, dilation=dilation)
        self.norm   = nn.GroupNorm(min(8, ch), ch)
        self.drop   = nn.Dropout(dropout)
        r           = max(ch // 4, 4)
        self.se1    = nn.Linear(ch, r)
        self.se2    = nn.Linear(r, ch)
    def forward(self, x):
        h  = torch.tanh(self.conv_f(x)) * torch.sigmoid(self.conv_g(x))
        h  = self.drop(self.norm(h))
        se = torch.sigmoid(self.se2(F.relu(self.se1(h.mean(-1))))).unsqueeze(-1)
        return x + h * se

class BabyDragonHatchlingV3(nn.Module):
    def __init__(self, in_features, hidden_dim=64, num_blocks=3, dropout=0.30):
        super().__init__()
        self.in_proj    = nn.Sequential(nn.Linear(in_features, hidden_dim),
                                        nn.LayerNorm(hidden_dim), nn.GELU())
        self.multiscale = MultiScaleTemporalConvV3(hidden_dim, hidden_dim)
        self.blocks     = nn.ModuleList([
            GatedResidualBlockV3(hidden_dim, dilation=2**i, dropout=dropout)
            for i in range(num_blocks)])
        self.attn_q     = nn.Linear(hidden_dim, 1, bias=False)
        self.attn_n     = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 32), nn.GELU(),
                                        nn.Dropout(dropout), nn.Linear(32, 1))
    def forward(self, x):
        h      = self.in_proj(x).transpose(1, 2)
        h      = self.multiscale(h)
        for blk in self.blocks: h = blk(h)
        h      = h.transpose(1, 2)
        attn   = torch.softmax(self.attn_q(self.attn_n(h)), dim=1)
        pooled = (h * attn).sum(dim=1)
        return self.classifier(pooled)

class LSTMFlarePredictorV3(nn.Module):
    def __init__(self, in_features, hidden_dim=64, num_layers=2, dropout=0.25):
        super().__init__()
        self.in_proj    = nn.Sequential(nn.Linear(in_features, hidden_dim),
                                        nn.LayerNorm(hidden_dim), nn.GELU())
        self.lstm       = nn.LSTM(hidden_dim, hidden_dim, num_layers,
                                  dropout=dropout if num_layers > 1 else 0.0,
                                  batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 32), nn.GELU(),
                                        nn.Dropout(dropout), nn.Linear(32, 1))
    def forward(self, x):
        out, _ = self.lstm(self.in_proj(x))
        return self.classifier(out[:, -1, :])

class GatedResidualNetworkTFTV3(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.1):
        super().__init__()
        self.fc1   = nn.Linear(input_dim, hidden_dim)
        self.fc2   = nn.Linear(hidden_dim, output_dim * 2)
        self.elu   = nn.ELU()
        self.drop  = nn.Dropout(dropout)
        self.norm  = nn.LayerNorm(output_dim)
        self.skip  = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
    def forward(self, x):
        h      = self.drop(self.fc2(self.elu(self.fc1(x))))
        h1, h2 = h.chunk(2, dim=-1)
        return self.norm(h1 * torch.sigmoid(h2) + self.skip(x))

class VariableSelectionNetworkV3(nn.Module):
    def __init__(self, num_features, hidden_dim, dropout=0.1):
        super().__init__()
        self.grn_per_feat = nn.ModuleList([
            GatedResidualNetworkTFTV3(1, hidden_dim, hidden_dim, dropout)
            for _ in range(num_features)])
        self.grn_weights  = GatedResidualNetworkTFTV3(num_features, hidden_dim, num_features, dropout)
    def forward(self, x):
        B, T, F = x.shape
        feat_embeds = []
        for i, grn in enumerate(self.grn_per_feat):
            fi = x[:, :, i:i+1]
            feat_embeds.append(grn(fi.reshape(B*T, 1)).reshape(B, T, -1))
        feat_stack = torch.stack(feat_embeds, dim=-2)
        weights    = torch.softmax(self.grn_weights(x.reshape(B*T, F)), dim=-1)
        return (feat_stack * weights.reshape(B, T, F, 1)).sum(dim=-2)

class TemporalFusionTransformerV3(nn.Module):
    def __init__(self, num_features, hidden_dim=64, lstm_layers=2, num_heads=4, dropout=0.25):
        super().__init__()
        self.vsn        = VariableSelectionNetworkV3(num_features, hidden_dim, dropout)
        self.lstm       = nn.LSTM(hidden_dim, hidden_dim, lstm_layers,
                                  batch_first=True,
                                  dropout=dropout if lstm_layers > 1 else 0.0)
        self.lstm_norm  = nn.LayerNorm(hidden_dim)
        self.attn       = nn.MultiheadAttention(hidden_dim, num_heads,
                                                dropout=dropout, batch_first=True)
        self.attn_norm  = nn.LayerNorm(hidden_dim)
        self.post_grn   = GatedResidualNetworkTFTV3(hidden_dim, hidden_dim, hidden_dim, dropout)
        self.post_norm  = nn.LayerNorm(hidden_dim)
        self.pool_q     = nn.Linear(hidden_dim, 1, bias=False)
        self.classifier = nn.Sequential(nn.Linear(hidden_dim, 32), nn.GELU(),
                                        nn.Dropout(dropout), nn.Linear(32, 1))
    def forward(self, x):
        h           = self.vsn(x)
        lstm_out, _ = self.lstm(h)
        lstm_out    = self.lstm_norm(lstm_out + h)
        attn_out, _ = self.attn(lstm_out, lstm_out, lstm_out)
        attn_out    = self.attn_norm(attn_out + lstm_out)
        B, T, D     = attn_out.shape
        post        = self.post_grn(attn_out.reshape(B*T, D)).reshape(B, T, D)
        post        = self.post_norm(post + attn_out)
        weights     = torch.softmax(self.pool_q(post), dim=1)
        pooled      = (post * weights).sum(dim=1)
        return self.classifier(pooled)


# ── Feature Extraction from FITS ──────────────────────────────────────────────

FEAT_NAMES_19 = [
    'mean_dn','std_dn','min_dn','max_dn',
    'p01_dn','p10_dn','p50_dn','p90_dn','p95_dn','p99_dn','p999_dn',
    'skew_dn','kurt_dn','enhancement_ratio','bright_pix_frac',
    'quality_percent','exposure_s',
    'hpc_tx_center_arcsec','hpc_ty_center_arcsec',
]
FEAT_NAMES_17 = [f for f in FEAT_NAMES_19
                  if f not in ('quality_percent', 'exposure_s')]

SEQ_LEN = 80  # number of frames required


def _safe_float(val, default=0.0):
    try:
        if val in (None, '', 'NA', 'NaN', 'null'):
            return float(default)
        return float(val)
    except (ValueError, TypeError):
        return float(default)


def extract_features_from_fits(fits_path):
    """Extract 19 features from one FITS file. Returns dict or raises."""
    try:
        from astropy.io import fits as astrofits
        from scipy.stats import skew, kurtosis
        from astropy.wcs import WCS
    except ImportError as e:
        raise ImportError(f"Missing dependency: {e}. Install astropy and scipy.")

    with astrofits.open(fits_path, memmap=False) as hdul:
        hdr  = hdul[0].header
        data = hdul[0].data

    if data is None:
        raise ValueError("FITS file has no image data.")

    img = data.astype(np.float32)
    flat = img.flatten()
    flat_valid = flat[np.isfinite(flat)]
    if len(flat_valid) == 0:
        raise ValueError("No valid pixel data.")

    # Percentiles
    pcts = np.percentile(flat_valid, [1, 10, 50, 90, 95, 99, 99.9])
    p01, p10, p50, p90, p95, p99, p999 = pcts

    mean_v = float(np.mean(flat_valid))
    std_v  = float(np.std(flat_valid)) if np.std(flat_valid) > 0 else 1.0

    enh_ratio   = float((p99 - p50) / std_v) if std_v > 0 else 0.0
    bright_frac = float(np.mean(flat_valid > (p50 + 2 * std_v)))

    # WCS centroid
    try:
        wcs = WCS(hdr, naxis=2)
        ny, nx = img.shape[-2], img.shape[-1]
        cx, cy = nx / 2, ny / 2
        world  = wcs.all_pix2world([[cx, cy]], 0)
        tx     = float(world[0][0]) * 3600  # deg -> arcsec
        ty     = float(world[0][1]) * 3600
    except Exception:
        tx = _safe_float(hdr.get('CRVAL1'), 0.0) * 3600
        ty = _safe_float(hdr.get('CRVAL2'), 0.0) * 3600

    return {
        'mean_dn':              mean_v,
        'std_dn':               std_v,
        'min_dn':               float(flat_valid.min()),
        'max_dn':               float(flat_valid.max()),
        'p01_dn':               float(p01),
        'p10_dn':               float(p10),
        'p50_dn':               float(p50),
        'p90_dn':               float(p90),
        'p95_dn':               float(p95),
        'p99_dn':               float(p99),
        'p999_dn':              float(p999),
        'skew_dn':              float(skew(flat_valid)),
        'kurt_dn':              float(kurtosis(flat_valid)),
        'enhancement_ratio':    enh_ratio,
        'bright_pix_frac':      bright_frac,
        'quality_percent':      _safe_float(hdr.get('QVAL'), 100.0),
        'exposure_s':           _safe_float(hdr.get('EXPTIME'), 0.0),
        'hpc_tx_center_arcsec': tx,
        'hpc_ty_center_arcsec': ty,
    }


def get_fits_timestamp(fits_path):
    """Return the observation timestamp from a FITS header."""
    try:
        from astropy.io import fits as astrofits
        with astrofits.open(fits_path, memmap=False) as hdul:
            hdr = hdul[0].header
        for key in ['DATE-BEG', 'DATE-OBS', 'T_OBS']:
            if key in hdr:
                return str(hdr[key])
    except Exception:
        pass
    return str(fits_path.stem)  # fallback: filename


def generate_fits_thumbnail(fits_path, size=(160, 160)):
    """Generate a solar chromosphere base64 PNG thumbnail from a FITS file."""
    try:
        from astropy.io import fits as astrofits
        with astrofits.open(fits_path, memmap=False) as hdul:
            data = hdul[0].data
        if data is None:
            return None
        data = data.astype(np.float32)
        valid = data[np.isfinite(data)]
        if len(valid) == 0:
            return None
        p1 = float(np.percentile(valid, 1))
        p99 = float(np.percentile(valid, 99.5))
        denom = max(p99 - p1, 1e-4)
        norm = np.clip((data - p1) / denom, 0.0, 1.0)
        colored = (cm.magma(norm)[:, :, :3] * 255).astype(np.uint8)
        img = Image.fromarray(colored)
        img.thumbnail(size)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return None


# ── Model Loading ─────────────────────────────────────────────────────────────

_model_cache = {}
_norm_params  = None
_NF           = 18
_HIDDEN       = 64

def _load_norm_params():
    global _norm_params
    if _norm_params is not None:
        return _norm_params
    p = MODELS / "preprocessing_params_v4.npz"
    if not p.exists():
        p = MODELS / "preprocessing_params_v3.npz"
    if not p.exists():
        raise FileNotFoundError(
            f"Preprocessing parameters not found at {p}. Run training script first.")
    d = np.load(p)
    _norm_params = {
        'mu':        d['mu'].astype(np.float32),
        'sig':       d['sig'].astype(np.float32),
        'keep_mask': d['keep_mask'].astype(bool),
    }
    return _norm_params


def _load_model(model_name, experiment="v4"):
    """Load a trained model from checkpoint (v4 authentic or v3 fallback)."""
    key = f"{model_name}_{experiment}"
    if key in _model_cache:
        return _model_cache[key]

    norm = _load_norm_params()
    nf = int(norm['keep_mask'].sum())

    ckpt_v4 = MODELS / f"{model_name.lower()}_v4.pt"
    if ckpt_v4.exists():
        if model_name == "BDH":
            model = BabyDragonHatchlingV4(nf, _HIDDEN, num_blocks=3, dropout=0.30)
        elif model_name == "LSTM":
            model = LSTMFlarePredictorV4(nf, _HIDDEN, num_layers=2, dropout=0.25)
        elif model_name == "TFT":
            model = TemporalFusionTransformerV4(nf, _HIDDEN, num_heads=4, dropout=0.20)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        model.load_state_dict(torch.load(ckpt_v4, map_location="cpu", weights_only=True))
        model.eval()
        _model_cache[key] = model
        return model

    # Fallback to v3
    ckpt_v3 = MODELS / f"{model_name.lower()}_v3_{experiment}.pt"
    if not ckpt_v3.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_v4} or {ckpt_v3}")

    if model_name == "BDH":
        model = BabyDragonHatchlingV3(_NF, _HIDDEN, num_blocks=3, dropout=0.30)
    elif model_name == "LSTM":
        model = LSTMFlarePredictorV3(_NF, _HIDDEN, num_layers=2, dropout=0.25)
    elif model_name == "TFT":
        model = TemporalFusionTransformerV3(_NF, _HIDDEN, num_heads=4, dropout=0.25)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.load_state_dict(torch.load(ckpt_v3, map_location="cpu", weights_only=True))
    model.eval()
    _model_cache[key] = model
    return model


def predict_sequence(X_norm):
    """
    Run all 3 models on a normalized sequence tensor.
    X_norm: (1, 80, 18) or (1, 80, 17) numpy float32
    Returns dict with probabilities and predictions.
    """
    x = torch.tensor(X_norm, dtype=torch.float32)

    results = {}
    thresholds = {"BDH": 0.20, "LSTM": 0.677, "TFT": 0.417}

    for mname in ["BDH", "LSTM", "TFT"]:
        try:
            model = _load_model(mname, "v4")
            experiment_used = "v4_authentic"
        except FileNotFoundError:
            try:
                model = _load_model(mname, "aug")
                experiment_used = "v3_aug"
            except FileNotFoundError:
                results[mname] = {"error": "Model not loaded yet. Train first."}
                continue

        with torch.no_grad():
            logit = model(x)
            prob  = float(torch.sigmoid(logit).item())

        th = thresholds.get(mname, 0.20)
        results[mname] = {
            "probability": round(prob, 4),
            "prediction":  "FLARE LIKELY" if prob >= th else "No Flare",
            "confidence":  "High" if prob > 0.6 else ("Medium" if prob >= th else "Low"),
            "experiment":  experiment_used,
            "threshold":   th,
        }
    return results


# ── ZIP Processing Pipeline ───────────────────────────────────────────────────

def process_zip(zip_bytes):
    """
    Full inference pipeline from raw ZIP bytes to model predictions.
    Returns: dict with predictions and metadata, or error info.
    """
    errors = []
    warnings = []

    # 1. Extract ZIP
    try:
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    except zipfile.BadZipFile:
        return {"error": "Invalid ZIP file. Please upload a valid ZIP archive."}

    all_names = zf.namelist()
    fits_names = sorted([
        n for n in all_names
        if n.lower().endswith('.fits') and 'NB04' in n
    ])

    if not fits_names:
        # Try without filter constraint
        fits_names = sorted([n for n in all_names if n.lower().endswith('.fits')])
        if fits_names:
            warnings.append(
                f"No NB04 (Mg II h) FITS files found. Using all {len(fits_names)} "
                f"FITS files. Results may not be meaningful without the correct filter.")
        else:
            return {"error": f"No FITS files found in ZIP. Contents: {all_names[:10]}"}

    if len(fits_names) < SEQ_LEN:
        return {
            "error": (f"Insufficient observations: found {len(fits_names)} FITS files, "
                      f"need at least {SEQ_LEN} for a {SEQ_LEN}-frame sequence. "
                      f"Upload at least 90 minutes of continuous NB04 observations.")
        }

    # 2. Use the last SEQ_LEN frames
    selected = fits_names[-SEQ_LEN:]
    warnings_extract = []

    # Pick 4 evenly spread timeline frames
    total_sel = len(selected)
    tl_indices = [0, int(total_sel * 0.33), int(total_sel * 0.66), total_sel - 1]
    timeline_frames = []

    # 3. Extract features
    features_list = []
    timestamps    = []
    temp_dir      = Path(tempfile.gettempdir())

    for idx, fname in enumerate(selected):
        tmp_path = temp_dir / f"_suit_{idx}_{fname.replace('/', '_').replace('\\', '_')}"
        try:
            raw_bytes = zf.read(fname)
            with open(tmp_path, "wb") as f:
                f.write(raw_bytes)
            feats = extract_features_from_fits(tmp_path)
            ts    = get_fits_timestamp(tmp_path)
            features_list.append(feats)
            timestamps.append(ts)

            # Generate thumbnail if one of the 4 timeline frames
            if idx in tl_indices:
                b64_img = generate_fits_thumbnail(tmp_path)
                ts_clean = ts.replace('T', ' ').split('.')[0] if ts else 'Unknown'
                parts = ts_clean.split(' ')
                d_str = parts[0] if len(parts) > 0 else 'Unknown'
                t_str = (parts[1] + ' UTC') if len(parts) > 1 else 'UTC'
                lbl = "Window Start" if idx == 0 else ("Pre-Flare Horizon (T-0)" if idx == total_sel - 1 else f"Frame {idx+1}")
                timeline_frames.append({
                    "frame_index": idx + 1,
                    "label": lbl,
                    "date": d_str,
                    "time": t_str,
                    "timestamp": ts,
                    "image_b64": b64_img
                })

            try: tmp_path.unlink()
            except: pass
        except Exception as e:
            warnings_extract.append(f"{fname}: {e}")
            features_list.append(None)
            timestamps.append("unknown")
            try: tmp_path.unlink()
            except: pass

    valid_mask = [f is not None for f in features_list]
    n_valid = sum(valid_mask)
    if n_valid < SEQ_LEN:
        return {"error": (f"Feature extraction failed for too many frames "
                          f"({SEQ_LEN - n_valid} failures). "
                          f"Minimum required: {SEQ_LEN} valid frames.")}

    # Fill failed frames with interpolated values from neighbors
    feat_arr_full = []
    for i, feat in enumerate(features_list):
        if feat is not None:
            feat_arr_full.append([feat[k] for k in FEAT_NAMES_19])
        else:
            # Forward fill
            prev_valid = next((features_list[j] for j in range(i-1, -1, -1)
                               if features_list[j] is not None), None)
            if prev_valid:
                feat_arr_full.append([prev_valid[k] for k in FEAT_NAMES_19])
            else:
                feat_arr_full.append([0.0] * 19)

    X_raw = np.array(feat_arr_full, dtype=np.float32)  # (80, 19)

    # 4. Apply preprocessing (drop constant features + normalize)
    norm = _load_norm_params()
    keep_mask = norm['keep_mask']
    X_kept = X_raw[:, keep_mask]                        # (80, 17)

    mu, sig = norm['mu'], norm['sig']
    X_norm = np.clip((X_kept - mu) / sig, -5.0, 5.0)
    X_norm = np.nan_to_num(X_norm, nan=0.0, posinf=0.0, neginf=0.0)
    X_norm = X_norm[np.newaxis, :, :]                   # (1, 80, 17)

    # 5. Run models
    t0 = time.time()
    model_results = predict_sequence(X_norm)
    inference_ms  = round((time.time() - t0) * 1000, 1)

    # 6. Build response
    return {
        "status":          "ok",
        "n_fits_found":    len(fits_names),
        "n_fits_used":     len(selected),
        "n_valid_frames":  n_valid,
        "timestamps": {
            "first": timestamps[0],
            "last":  timestamps[-1],
        },
        "frame_timestamps":  timestamps,
        "timeline_frames":   timeline_frames,
        "predictions":       model_results,
        "inference_ms":      inference_ms,
        "warnings":          warnings + warnings_extract,
        "prediction_horizon_seconds": 90,
        "sequence_length_frames":     SEQ_LEN,
    }


# ── Results Data ──────────────────────────────────────────────────────────────

def load_results_json():
    """Load model_comparison_v4.json. Falls back to v3 or v2 if not found."""
    for fname in ["model_comparison_v4.json", "model_comparison_v3.json", "model_comparison_v2.json"]:
        p = MODELS / fname
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data, fname
    return None, None


# ── Flask App ─────────────────────────────────────────────────────────────────

app = Flask(__name__, static_folder=str(DOCS), static_url_path="/docs")

@app.route("/")
def index():
    """Serve the main two-page dashboard."""
    return render_template_string(DASHBOARD_HTML)

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded. Send a ZIP as 'file' field."}), 400
    f = request.files["file"]
    if not f.filename.lower().endswith(".zip"):
        return jsonify({"error": "Only .zip files are accepted."}), 400
    try:
        norm = _load_norm_params()
    except FileNotFoundError as e:
        return jsonify({"error": str(e)}), 503

    zip_bytes = f.read()
    result = process_zip(zip_bytes)
    return jsonify(result)

@app.route("/api/results_data")
def results_data():
    data, source = load_results_json()
    if data is None:
        return jsonify({"error": "No results found. Run training script first."}), 404

    # Format v4 summary structure if loaded
    if isinstance(data, dict) and "BDH" in data and "test_pr_auc" in data["BDH"]:
        v4_metrics = {}
        for m in ["BDH", "LSTM", "TFT"]:
            info = data.get(m, {})
            cm = info.get("confusion_matrix", [[0, 0], [0, 0]])
            v4_metrics[m] = {
                "pr_auc": info.get("test_pr_auc"),
                "roc_auc": info.get("test_roc_auc"),
                "optimal_f1_threshold": {
                    "f1": info.get("test_f1"),
                    "precision": info.get("test_precision"),
                    "recall": info.get("test_recall"),
                    "fp": cm[0][1],
                    "fn": cm[1][0]
                },
                "n_params": 119346 if m == "BDH" else (70081 if m == "LSTM" else 237577)
            }
        data = {
            "v4_stats": {
                "train_samples": 1778,
                "train_positives": 28,
                "train_flares": 19,
                "val_samples": 518,
                "val_positives": 8,
                "val_flares": 7,
                "test_samples": 394,
                "test_positives": 6,
                "test_flares": 6,
                "total_sequences": 2690,
                "total_positives": 42,
                "total_flares": 32,
            },
            "original": {
                "train_samples": 1778,
                "train_positives": 28,
                "test_samples": 394,
                "test_positives": 6,
                "metrics": v4_metrics
            },
            "augmented": {
                "train_samples": 1778,
                "train_positives": 28,
                "test_samples": 394,
                "test_positives": 6,
                "n_synth": 0,
                "metrics": v4_metrics
            },
            "augmentation": {
                "n_synth": 0,
                "n_flare_events": 32,
                "method": "Zero synthetic augmentation — 100% authentic observational data"
            }
        }

    # Attach plot URLs
    plots = {}
    for fname in DOCS.glob("*.png"):
        plots[fname.stem] = f"/docs/{fname.name}"

    return jsonify({"results": data, "plots": plots, "source": source})

@app.route("/docs/<path:filename>")
def serve_doc(filename):
    return send_from_directory(str(DOCS), filename)

@app.route("/health")
def health():
    return jsonify({"status": "ok", "time": datetime.now(timezone.utc).isoformat()})


# ═══════════════════════════════════════════════════════════════════════════════
# EMBEDDED DASHBOARD HTML
# ═══════════════════════════════════════════════════════════════════════════════

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Flare-Forecast v4 | Aditya-L1 SUIT</title>
<style>
  :root {
    --bg:       #0a0f1e;
    --surface:  #111827;
    --surface2: #1e293b;
    --border:   #334155;
    --text:     #e2e8f0;
    --muted:    #94a3b8;
    --bdh:      #3b82f6;
    --lstm:     #f97316;
    --tft:      #22c55e;
    --danger:   #ef4444;
    --warn:     #f59e0b;
    --success:  #10b981;
    --radius:   12px;
    --shadow:   0 4px 24px rgba(0,0,0,0.5);
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; min-height: 100vh; }

  /* NAV */
  nav { background: var(--surface); border-bottom: 1px solid var(--border); padding: 0 2rem; display: flex; align-items: center; gap: 2rem; height: 60px; position: sticky; top: 0; z-index: 100; }
  nav .logo { font-weight: 700; font-size: 1.1rem; color: var(--bdh); letter-spacing: .5px; }
  nav .logo span { color: var(--text); }
  nav .tab { padding: .5rem 1rem; border-radius: 8px; cursor: pointer; color: var(--muted); font-size: .9rem; transition: all .2s; border: none; background: none; }
  nav .tab.active, nav .tab:hover { background: var(--surface2); color: var(--text); }
  nav .badge { margin-left: auto; font-size: .75rem; padding: .2rem .6rem; border-radius: 20px; background: var(--surface2); color: var(--muted); }

  /* PAGE */
  .page { display: none; padding: 2rem; max-width: 1200px; margin: 0 auto; }
  .page.active { display: block; }

  /* HERO */
  .hero { text-align: center; padding: 2.5rem 1rem; }
  .hero h1 { font-size: 2rem; font-weight: 700; margin-bottom: .5rem; }
  .hero p { color: var(--muted); max-width: 600px; margin: .5rem auto; line-height: 1.6; }

  /* CARDS */
  .card { background: var(--surface); border: 1px solid var(--border); border-radius: var(--radius); padding: 1.5rem; margin-bottom: 1.5rem; box-shadow: var(--shadow); }
  .card h2 { font-size: 1.1rem; margin-bottom: 1rem; display: flex; align-items: center; gap: .5rem; }

  /* UPLOAD ZONE */
  .upload-zone { border: 2px dashed var(--border); border-radius: var(--radius); padding: 3rem 2rem; text-align: center; cursor: pointer; transition: all .3s; position: relative; }
  .upload-zone:hover, .upload-zone.drag { border-color: var(--bdh); background: rgba(59,130,246,.05); }
  .upload-zone input[type=file] { position: absolute; inset: 0; opacity: 0; cursor: pointer; }
  .upload-zone .icon { font-size: 3rem; margin-bottom: 1rem; }
  .upload-zone .hint { color: var(--muted); font-size: .85rem; margin-top: .5rem; }

  /* TIMELINE */
  .timeline-strip { display: flex; gap: 1rem; justify-content: center; align-items: flex-end; padding: 1rem; overflow-x: auto; }
  .timeline-frame { display: flex; flex-direction: column; align-items: center; gap: .5rem; min-width: 140px; }
  .timeline-frame .frame-img { width: 130px; height: 130px; background: var(--surface2); border-radius: 8px; border: 1px solid var(--border); display: flex; align-items: center; justify-content: center; color: var(--muted); font-size: .75rem; text-align: center; overflow: hidden; padding: .5rem; }
  .timeline-frame .frame-label { font-size: .7rem; color: var(--muted); text-align: center; }
  .timeline-frame .frame-time { font-size: .65rem; color: var(--bdh); font-family: monospace; }
  .timeline-connector { flex: 1; height: 2px; background: linear-gradient(to right, var(--border), var(--bdh), var(--border)); min-width: 20px; align-self: center; margin-bottom: 2rem; }

  /* PREDICTION TABLE */
  .pred-table { width: 100%; border-collapse: collapse; }
  .pred-table th { background: var(--surface2); padding: .75rem 1rem; text-align: left; font-size: .85rem; color: var(--muted); border-bottom: 1px solid var(--border); }
  .pred-table td { padding: .75rem 1rem; border-bottom: 1px solid var(--border); font-size: .9rem; }
  .pred-table tr:last-child td { border: none; }
  .pill { display: inline-block; padding: .25rem .75rem; border-radius: 20px; font-size: .8rem; font-weight: 600; }
  .pill.flare { background: rgba(239,68,68,.2); color: #f87171; border: 1px solid rgba(239,68,68,.4); }
  .pill.no-flare { background: rgba(16,185,129,.2); color: #34d399; border: 1px solid rgba(16,185,129,.4); }
  .prob-bar { display: flex; align-items: center; gap: .75rem; }
  .prob-bg { flex: 1; height: 8px; background: var(--surface2); border-radius: 4px; overflow: hidden; }
  .prob-fill { height: 100%; border-radius: 4px; transition: width .5s ease; }

  /* MODEL DOTS */
  .model-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
  .dot-bdh  { background: var(--bdh); }
  .dot-lstm { background: var(--lstm); }
  .dot-tft  { background: var(--tft); }

  /* AGREEMENT BADGE */
  .agreement { display: flex; align-items: center; gap: 1rem; padding: 1rem 1.5rem; border-radius: 8px; margin-top: 1rem; }
  .agreement.agree    { background: rgba(16,185,129,.1); border: 1px solid rgba(16,185,129,.3); }
  .agreement.disagree { background: rgba(245,158,11,.1); border: 1px solid rgba(245,158,11,.3); }

  /* STATUS & WARNINGS */
  .status-row { display: flex; align-items: center; gap: .75rem; padding: .75rem 1rem; border-radius: 8px; margin-top: .5rem; font-size: .85rem; }
  .status-row.ok   { background: rgba(16,185,129,.08); border: 1px solid rgba(16,185,129,.2); }
  .status-row.warn { background: rgba(245,158,11,.08); border: 1px solid rgba(245,158,11,.2); }
  .status-row.err  { background: rgba(239,68,68,.08);  border: 1px solid rgba(239,68,68,.2); }

  /* RESULTS PAGE — GRIDS */
  .stats-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 1rem; }
  .stat-box { background: var(--surface2); border-radius: 8px; padding: 1rem; text-align: center; }
  .stat-box .val { font-size: 1.8rem; font-weight: 700; }
  .stat-box .lbl { color: var(--muted); font-size: .78rem; margin-top: .25rem; }
  .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1.5rem; }
  .model-card { border-radius: var(--radius); padding: 1.25rem; border: 2px solid transparent; }
  .model-card.bdh  { border-color: var(--bdh);  background: rgba(59,130,246,.06); }
  .model-card.lstm { border-color: var(--lstm); background: rgba(249,115,22,.06); }
  .model-card.tft  { border-color: var(--tft);  background: rgba(34,197,94,.06);  }
  .model-card h3 { font-size: 1rem; margin-bottom: .75rem; }
  .metric-row { display: flex; justify-content: space-between; padding: .3rem 0; border-bottom: 1px solid var(--border); font-size: .88rem; }
  .metric-row:last-child { border: none; }
  .metric-row .val { font-weight: 600; }

  /* PLOT GALLERY */
  .plot-tabs { display: flex; gap: .5rem; margin-bottom: 1rem; flex-wrap: wrap; }
  .plot-tab { padding: .4rem .9rem; border-radius: 8px; cursor: pointer; font-size: .82rem; border: 1px solid var(--border); background: var(--surface2); color: var(--muted); transition: all .2s; }
  .plot-tab.active, .plot-tab:hover { border-color: var(--bdh); color: var(--text); }
  .plot-pane { display: none; }
  .plot-pane.active { display: block; }
  .plot-pane img { width: 100%; border-radius: 8px; border: 1px solid var(--border); }

  /* SPINNER */
  .spinner { width: 40px; height: 40px; border: 3px solid var(--border); border-top-color: var(--bdh); border-radius: 50%; animation: spin 1s linear infinite; margin: 2rem auto; }
  @keyframes spin { to { transform: rotate(360deg); } }

  /* BUTTON */
  .btn { padding: .6rem 1.5rem; border-radius: 8px; border: none; cursor: pointer; font-size: .9rem; font-weight: 600; transition: all .2s; }
  .btn-primary { background: var(--bdh); color: white; }
  .btn-primary:hover { background: #2563eb; }
  .btn-sm { padding: .35rem .9rem; font-size: .8rem; }

  /* DISCLAIMER */
  .disclaimer { background: rgba(245,158,11,.08); border: 1px solid rgba(245,158,11,.3); border-radius: 8px; padding: 1rem 1.25rem; font-size: .82rem; color: var(--muted); line-height: 1.6; }

  /* COMPARISON TABLE */
  .cmp-table { width: 100%; border-collapse: collapse; font-size: .88rem; }
  .cmp-table th { background: var(--surface2); padding: .6rem .9rem; text-align: left; color: var(--muted); border-bottom: 1px solid var(--border); }
  .cmp-table td { padding: .6rem .9rem; border-bottom: 1px solid var(--border); }
  .cmp-table tr:last-child td { border: none; }
  .cmp-table .best { color: var(--success); font-weight: 700; }
  .winner-badge { display: inline-block; font-size: .7rem; padding: .1rem .5rem; border-radius: 10px; background: rgba(16,185,129,.2); color: var(--success); margin-left: .4rem; }

  /* RESPONSIVE */
  @media (max-width: 700px) { .hero h1 { font-size: 1.5rem; } .timeline-strip { flex-direction: column; align-items: center; } .timeline-connector { width: 2px; height: 20px; } }
</style>
</head>
<body>

<!-- NAVIGATION -->
<nav>
  <div class="logo">&#9728; Flare<span>-Forecast</span></div>
  <button class="tab active" onclick="showPage('predict')">&#128293; Solar Flare Prediction</button>
  <button class="tab"        onclick="showPage('results')">&#128202; Model Training Results</button>
  <span class="badge">Aditya-L1 SUIT &bull; v4 (Authentic Observational Data)</span>
</nav>

<!-- ===== PAGE 1: PREDICTION ===== -->
<div class="page active" id="page-predict">
  <div class="hero">
    <h1>Solar Flare Prediction</h1>
    <p>Upload a ZIP archive containing SUIT NB04 (Mg II h, 2803&nbsp;&Aring;) FITS observations.
       All three models (BDH v4, LSTM, TFT) will independently predict whether a flare will occur in the next ~90&nbsp;seconds.</p>
  </div>

  <!-- Upload -->
  <div class="card">
    <h2>&#128229; Upload SUIT Observation Archive</h2>
    <div class="upload-zone" id="drop-zone">
      <input type="file" accept=".zip" id="zip-input" onchange="handleFile(this.files[0])">
      <div class="icon">&#128230;</div>
      <div>Drop a <strong>.zip</strong> file here or click to browse</div>
      <div class="hint">ZIP should contain at least 80 NB04 FITS files (~90 minutes of observations)</div>
    </div>
    <div id="upload-status" style="margin-top:.75rem;"></div>
  </div>

  <!-- Timeline (shown after upload) -->
  <div class="card" id="timeline-card" style="display:none">
    <h2>&#127959; Observation Timeline</h2>
    <div class="timeline-strip" id="timeline-strip"><!-- populated by JS --></div>
    <div style="text-align:center; color:var(--muted); font-size:.8rem; margin-top:.5rem;" id="timeline-info"></div>
  </div>

  <!-- Prediction Results -->
  <div class="card" id="pred-card" style="display:none">
    <h2>&#128270; Prediction Results</h2>
    <div style="margin-bottom:1rem; display:flex; gap:1.5rem; flex-wrap:wrap; font-size:.82rem; color:var(--muted);" id="pred-meta"></div>

    <table class="pred-table" id="pred-table">
      <thead>
        <tr>
          <th>Model</th>
          <th>Prediction</th>
          <th>Probability</th>
          <th>Confidence</th>
        </tr>
      </thead>
      <tbody id="pred-tbody"></tbody>
    </table>

    <div id="agreement-box"></div>
    <div id="warnings-box" style="margin-top:1rem;"></div>
  </div>

  <!-- Disclaimer -->
  <div class="disclaimer">
    <strong>&#9888; Research prototype disclaimer:</strong> Trained on authentic Aditya-L1 SUIT NB04 (2803&nbsp;&Aring;) observations across 32 independent flare events (spanning 2024–2026) with zero synthetic augmentation.
    Due to natural solar flare rarity (1.52% positive prevalence), models are calibrated for high-recall early warning (~90-second horizon before flare onset).
    Predictions are experimental and intended for solar physics research.
  </div>
</div>

<!-- ===== PAGE 2: RESULTS ===== -->
<div class="page" id="page-results">
  <div class="hero">
    <h1>Model Training Results</h1>
    <p>Benchmark of BDH v4, LSTM, and TFT trained from scratch on 2,690 authentic SUIT observation sequences across 32 independent flare events. Strict chronological event-aware split with zero synthetic augmentation.</p>
  </div>

  <div id="results-loading" style="text-align:center; padding:3rem;">
    <div class="spinner"></div>
    <div style="color:var(--muted); margin-top:1rem;">Loading results...</div>
  </div>

  <div id="results-content" style="display:none">
    <!-- Dataset stats -->
    <div class="card">
      <h2>&#128202; Dataset Summary</h2>
      <div style="display:grid; grid-template-columns:1fr 1fr; gap:1.5rem; flex-wrap:wrap;">
        <div>
          <div id="stats-orig-title" style="font-size:.85rem; color:var(--muted); margin-bottom:.75rem; font-weight:600;">TRAIN & VAL SETS (AUTHENTIC SUIT)</div>
          <div class="stats-grid" id="stats-orig"></div>
        </div>
        <div>
          <div id="stats-aug-title" style="font-size:.85rem; color:var(--muted); margin-bottom:.75rem; font-weight:600;">INDEPENDENT TEST SET & TOTALS</div>
          <div class="stats-grid" id="stats-aug"></div>
        </div>
      </div>
      <div style="margin-top:1rem; padding:1rem; background:var(--surface2); border-radius:8px; font-size:.82rem; color:var(--muted); line-height:1.6;" id="aug-note"></div>
    </div>

    <!-- Experiment selector -->
    <div class="card">
      <h2>&#128200; Model Performance Comparison</h2>
      <div style="margin-bottom:1rem; display:flex; gap:.5rem; align-items:center;">
        <span style="color:var(--muted); font-size:.85rem;">Evaluation Split:</span>
        <button class="btn btn-primary btn-sm" id="btn-orig" onclick="showExp('original')">v4 Authentic SUIT</button>
      </div>
      <div class="metrics-grid" id="metrics-grid"></div>
    </div>

    <!-- Comparison table -->
    <div class="card">
      <h2>&#128203; Full Comparison Table</h2>
      <div style="overflow-x:auto;" id="cmp-table-wrap"></div>
    </div>

    <!-- Plot gallery -->
    <div class="card">
      <h2>&#127912; Plots</h2>
      <div class="plot-tabs" id="plot-tabs"></div>
      <div id="plot-panes"></div>
    </div>
  </div>

  <div class="disclaimer" style="margin-top:1.5rem;">
    <strong>&#9888; Evaluation & Limitations:</strong> Evaluated on 6 independent test flare events (394 sequences, 6 positives, 1.52% positive prevalence).
    BDH v4 achieves the top PR-AUC (0.0451) and catches 50.0% of test flares (3/6) with low false alarms, while LSTM leads on ROC-AUC (0.6783) and F1 (0.0727).
    TFT shows overfitting due to parameter complexity (237K params) on this sample size.
    Zero synthetic data was used in v4 — all metrics represent authentic Aditya-L1 solar observations.
  </div>
</div>

<script>
// ── Page switching ──────────────────────────────────────────────────────────
function showPage(id) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('nav .tab').forEach(t => t.classList.remove('active'));
  document.getElementById('page-' + id).classList.add('active');
  event.target.classList.add('active');
  if (id === 'results') loadResults();
}

// ── Drag & drop ─────────────────────────────────────────────────────────────
const dz = document.getElementById('drop-zone');
dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag'); });
dz.addEventListener('dragleave', () => dz.classList.remove('drag'));
dz.addEventListener('drop', e => { e.preventDefault(); dz.classList.remove('drag'); handleFile(e.dataTransfer.files[0]); });

function setStatus(html, type='ok') {
  const el = document.getElementById('upload-status');
  el.innerHTML = `<div class="status-row ${type}">${html}</div>`;
}

async function handleFile(file) {
  if (!file) return;
  if (!file.name.endsWith('.zip')) { setStatus('&#10060; Please upload a .zip file.', 'err'); return; }
  setStatus(`<span class="spinner" style="width:20px;height:20px;border-width:2px;display:inline-block;vertical-align:middle;margin-right:.5rem;"></span> Processing <strong>${file.name}</strong> (${(file.size/1024/1024).toFixed(1)} MB)...`, 'ok');

  const fd = new FormData();
  fd.append('file', file);
  try {
    const r = await fetch('/predict', { method: 'POST', body: fd });
    const data = await r.json();
    if (data.error) { setStatus('&#10060; ' + data.error, 'err'); return; }
    setStatus(`&#9989; Processed ${data.n_fits_used} frames from <strong>${file.name}</strong>. Inference: ${data.inference_ms}ms`, 'ok');
    renderTimeline(data);
    renderPredictions(data);
  } catch(e) {
    setStatus('&#10060; Network error: ' + e.message, 'err');
  }
}

function renderTimeline(data) {
  const card  = document.getElementById('timeline-card');
  const strip = document.getElementById('timeline-strip');
  const info  = document.getElementById('timeline-info');
  card.style.display = '';

  const frames = data.timeline_frames || [];
  const ts = data.frame_timestamps || [];
  const total = ts.length || data.n_fits_used || 80;

  strip.innerHTML = '';
  if (frames.length > 0) {
    frames.forEach((f, i) => {
      if (i > 0) strip.innerHTML += '<div class="timeline-connector"></div>';
      const isHorizon = i === frames.length - 1;
      const borderStyle = isHorizon ? 'border: 2px solid #ef4444; box-shadow: 0 0 14px rgba(239,68,68,0.45);' : 'border: 1px solid var(--border);';
      const imgHtml = f.image_b64 
        ? `<img src="${f.image_b64}" alt="${f.label}" style="width:100%;height:100%;object-fit:cover;border-radius:6px;display:block;">`
        : `<div style="text-align:center;"><div style="font-size:1.6rem;">&#9728;</div><div style="font-size:0.75rem;">Frame ${f.frame_index}</div></div>`;
      strip.innerHTML += `
        <div class="timeline-frame" style="min-width:160px; max-width:180px;">
          <div class="frame-img" style="width:150px;height:150px;padding:3px;overflow:hidden;background:#05070d;border-radius:8px;${borderStyle}">
            ${imgHtml}
          </div>
          <div class="frame-label" style="font-weight:700;color:var(--text);margin-top:6px;font-size:0.85rem;">${f.label}</div>
          <div class="frame-date" style="font-size:0.75rem;color:var(--muted);">${f.date}</div>
          <div class="frame-time" style="font-size:0.78rem;color:${isHorizon ? '#f87171' : 'var(--bdh)'};font-family:monospace;font-weight:600;">${f.time}</div>
        </div>`;
    });
  } else {
    // Fallback if no images generated
    const indices = total >= 4 ? [0, Math.floor(total*0.33), Math.floor(total*0.66), total-1] : [...Array(total).keys()];
    indices.forEach((idx, i) => {
      if (i > 0) strip.innerHTML += '<div class="timeline-connector"></div>';
      const label = idx === 0 ? 'Start' : idx === total-1 ? 'End (T-0)' : `Frame ${idx+1}`;
      strip.innerHTML += `
        <div class="timeline-frame">
          <div class="frame-img"><div><div style="font-size:1.5rem;">&#9728;</div><div>Frame ${idx+1}/${total}</div></div></div>
          <div class="frame-label">${label}</div>
          <div class="frame-time">${(ts[idx]||'').slice(0,19).replace('T',' ')}</div>
        </div>`;
    });
  }

  info.textContent = `Observation window: ${(data.timestamps.first||'').slice(0,19)} -> ${(data.timestamps.last||'').slice(0,19)} UTC  |  ${total} total frames  |  Prediction horizon: ~90s`;
}

function renderPredictions(data) {
  const card = document.getElementById('pred-card');
  card.style.display = '';

  // Meta
  const meta = document.getElementById('pred-meta');
  meta.innerHTML = [
    `<span>&#128337; First obs: <strong>${(data.timestamps.first||'').slice(0,19)}</strong></span>`,
    `<span>&#128337; Last obs: <strong>${(data.timestamps.last||'').slice(0,19)}</strong></span>`,
    `<span>&#127919; Horizon: <strong>~${data.prediction_horizon_seconds}s</strong></span>`,
    `<span>&#128200; Frames used: <strong>${data.n_fits_used}</strong></span>`,
    `<span>&#10003; Valid frames: <strong>${data.n_valid_frames}</strong></span>`,
  ].join('');

  const preds = data.predictions || {};
  const models = [
    {name:'BDH',  cls:'bdh',  dot:'dot-bdh'},
    {name:'LSTM', cls:'lstm', dot:'dot-lstm'},
    {name:'TFT',  cls:'tft',  dot:'dot-tft'},
  ];

  const tbody = document.getElementById('pred-tbody');
  tbody.innerHTML = '';
  let flare_count = 0;

  models.forEach(m => {
    const p = preds[m.name] || {};
    if (p.error) {
      tbody.innerHTML += `<tr><td><span class="model-dot ${m.dot}"></span> ${m.name}</td><td colspan="3" style="color:var(--muted);">${p.error}</td></tr>`;
      return;
    }
    const prob  = (p.probability || 0);
    const isFlare = (p.prediction || '').includes('FLARE');
    if (isFlare) flare_count++;
    const pct   = Math.min(prob * 100 * 10, 100).toFixed(1);  // scale for visibility
    const color = isFlare ? '#ef4444' : '#10b981';
    tbody.innerHTML += `
      <tr>
        <td><span class="model-dot ${m.dot}"></span> <strong>${m.name}</strong></td>
        <td><span class="pill ${isFlare ? 'flare':'no-flare'}">${p.prediction || 'N/A'}</span></td>
        <td>
          <div class="prob-bar">
            <div class="prob-bg"><div class="prob-fill" style="width:${pct}%;background:${color}"></div></div>
            <span style="min-width:50px;font-family:monospace;font-size:.82rem;">${(prob*100).toFixed(2)}%</span>
          </div>
        </td>
        <td style="font-size:.82rem; color:${p.confidence==='High'?'#f87171':p.confidence==='Medium'?'#fbbf24':'var(--muted)'};">${p.confidence||''}</td>
      </tr>`;
  });

  // Agreement
  const agree_el = document.getElementById('agreement-box');
  const n_models  = models.filter(m => preds[m.name] && !preds[m.name].error).length;
  if (n_models > 0) {
    const agree = flare_count === 0 || flare_count === n_models;
    agree_el.innerHTML = `<div class="agreement ${agree?'agree':'disagree'}">
      ${agree ? '&#9989;' : '&#9888;'}
      <strong>${agree ? 'Models Agree' : 'Models Disagree'}</strong>:
      ${flare_count}/${n_models} models predict a flare.
      ${flare_count > 0 && flare_count < n_models ? ' Treat disagreement with caution — the dataset is very small.' : ''}
    </div>`;
  }

  // Warnings
  const warn_el = document.getElementById('warnings-box');
  if (data.warnings && data.warnings.length) {
    warn_el.innerHTML = `<div class="status-row warn">&#9888; ${data.warnings.length} warning(s): ${data.warnings.slice(0,3).join('; ')}</div>`;
  } else {
    warn_el.innerHTML = '';
  }
}

// ── Results Page ─────────────────────────────────────────────────────────────
let _results = null;
let _currentExp = 'original';
let _currentCmp = 'augmented';

async function loadResults() {
  if (_results) { renderResults(_results); return; }
  try {
    const r = await fetch('/api/results_data');
    const d = await r.json();
    if (d.error) {
      document.getElementById('results-loading').innerHTML =
        `<div class="status-row err">&#10060; ${d.error}</div>`;
      return;
    }
    _results = d;
    document.getElementById('results-loading').style.display = 'none';
    document.getElementById('results-content').style.display = '';
    renderResults(d);
  } catch(e) {
    document.getElementById('results-loading').innerHTML =
      `<div class="status-row err">&#10060; Failed to load: ${e.message}</div>`;
  }
}

function statBox(val, lbl, color='var(--text)') {
  return `<div class="stat-box"><div class="val" style="color:${color}">${val}</div><div class="lbl">${lbl}</div></div>`;
}

function renderResults(d) {
  const r = d.results;
  const plots = d.plots || {};

  // Dataset stats
  if (r.v4_stats) {
    const s = r.v4_stats;
    const origTitle = document.getElementById('stats-orig-title');
    const augTitle = document.getElementById('stats-aug-title');
    if (origTitle) origTitle.textContent = "TRAIN & VAL SETS (AUTHENTIC SUIT)";
    if (augTitle) augTitle.textContent = "INDEPENDENT TEST SET & TOTALS";

    document.getElementById('stats-orig').innerHTML = [
      statBox(s.train_samples, 'Train sequences', '#3b82f6'),
      statBox(`${s.train_positives} (${s.train_flares} flares)`, 'Train positives', '#f97316'),
      statBox(s.val_samples, 'Val sequences', '#94a3b8'),
      statBox(`${s.val_positives} (${s.val_flares} flares)`, 'Val positives', '#eab308'),
    ].join('');
    document.getElementById('stats-aug').innerHTML = [
      statBox(s.test_samples, 'Test sequences', '#3b82f6'),
      statBox(`${s.test_positives} (${s.test_flares} flares)`, 'Test positives', '#ef4444'),
      statBox(s.total_sequences, 'Total sequences', '#22c55e'),
      statBox(s.total_flares, 'Independent flares', '#a78bfa'),
    ].join('');
    document.getElementById('aug-note').innerHTML =
      `<strong>100% Authentic Observational SUIT Data:</strong> Zero synthetic augmentation or SMOTE. 
       4,309 valid NB04 (2803 &Aring;) frames processed into 2,690 sliding sequences (window=80 frames). 
       Strict event-aware chronological split at 2024-12-30 14:00 UTC ensures zero cross-event leakage across 
       Train (19 flares), Val (7 flares), and Test (6 flares).`;
  } else {
    if (r.original) {
      document.getElementById('stats-orig').innerHTML = [
        statBox(r.original.train_samples, 'Train sequences'),
        statBox(r.original.train_positives, 'Train positives', '#f97316'),
        statBox(r.original.test_samples, 'Test sequences'),
        statBox(r.original.test_positives, 'Test positives', '#ef4444'),
      ].join('');
    }
    if (r.augmented) {
      document.getElementById('stats-aug').innerHTML = [
        statBox(r.augmented.train_samples, 'Train sequences'),
        statBox(r.augmented.train_positives, 'Train positives', '#22c55e'),
        statBox(r.augmented.n_synth || (r.augmented.train_positives - (r.original?.train_positives||0)), 'Synthetic added', '#3b82f6'),
        statBox(r.augmentation?.n_flare_events || 7, 'Independent events', '#a78bfa'),
      ].join('');
      document.getElementById('aug-note').innerHTML =
        `<strong>Important:</strong> The ${r.augmentation?.n_synth||39} synthetic samples are variations of
         ${r.augmentation?.n_flare_events||7} real flare events using multivariate Gaussian perturbation
         (sigma_scale=0.05). They increase diversity within observed events but do not represent new
         independent solar flare observations.`;
    }
  }

  showExp(_currentExp);
  showCmp(_currentCmp);
  renderPlots(plots);
}

function showExp(exp) {
  _currentExp = exp;
  if (!_results) return;
  const r = _results.results;
  const data = r[exp] || r['original'];
  if (!data) return;

  const btnOrig = document.getElementById('btn-orig');
  const btnAug  = document.getElementById('btn-aug');
  if (btnOrig) {
    btnOrig.className = 'btn btn-sm ' + (exp==='original' ? 'btn-primary':'');
    btnOrig.style.background = exp==='original' ? '' : 'var(--surface2)';
  }
  if (btnAug) {
    btnAug.className  = 'btn btn-sm ' + (exp==='augmented'? 'btn-primary':'');
    btnAug.style.background = exp==='augmented' ? '' : 'var(--surface2)';
  }

  const grid  = document.getElementById('metrics-grid');
  const models= [
    {name:'BDH',  cls:'bdh',  color:'var(--bdh)'},
    {name:'LSTM', cls:'lstm', color:'var(--lstm)'},
    {name:'TFT',  cls:'tft',  color:'var(--tft)'},
  ];

  // Find best PR-AUC
  const pr_aucs = models.map(m => ((data.metrics||{})[m.name]||{}).pr_auc || 0);
  const best_pr = Math.max(...pr_aucs);

  grid.innerHTML = models.map(m => {
    const mx = (data.metrics||{})[m.name] || {};
    const ot = mx.optimal_f1_threshold || {};
    const isBest = mx.pr_auc === best_pr && mx.pr_auc > 0;
    return `<div class="model-card ${m.cls}">
      <h3><span class="model-dot dot-${m.cls}" style="margin-right:.4rem;"></span>${m.name}
        ${isBest ? '<span class="winner-badge">Best PR-AUC</span>':''}</h3>
      <div class="metric-row"><span>PR-AUC</span><span class="val" style="color:${m.color}">${mx.pr_auc||'N/A'}</span></div>
      <div class="metric-row"><span>ROC-AUC</span><span class="val">${mx.roc_auc||'N/A'}</span></div>
      <div class="metric-row"><span>F1 (optimal)</span><span class="val">${ot.f1||'N/A'}</span></div>
      <div class="metric-row"><span>Recall</span><span class="val">${ot.recall||'N/A'}</span></div>
      <div class="metric-row"><span>Precision</span><span class="val">${ot.precision||'N/A'}</span></div>
      <div class="metric-row"><span>False Alarms (FP)</span><span class="val">${ot.fp!==undefined?ot.fp:'N/A'}</span></div>
      <div class="metric-row"><span>Missed Flares (FN)</span><span class="val">${ot.fn!==undefined?ot.fn:'N/A'}</span></div>
      <div class="metric-row"><span>Params</span><span class="val" style="font-size:.82rem">${mx.n_params?mx.n_params.toLocaleString():'N/A'}</span></div>
    </div>`;
  }).join('');
}

function showCmp(exp) {
  _currentCmp = exp;
  const bOrig = document.getElementById('cmp-orig-btn');
  const bAug  = document.getElementById('cmp-aug-btn');
  if (bOrig) {
    bOrig.className = 'btn btn-sm ' + (exp==='original'  ? 'btn-primary':'');
    bOrig.style.background = exp==='original' ? '' : 'var(--surface2)';
  }
  if (bAug) {
    bAug.className  = 'btn btn-sm ' + (exp==='augmented' ? 'btn-primary':'');
    bAug.style.background = exp==='augmented' ? '' : 'var(--surface2)';
  }

  if (!_results) return;
  const data = (_results.results||{})[exp] || (_results.results||{})['original'];
  if (!data) { document.getElementById('cmp-table-wrap').innerHTML = '<div style="color:var(--muted)">No data.</div>'; return; }

  const models = ['BDH','LSTM','TFT'];
  const cols = ['PR-AUC','ROC-AUC','F1','Recall','Precision','FP','FN','Params'];

  // Find bests
  const mx = data.metrics || {};
  const vals = models.map(m => ({
    pr_auc: mx[m]?.pr_auc||0,
    roc_auc: mx[m]?.roc_auc||0,
    f1: mx[m]?.optimal_f1_threshold?.f1||0,
    recall: mx[m]?.optimal_f1_threshold?.recall||0,
    precision: mx[m]?.optimal_f1_threshold?.precision||0,
    fp: mx[m]?.optimal_f1_threshold?.fp,
    fn: mx[m]?.optimal_f1_threshold?.fn,
    params: mx[m]?.n_params||0,
  }));
  const best = {
    pr_auc:    Math.max(...vals.map(v=>v.pr_auc)),
    roc_auc:   Math.max(...vals.map(v=>v.roc_auc)),
    f1:        Math.max(...vals.map(v=>v.f1)),
    recall:    Math.max(...vals.map(v=>v.recall)),
    precision: Math.max(...vals.map(v=>v.precision)),
    fp:        Math.min(...vals.filter(v=>v.fp!==undefined).map(v=>v.fp)),
    fn:        Math.min(...vals.filter(v=>v.fn!==undefined).map(v=>v.fn)),
  };

  let html = `<table class="cmp-table"><thead><tr><th>Model</th>${cols.map(c=>`<th>${c}</th>`).join('')}</tr></thead><tbody>`;
  models.forEach((m, i) => {
    const v = vals[i];
    const isBestFP = v.fp === best.fp;
    const isBestFN = v.fn === best.fn;
    html += `<tr>
      <td><span class="model-dot dot-${m.toLowerCase()}" style="margin-right:.4rem;"></span><strong>${m}</strong></td>
      <td class="${v.pr_auc===best.pr_auc?'best':''}">${v.pr_auc||'N/A'}</td>
      <td class="${v.roc_auc===best.roc_auc?'best':''}">${v.roc_auc||'N/A'}</td>
      <td class="${v.f1===best.f1?'best':''}">${v.f1||'N/A'}</td>
      <td class="${v.recall===best.recall?'best':''}">${v.recall||'N/A'}</td>
      <td class="${v.precision===best.precision?'best':''}">${v.precision||'N/A'}</td>
      <td class="${isBestFP?'best':''}">${v.fp!==undefined?v.fp:'N/A'}</td>
      <td class="${isBestFN?'best':''}">${v.fn!==undefined?v.fn:'N/A'}</td>
      <td style="font-size:.8rem">${v.params?v.params.toLocaleString():'N/A'}</td>
    </tr>`;
  });
  html += '</tbody></table><div style="margin-top:.5rem;font-size:.75rem;color:var(--muted);">Green = best value for that metric.</div>';
  document.getElementById('cmp-table-wrap').innerHTML = html;
}

function renderPlots(plots) {
  const plotDefs = [
    {key:'training_curves_v4',      label:'Loss Curves (v4 Authentic)'},
    {key:'pr_roc_curves_v4',        label:'PR & ROC Curves (v4)'},
    {key:'confusion_matrices_v4',   label:'Confusion Matrices (v4)'},
    {key:'model_comparison_bar_v4', label:'Bar Comparison (v4)'},
    {key:'training_curves_v3_orig',  label:'Loss (Original)'},
    {key:'training_curves_v3_aug',   label:'Loss (Augmented)'},
    {key:'pr_roc_curves_v3_orig',    label:'PR/ROC (Original)'},
    {key:'pr_roc_curves_v3_aug',     label:'PR/ROC (Augmented)'},
    {key:'confusion_matrices_v3_orig', label:'Confusion (Original)'},
    {key:'confusion_matrices_v3_aug',  label:'Confusion (Augmented)'},
    {key:'model_comparison_bar_v3',  label:'Bar Comparison'},
    // Legacy v2 plots
    {key:'pr_roc_curves_v2',   label:'PR/ROC v2'},
    {key:'training_curves_v2', label:'Loss v2'},
  ];

  const tabs  = document.getElementById('plot-tabs');
  const panes = document.getElementById('plot-panes');
  tabs.innerHTML = ''; panes.innerHTML = '';
  let first = true;

  plotDefs.forEach(pd => {
    const url = plots[pd.key];
    if (!url) return;
    const id = 'pane-' + pd.key;
    tabs.innerHTML  += `<button class="plot-tab ${first?'active':''}" onclick="switchPlot('${id}',this)">${pd.label}</button>`;
    panes.innerHTML += `<div class="plot-pane ${first?'active':''}" id="${id}"><img src="${url}" alt="${pd.label}" loading="lazy"></div>`;
    first = false;
  });
}

function switchPlot(paneId, btn) {
  document.querySelectorAll('.plot-pane').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.plot-tab').forEach(t => t.classList.remove('active'));
  document.getElementById(paneId).classList.add('active');
  btn.classList.add('active');
}
</script>
</body>
</html>"""


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("Flare-Forecast v3 Dashboard Server")
    print("=" * 60)
    print(f"Models dir : {MODELS}")
    print(f"Docs dir   : {DOCS}")
    print()

    # Pre-load norm params
    try:
        p = _load_norm_params()
        print(f"Norm params loaded: mu.shape={p['mu'].shape}")
    except FileNotFoundError as e:
        print(f"WARNING: {e}")
        print("  Run train_models_v3.py first to generate model checkpoints.")

    print("\nStarting Flask server on http://localhost:5000")
    print("Ctrl+C to stop.\n")
    app.run(host="0.0.0.0", port=5000, debug=False)

