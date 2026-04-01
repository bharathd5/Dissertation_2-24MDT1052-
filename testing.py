import torch
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from model import PrimaryTransformer
from pipeline_data import (
    load_and_prepare,
    apply_scaler,
    SlidingWindowDataset,
    INV_LABEL_MAP
)

from torch.utils.data import DataLoader

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
DATA_PATH = "data/features_3min.csv"
SCALER_PATH = "models/scaler.pkl"
MODEL_PATH = "models/primary_transformer.pt"
META_MODEL_PATH = "models/meta_lgbm.txt"

BATCH_SIZE = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TAU = 0.66   # from training

# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────
print("Loading data...")

df, feat_cols = load_and_prepare(Path(DATA_PATH))

# IMPORTANT: DO NOT FIT SCALER AGAIN
scaler = joblib.load(SCALER_PATH)
df = apply_scaler(df, feat_cols, scaler)

# ─────────────────────────────────────────────
# DATASET
# ─────────────────────────────────────────────
dataset = SlidingWindowDataset(df, feat_cols)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
print("Loading transformer model...")

model = PrimaryTransformer(n_features=len(feat_cols))
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(checkpoint["model_state"])
model.to(DEVICE)
model.eval()

# ─────────────────────────────────────────────
# STAGE 1: PREDICTIONS
# ─────────────────────────────────────────────
print("Running Stage 1 predictions...")

all_probs = []
all_preds = []

with torch.no_grad():
    for X, _ in loader:
        X = X.to(DEVICE)
        probs = model.predict_proba(X)   # [B, 3]
        preds = torch.argmax(probs, dim=1)

        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())

probs = np.vstack(all_probs)
preds = np.concatenate(all_preds)

# ─────────────────────────────────────────────
# ALIGN WITH DATAFRAME
# ─────────────────────────────────────────────
valid_idx = df.index[len(df) - len(preds):]

df = df.iloc[-len(preds):].copy()
df["pred_class"] = preds
df["pred_label"] = [INV_LABEL_MAP[p] for p in preds]

# ─────────────────────────────────────────────
# STAGE 2: META MODEL
# ─────────────────────────────────────────────
print("Loading meta model...")

import lightgbm as lgb
meta_model = lgb.Booster(model_file=META_MODEL_PATH)

# Build meta features
sorted_p = np.sort(probs, axis=1)[:, ::-1]
confidence = sorted_p[:, 0] - sorted_p[:, 1]

X_meta = np.concatenate([
    df[feat_cols].values,
    probs,
    preds.reshape(-1, 1),
    confidence.reshape(-1, 1)
], axis=1)

meta_probs = meta_model.predict(X_meta)

df["meta_prob"] = meta_probs

# ─────────────────────────────────────────────
# APPLY THRESHOLD (τ)
# ─────────────────────────────────────────────
df["trade"] = (df["meta_prob"] > TAU).astype(int)

# Keep only trades
trades = df[df["trade"] == 1].copy()

print(f"Total trades: {len(trades)}")

# ─────────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────────
df.to_csv("results/full_predictions.csv", index=False)
trades.to_csv("results/trades_only.csv", index=False)

print("Saved results:")
print(" - results/full_predictions.csv")
print(" - results/trades_only.csv")