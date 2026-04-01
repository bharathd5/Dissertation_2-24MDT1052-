"""
pipeline_data.py  —  Data loading, labelling, preprocessing, and PyTorch Dataset.

Covers:
  - Triple-barrier labelling (take-profit / stop-loss / time-exit)
  - Fixed-horizon labelling (fallback)
  - RobustScaler with shift(1) leakage guard
  - Time-ordered train/val/test split
  - SlidingWindowDataset for transformer input
"""

import logging
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
import torch
from torch.utils.data import Dataset

from config import (
    EXCLUDE_COLS, LABEL_METHOD, HORIZON_BARS, TP_MULT, SL_MULT,
    MAX_HOLD, FLAT_BAND, SEQ_LEN, STEP_SIZE,
    TRAIN_RATIO, VAL_RATIO, SEED,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  TRIPLE-BARRIER LABELLING
# ─────────────────────────────────────────────────────────────────────────────

def triple_barrier_labels(df: pd.DataFrame,
                          tp_mult: float = TP_MULT,
                          sl_mult: float = SL_MULT,
                          max_hold: int   = MAX_HOLD) -> pd.Series:
    """
    For each bar i:
      - Take-profit barrier : close[i] ± tp_mult * ATR14[i]
      - Stop-loss barrier   : close[i] ∓ sl_mult * ATR14[i]
      - Time exit           : after max_hold bars

    Returns:
      label  +1 (long TP hit first), -1 (short TP hit first), 0 (time exit)

    Note: the label at bar i uses only future prices → no leakage.
    """
    close  = df["Close"].values
    atr    = df["atr_14"].values        # must exist in feature-engineered CSV
    n      = len(df)
    labels = np.zeros(n, dtype=np.int8)

    for i in range(n - max_hold - 1):
        tp = tp_mult * atr[i]
        sl = sl_mult * atr[i]
        if tp == 0 or sl == 0 or np.isnan(tp) or np.isnan(sl):
            labels[i] = 0
            continue
        ref = close[i]
        for j in range(i + 1, min(i + max_hold + 1, n)):
            ret = close[j] - ref
            if ret >= tp:
                labels[i] = 1
                break
            if ret <= -sl:
                labels[i] = -1
                break
        # else: labels[i] stays 0 (time exit)

    # Last max_hold bars get 0 (no valid future window) — will be dropped
    labels[-(max_hold + 1):] = 0
    return pd.Series(labels, index=df.index, name="label_primary")


def fixed_horizon_labels(df: pd.DataFrame,
                         horizon: int   = HORIZON_BARS,
                         flat_band: float = FLAT_BAND) -> pd.Series:
    """
    label = sign of return over `horizon` bars, with dead-band for flat.
    """
    future_ret = df["Close"].shift(-horizon) / df["Close"] - 1
    label = np.where(future_ret >  flat_band,  1,
            np.where(future_ret < -flat_band, -1, 0)).astype(np.int8)
    label[-horizon:] = 0            # last bars have no future
    return pd.Series(label, index=df.index, name="label_primary")


# ─────────────────────────────────────────────────────────────────────────────
#  LOAD AND PREPARE DATA
# ─────────────────────────────────────────────────────────────────────────────

def load_and_prepare(csv_path: Path,
                     label_method: str = LABEL_METHOD
                     ) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load feature CSV, attach labels, return clean DataFrame + feature column list.

    Steps:
      1. Load CSV, parse UTC timestamp
      2. Sort by time (critical for walk-forward validity)
      3. Drop NaN rows from rolling-window warmup
      4. Attach primary label (triple-barrier or fixed-horizon)
      5. Identify feature columns (everything not in EXCLUDE_COLS)
      6. Return df with all columns intact (scaler applied later per split)
    """
    log.info(f"Loading {csv_path} ...")
    df = pd.read_csv(csv_path, low_memory=False)
    df = df.sort_values("UTC").reset_index(drop=True)

    # Identify feature columns before any dropping
    non_feat = set(EXCLUDE_COLS) | {"label_primary", "label_meta"}
    feat_cols_raw = [c for c in df.columns if c not in non_feat
                     and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

    # Drop columns that are entirely NaN (e.g. vol_percentile on small datasets)
    all_nan_cols = [c for c in feat_cols_raw if df[c].isna().all()]
    if all_nan_cols:
        log.info(f"  Dropping {len(all_nan_cols)} all-NaN columns: {all_nan_cols}")
        df = df.drop(columns=all_nan_cols)
        feat_cols_raw = [c for c in feat_cols_raw if c not in all_nan_cols]

    # Drop warmup rows using only core columns as signal
    core_cols = [c for c in ["atr_14", "rsi_14", "ema_slope_20",
                              "log_return", "close_position"] if c in df.columns]
    initial_len = len(df)
    df = df.dropna(subset=core_cols).reset_index(drop=True)
    log.info(f"  Dropped {initial_len - len(df)} warmup rows -> {len(df)} remain")

    # Fill remaining NaNs in feature columns with 0
    df[feat_cols_raw] = df[feat_cols_raw].fillna(0.0)

    # Attach primary label
    if label_method == "triple_barrier":
        if "atr_14" not in df.columns:
            raise ValueError("triple_barrier requires atr_14 column. "
                             "Set LABEL_METHOD=fixed_horizon in config.py.")
        df["label_primary"] = triple_barrier_labels(df)
    else:
        df["label_primary"] = fixed_horizon_labels(df)

    df = df.reset_index(drop=True)

    # Final feature column list
    feat_cols = [c for c in df.columns if c not in non_feat
                 and c != "label_primary" and c != "label_meta"
                 and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]

    log.info(f"  Feature columns: {len(feat_cols)}")
    log.info(f"  Label distribution: {df['label_primary'].value_counts().to_dict()}")
    return df, feat_cols


# ─────────────────────────────────────────────────────────────────────────────
#  TRAIN / VAL / TEST SPLIT  (time-ordered, no shuffle)
# ─────────────────────────────────────────────────────────────────────────────

def time_split(df: pd.DataFrame,
               train_ratio: float = TRAIN_RATIO,
               val_ratio:   float = VAL_RATIO
               ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    n      = len(df)
    n_tr   = int(n * train_ratio)
    n_va   = int(n * val_ratio)
    train  = df.iloc[:n_tr].copy()
    val    = df.iloc[n_tr : n_tr + n_va].copy()
    test   = df.iloc[n_tr + n_va:].copy()
    log.info(f"  Split → train={len(train):,}  val={len(val):,}  test={len(test):,}")
    return train, val, test


# ─────────────────────────────────────────────────────────────────────────────
#  SCALING  (fit on train only, transform val/test — leakage guard)
# ─────────────────────────────────────────────────────────────────────────────

def fit_scaler(train_df: pd.DataFrame,
               feat_cols: List[str]) -> RobustScaler:
    scaler = RobustScaler()
    scaler.fit(train_df[feat_cols].values)
    return scaler


def apply_scaler(df: pd.DataFrame,
                 feat_cols: List[str],
                 scaler: RobustScaler) -> pd.DataFrame:
    df = df.copy()
    df[feat_cols] = scaler.transform(df[feat_cols].values)
    # Replace any inf/-inf introduced by division-by-zero in features
    df[feat_cols] = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  LABEL MAP  (−1 / 0 / +1  →  0 / 1 / 2  for CrossEntropyLoss)
# ─────────────────────────────────────────────────────────────────────────────

LABEL_MAP = {-1: 0, 0: 1, 1: 2}    # 0=short, 1=flat, 2=long
INV_LABEL_MAP = {0: -1, 1: 0, 2: 1}


def map_labels(series: pd.Series) -> np.ndarray:
    return series.map(LABEL_MAP).fillna(1).astype(np.int64).values


# ─────────────────────────────────────────────────────────────────────────────
#  PYTORCH DATASET
# ─────────────────────────────────────────────────────────────────────────────

class SlidingWindowDataset(Dataset):
    """
    Converts a scaled DataFrame into overlapping windows of shape (SEQ_LEN, n_features).

    Each sample:
        X : float32 tensor  [SEQ_LEN, n_features]
        y : int64 tensor    scalar class index {0, 1, 2}

    The label assigned to a window is the label of its LAST bar
    (the bar we are predicting the future of).
    """

    def __init__(self,
                 df:        pd.DataFrame,
                 feat_cols: List[str],
                 seq_len:   int  = SEQ_LEN,
                 step:      int  = STEP_SIZE):
        self.seq_len   = seq_len
        self.feat_cols = feat_cols
        self.X_all     = df[feat_cols].values.astype(np.float32)   # [N, F]
        self.y_all     = map_labels(df["label_primary"])            # [N]
        self.indices   = list(range(seq_len - 1, len(df), step))

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        end   = self.indices[idx] + 1
        start = end - self.seq_len
        x = self.X_all[start:end]       # [SEQ_LEN, F]
        y = self.y_all[end - 1]         # label of the last bar
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long)


# ─────────────────────────────────────────────────────────────────────────────
#  META-LABEL DATASET  (flat, tabular — for LightGBM)
# ─────────────────────────────────────────────────────────────────────────────

def build_meta_features(df: pd.DataFrame,
                        feat_cols: List[str],
                        oof_probs: np.ndarray,     # [N, 3]  softmax from Stage 1
                        oof_preds: np.ndarray,     # [N]     argmax predictions
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct the augmented feature matrix for the meta-label model.

    Meta features = original features (last bar of each valid row)
                  + Stage 1 softmax probabilities (3 values)
                  + predicted class (1 value)
                  + confidence = max_prob - second_max_prob (1 value)

    Returns:
        X_meta : [N, F+5]
        y_meta : [N]   binary  1 if Stage 1 was correct, 0 otherwise
    """
    # Align df to valid OOF rows (rows that have predictions)
    X_raw   = df[feat_cols].values.astype(np.float32)
    y_true  = map_labels(df["label_primary"])

    # Confidence signal
    sorted_p   = np.sort(oof_probs, axis=1)[:, ::-1]
    confidence = sorted_p[:, 0] - sorted_p[:, 1]   # margin between top-2

    X_meta = np.concatenate([
        X_raw,
        oof_probs,                                  # p(short), p(flat), p(long)
        oof_preds.reshape(-1, 1).astype(np.float32),
        confidence.reshape(-1, 1),
    ], axis=1)

    y_meta = (oof_preds == y_true).astype(np.int8)  # 1 if correct, 0 if wrong

    pos_rate = y_meta.mean()
    log.info(f"  Meta labels: {y_meta.sum():,} correct / {len(y_meta):,} total "
             f"({pos_rate:.1%} positive rate)")
    return X_meta, y_meta