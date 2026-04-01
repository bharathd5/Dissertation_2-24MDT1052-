"""
backtest.py  —  Walk-forward backtester for the two-stage signal system.

Walk-forward logic:
  - Expanding window: train on [0..i], test on [i..i+step]
  - Runs N windows, evaluates each independently
  - Aggregates PnL, Sharpe, drawdown, win-rate across all windows

This is the correct way to validate a financial ML model.
Random train/test split will produce grossly inflated backtest metrics.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import lightgbm as lgb

from config import (
    BATCH_SIZE, NUM_WORKERS, SEQ_LEN, SEED,
    D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT,
)
from pipeline_data import (
    load_and_prepare, fit_scaler, apply_scaler,
    SlidingWindowDataset, build_meta_features, LABEL_MAP,
)
from model import PrimaryTransformer
from trainer import get_device, train_primary, generate_oof_predictions, batch_predict
from meta_model import train_meta_model, tune_threshold, apply_signal_gate, compute_position_sizes

log = logging.getLogger(__name__)


@dataclass
class WalkForwardResult:
    window_id:      int
    train_size:     int
    test_size:      int
    n_trades:       int
    win_rate:       float
    sharpe:         float
    cum_return:     float
    max_drawdown:   float
    s1_accuracy:    float
    meta_auc:       float = 0.0
    tau:            float = 0.55

    def to_dict(self):
        return self.__dict__


def run_walk_forward(csv_path:    Path,
                     n_windows:   int = 5,
                     test_frac:   float = 0.10,
                     min_train_frac: float = 0.40,
                     ) -> pd.DataFrame:
    """
    Walk-forward validation.

    Parameters
    ----------
    csv_path      : path to the feature-engineered CSV
    n_windows     : number of walk-forward windows
    test_frac     : fraction of total data used per test window
    min_train_frac: minimum fraction of data in the first training window

    Returns
    -------
    DataFrame of per-window results + aggregate row
    """
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = get_device()

    df, feat_cols = load_and_prepare(csv_path)
    n             = len(df)
    test_size     = int(n * test_frac)
    n_features    = len(feat_cols)

    # Build window start/end indices
    # Expanding window: each test window starts after the previous one
    first_train_end = int(n * min_train_frac)
    window_starts   = np.linspace(first_train_end,
                                  n - test_size,
                                  n_windows, dtype=int)

    log.info(f"Walk-forward: {n_windows} windows  "
             f"test_size={test_size:,}  n={n:,}")

    results: List[WalkForwardResult] = []

    for w_idx, train_end in enumerate(window_starts, 1):
        test_start = train_end
        test_end   = min(train_end + test_size, n)

        log.info(f"\n{'─'*50}")
        log.info(f"Window {w_idx}/{n_windows}  "
                 f"train=[0..{train_end:,}]  "
                 f"test=[{test_start:,}..{test_end:,}]")

        df_tr = df.iloc[:train_end].copy()
        df_te = df.iloc[test_start:test_end].copy()

        if len(df_tr) < SEQ_LEN * 10 or len(df_te) < SEQ_LEN * 2:
            log.warning(f"  Window {w_idx}: insufficient data, skipping.")
            continue

        # Scaler fitted on train only
        scaler = fit_scaler(df_tr, feat_cols)
        df_tr  = apply_scaler(df_tr, feat_cols, scaler)
        df_te  = apply_scaler(df_te, feat_cols, scaler)

        # Reserve last 15% of train as internal val
        val_split  = int(len(df_tr) * 0.85)
        df_tr_sub  = df_tr.iloc[:val_split].reset_index(drop=True)
        df_val_sub = df_tr.iloc[val_split:].reset_index(drop=True)

        tr_ds  = SlidingWindowDataset(df_tr_sub,  feat_cols)
        va_ds  = SlidingWindowDataset(df_val_sub, feat_cols)
        te_ds  = SlidingWindowDataset(df_te.reset_index(drop=True), feat_cols)

        # ── Primary transformer ───────────────────────────────────────────
        model = PrimaryTransformer(n_features=n_features).to(device)
        wf_model_path = Path(f"models/wf_primary_w{w_idx}.pt")
        train_primary(model, tr_ds, va_ds, device, save_path=wf_model_path)

        ckpt = torch.load(wf_model_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])

        # ── OOF on training subset ────────────────────────────────────────
        oof_probs, oof_preds, valid_mask = generate_oof_predictions(
            df_tr_sub, feat_cols, n_features, device
        )
        df_tr_oof   = df_tr_sub.iloc[valid_mask].reset_index(drop=True)
        oof_probs_v = oof_probs[valid_mask]
        oof_preds_v = oof_preds[valid_mask]

        X_meta_tr, y_meta_tr = build_meta_features(
            df_tr_oof, feat_cols, oof_probs_v, oof_preds_v
        )

        # Val meta features
        va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE*2,
                               shuffle=False, num_workers=NUM_WORKERS)
        val_probs, val_preds, val_true = batch_predict(model, va_loader, device)
        df_val_aligned = df_val_sub.iloc[SEQ_LEN-1:].reset_index(drop=True)
        X_meta_va, y_meta_va = build_meta_features(
            df_val_aligned, feat_cols, val_probs, val_preds
        )

        # ── Meta model ────────────────────────────────────────────────────
        wf_meta_path = Path(f"models/wf_meta_w{w_idx}.txt")
        meta_model   = train_meta_model(X_meta_tr, y_meta_tr,
                                        X_meta_va, y_meta_va,
                                        save_path=wf_meta_path)

        # ── Threshold tuning on val ───────────────────────────────────────
        meta_va_probs = meta_model.predict_proba(X_meta_va)[:, 1]
        val_returns   = (
            df_val_sub["Close"].diff(1).shift(-1)
            .iloc[SEQ_LEN-1:].reset_index(drop=True).values[:len(meta_va_probs)]
        )
        tau, _ = tune_threshold(meta_va_probs, y_meta_va, val_returns, val_preds)

        # ── Test set evaluation ───────────────────────────────────────────
        te_loader = DataLoader(te_ds, batch_size=BATCH_SIZE*2,
                               shuffle=False, num_workers=NUM_WORKERS)
        te_probs, te_preds, te_true = batch_predict(model, te_loader, device)

        df_te_aligned = df_te.reset_index(drop=True).iloc[SEQ_LEN-1:].reset_index(drop=True)
        X_meta_te, y_meta_te = build_meta_features(
            df_te_aligned, feat_cols, te_probs, te_preds
        )
        meta_te_probs = meta_model.predict_proba(X_meta_te)[:, 1]
        signals       = apply_signal_gate(te_preds, meta_te_probs, tau)
        sizes         = compute_position_sizes(meta_te_probs, signals, tau)

        te_returns = (
            df_te.reset_index(drop=True)["Close"].diff(1).shift(-1)
            .iloc[SEQ_LEN-1:].reset_index(drop=True).values[:len(signals)]
        )

        # Metrics
        trade_mask = signals != 1
        s1_acc     = (te_preds == te_true).mean()

        try:
            from sklearn.metrics import roc_auc_score
            m_auc = roc_auc_score(y_meta_te, meta_te_probs)
        except Exception:
            m_auc = 0.0

        if trade_mask.sum() > 0:
            direction  = np.where(signals[trade_mask] == 2,  1.0, -1.0)
            pnl        = direction * te_returns[trade_mask] * sizes[trade_mask]
            sharpe     = (pnl.mean() / (pnl.std() + 1e-9)) * np.sqrt(252 * 1440)
            cum_ret    = float(np.sum(pnl))
            win_rate   = float((pnl > 0).mean())
            cum        = np.cumsum(pnl)
            rm         = np.maximum.accumulate(cum)
            max_dd     = float((rm - cum).max())
        else:
            sharpe = cum_ret = win_rate = max_dd = 0.0

        r = WalkForwardResult(
            window_id    = w_idx,
            train_size   = len(df_tr),
            test_size    = len(df_te),
            n_trades     = int(trade_mask.sum()),
            win_rate     = round(win_rate, 4),
            sharpe       = round(sharpe, 4),
            cum_return   = round(cum_ret, 6),
            max_drawdown = round(max_dd, 6),
            s1_accuracy  = round(s1_acc, 4),
            meta_auc     = round(m_auc, 4),
            tau          = round(tau, 3),
        )
        results.append(r)
        log.info(f"  Window {w_idx} results: {r.to_dict()}")

    results_df = pd.DataFrame([r.to_dict() for r in results])

    # Aggregate row
    if not results_df.empty:
        agg = results_df[["sharpe","cum_return","win_rate","max_drawdown",
                          "s1_accuracy","meta_auc","n_trades"]].mean()
        agg_row = {"window_id": "MEAN", **agg.round(4).to_dict()}
        results_df = pd.concat(
            [results_df, pd.DataFrame([agg_row])], ignore_index=True
        )

    out_path = Path("results/walk_forward_results.csv")
    results_df.to_csv(out_path, index=False)
    log.info(f"\nWalk-forward results saved → {out_path}")
    log.info(f"\n{results_df.to_string()}")
    return results_df


if __name__ == "__main__":
    import argparse
    from config import DATA_FILES, PRIMARY_TF

    parser = argparse.ArgumentParser()
    parser.add_argument("--tf",         default=PRIMARY_TF)
    parser.add_argument("--n_windows",  type=int, default=5)
    parser.add_argument("--test_frac",  type=float, default=0.10)
    args = parser.parse_args()

    run_walk_forward(
        csv_path   = DATA_FILES[args.tf],
        n_windows  = args.n_windows,
        test_frac  = args.test_frac,
    )