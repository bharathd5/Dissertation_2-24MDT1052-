"""
meta_model.py  —  Stage 2 Meta-Label Model (LightGBM binary classifier).

Predicts: will the primary transformer's signal be correct? (binary 0/1)

Includes:
  - LightGBM training with early stopping on val set
  - Precision-recall threshold sweep on validation set
  - Sharpe-optimal threshold selection
  - Position sizing by meta confidence score
  - Feature importance logging
"""

import logging
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, classification_report,
)

from config import (
    LGBM_PARAMS, META_MODEL_PATH, AUTO_TUNE_THRESH, META_THRESHOLD,
    USE_META_SIZING, BASE_LOT, MAX_LOT_MULT, SEED,
)
from pipeline_data import INV_LABEL_MAP

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  TRAIN META MODEL
# ─────────────────────────────────────────────────────────────────────────────

def train_meta_model(X_train: np.ndarray, y_train: np.ndarray,
                     X_val:   np.ndarray, y_val:   np.ndarray,
                     save_path: Path = META_MODEL_PATH
                     ) -> lgb.LGBMClassifier:
    """
    Train LightGBM binary classifier on meta-features.
    Uses eval set for early stopping.
    """
    log.info(f"Training meta-label model  "
             f"[train={len(y_train):,}  val={len(y_val):,}]")
    log.info(f"  Positive rate — train: {y_train.mean():.3f}  "
             f"val: {y_val.mean():.3f}")

    model = lgb.LGBMClassifier(**LGBM_PARAMS)

    callbacks = [
        lgb.early_stopping(stopping_rounds=50, verbose=False),
        lgb.log_evaluation(period=100),
    ]

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric="binary_logloss",
        callbacks=callbacks,
    )

    model.booster_.save_model(str(save_path))
    log.info(f"  Meta model saved → {save_path}")
    log.info(f"  Best iteration: {model.best_iteration_}")

    # Feature importance (top 20)
    imp = pd.Series(model.feature_importances_).nlargest(20)
    log.info(f"  Top feature importances:\n{imp.to_string()}")

    return model


def load_meta_model(path: Path = META_MODEL_PATH) -> lgb.Booster:
    return lgb.Booster(model_file=str(path))


# ─────────────────────────────────────────────────────────────────────────────
#  THRESHOLD TUNING
# ─────────────────────────────────────────────────────────────────────────────

def tune_threshold(meta_probs:  np.ndarray,      # [N] p(signal_correct)
                   true_meta:   np.ndarray,       # [N] binary ground truth
                   primary_returns: np.ndarray,   # [N] next-bar returns (for Sharpe)
                   primary_preds:   np.ndarray,   # [N] predicted direction {0,1,2}
                   ) -> Tuple[float, pd.DataFrame]:
    """
    Sweep τ over [0.40, 0.85] and compute for each:
      - precision, recall, F1  on meta labels
      - Sharpe ratio of the filtered trading signals
      - trade count

    Returns:
        best_tau : float  — threshold maximising Sharpe
        sweep_df : DataFrame — full sweep results for analysis
    """
    thresholds = np.arange(0.40, 0.86, 0.01)
    records    = []

    for tau in thresholds:
        mask = meta_probs >= tau
        n_trades = mask.sum()

        if n_trades < 10:
            records.append({"tau": tau, "n_trades": n_trades,
                             "precision": np.nan, "recall": np.nan,
                             "f1": np.nan, "sharpe": np.nan})
            continue

        # Meta-label metrics
        prec = precision_score(true_meta[mask], (meta_probs[mask] >= tau).astype(int),
                               zero_division=0)
        rec  = recall_score(true_meta, mask.astype(int), zero_division=0)
        f1   = f1_score(true_meta[mask], (meta_probs[mask] >= tau).astype(int),
                        zero_division=0)

        # Trading Sharpe on filtered signals
        # primary_preds: 0=short, 1=flat, 2=long
        direction  = np.where(primary_preds[mask] == 2,  1.0,
                    np.where(primary_preds[mask] == 0, -1.0, 0.0))
        pnl        = direction * primary_returns[mask]
        sharpe     = (pnl.mean() / (pnl.std() + 1e-9)) * np.sqrt(252 * 1440)

        records.append({"tau": round(tau, 2), "n_trades": int(n_trades),
                        "precision": round(prec, 4),
                        "recall":    round(rec,  4),
                        "f1":        round(f1,   4),
                        "sharpe":    round(sharpe, 4)})

    sweep_df = pd.DataFrame(records)
    valid    = sweep_df.dropna(subset=["sharpe"])

    if valid.empty:
        log.warning("No valid threshold found — defaulting to 0.55")
        return 0.55, sweep_df

    best_tau = float(valid.loc[valid["sharpe"].idxmax(), "tau"])
    best_row = valid.loc[valid["sharpe"].idxmax()]

    log.info(f"\nThreshold sweep (top 5 by Sharpe):\n"
             f"{valid.nlargest(5, 'sharpe').to_string(index=False)}")
    log.info(f"\n  ► Best τ = {best_tau:.2f}  "
             f"Sharpe={best_row['sharpe']:.3f}  "
             f"Precision={best_row['precision']:.3f}  "
             f"Trades={int(best_row['n_trades']):,}")

    return best_tau, sweep_df


# ─────────────────────────────────────────────────────────────────────────────
#  INFERENCE GATE
# ─────────────────────────────────────────────────────────────────────────────

def apply_signal_gate(primary_preds:   np.ndarray,   # [N] {0,1,2}
                      meta_probs:      np.ndarray,   # [N] p(correct)
                      tau:             float,
                      ) -> np.ndarray:
    """
    Returns filtered signals:
      - If meta_prob >= tau AND primary pred is directional (not flat): keep signal
      - Otherwise: output 1 (flat / no trade)
    """
    directional = primary_preds != 1     # 0=short or 2=long
    trusted     = meta_probs >= tau
    signal      = np.where(directional & trusted, primary_preds, 1)
    n_trades    = (signal != 1).sum()
    log.info(f"  Signal gate  τ={tau:.2f}  "
             f"→ {n_trades:,} trades / {len(signal):,} bars "
             f"({n_trades/len(signal):.1%} of bars)")
    return signal


# ─────────────────────────────────────────────────────────────────────────────
#  POSITION SIZING
# ─────────────────────────────────────────────────────────────────────────────

def compute_position_sizes(meta_probs: np.ndarray,
                           signals:    np.ndarray,
                           tau:        float,
                           base_lot:   float = BASE_LOT,
                           max_mult:   float = MAX_LOT_MULT,
                           ) -> np.ndarray:
    """
    Scale position size linearly by meta confidence above threshold.

    size = base_lot × clamp( (p - τ) / (1 - τ) × max_mult + 1,  1,  max_mult )

    Flat signals get size 0.
    """
    sizes = np.zeros(len(meta_probs), dtype=np.float32)
    trade_mask = signals != 1

    if trade_mask.sum() == 0:
        return sizes

    p      = meta_probs[trade_mask]
    scale  = np.clip((p - tau) / (1.0 - tau + 1e-9), 0.0, 1.0)
    lot    = base_lot * (1.0 + scale * (max_mult - 1.0))
    sizes[trade_mask] = lot
    return sizes


# ─────────────────────────────────────────────────────────────────────────────
#  FULL EVALUATION REPORT
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_combined(primary_preds:  np.ndarray,
                      primary_true:   np.ndarray,
                      meta_probs:     np.ndarray,
                      true_meta:      np.ndarray,
                      next_bar_returns: np.ndarray,
                      tau:            float,
                      ) -> pd.DataFrame:
    """
    Full evaluation of the two-stage system on the test set.
    Returns a DataFrame of metrics.
    """
    signals = apply_signal_gate(primary_preds, meta_probs, tau)
    sizes   = compute_position_sizes(meta_probs, signals, tau)

    # ── Stage 1 performance (unfiltered) ──────────────────────────────────
    s1_acc = (primary_preds == primary_true).mean()

    # ── Stage 2 meta AUC ─────────────────────────────────────────────────
    try:
        meta_auc = roc_auc_score(true_meta, meta_probs)
    except Exception:
        meta_auc = np.nan

    # ── Filtered trading performance ──────────────────────────────────────
    trade_mask = signals != 1
    if trade_mask.sum() > 0:
        direction  = np.where(signals[trade_mask] == 2,  1.0, -1.0)
        pnl        = direction * next_bar_returns[trade_mask] * sizes[trade_mask]
        sharpe     = (pnl.mean() / (pnl.std() + 1e-9)) * np.sqrt(252 * 1440)
        cum_return = pnl.cumsum().iloc[-1] if isinstance(pnl, pd.Series) else pnl.cumsum()[-1]
        win_rate   = (pnl > 0).mean()
        # Max drawdown
        cum = np.cumsum(pnl)
        running_max = np.maximum.accumulate(cum)
        drawdowns   = running_max - cum
        max_dd      = drawdowns.max() if len(drawdowns) > 0 else 0.0
    else:
        sharpe = win_rate = cum_return = max_dd = np.nan

    metrics = {
        "Stage1_Accuracy_Unfiltered": round(s1_acc, 4),
        "Meta_AUC":                   round(meta_auc, 4) if not np.isnan(meta_auc) else "N/A",
        "Tau":                        tau,
        "N_Trades":                   int(trade_mask.sum()),
        "Trade_Rate_pct":             round(trade_mask.mean() * 100, 2),
        "Win_Rate":                   round(win_rate, 4) if not np.isnan(win_rate) else "N/A",
        "Sharpe_Annualised":          round(sharpe, 3)   if not np.isnan(sharpe) else "N/A",
        "Cumulative_Return":          round(cum_return, 4) if not np.isnan(cum_return) else "N/A",
        "Max_Drawdown":               round(max_dd, 4)   if not np.isnan(max_dd) else "N/A",
    }

    report_df = pd.DataFrame([metrics]).T.rename(columns={0: "Value"})
    log.info(f"\n{'═'*50}\nEVALUATION REPORT\n{'═'*50}\n{report_df.to_string()}")
    return report_df