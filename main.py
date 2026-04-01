"""
main.py  —  End-to-end orchestration of the XAU/USD Transformer + Meta-Label pipeline.

Usage:
    python main.py --mode train           # full train pipeline
    python main.py --mode eval            # evaluate saved models on test set
    python main.py --mode inference       # run inference on new data (CSV path via --data)
    python main.py --mode train --tf 5min # train on 5min timeframe

Steps when mode=train:
  1.  Load & label data
  2.  Time-ordered split
  3.  Fit scaler on train
  4.  Build PyTorch datasets
  5.  Train primary transformer (train+val)
  6.  Generate out-of-fold predictions on train set
  7.  Build meta-features + meta-labels
  8.  Train LightGBM meta-model
  9.  Tune threshold τ on validation set
  10. Evaluate full two-stage system on test set
  11. Save all artefacts
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# ─── Local imports ────────────────────────────────────────────────────────────
from config import (
    DATA_FILES, PRIMARY_TF, PRIMARY_MODEL_PATH, META_MODEL_PATH,
    RESULT_DIR, BATCH_SIZE, NUM_WORKERS, SEQ_LEN,
    AUTO_TUNE_THRESH, META_THRESHOLD, SEED,
)
from pipeline_data import (
    load_and_prepare, time_split, fit_scaler, apply_scaler,
    SlidingWindowDataset, build_meta_features, LABEL_MAP, INV_LABEL_MAP,
)
from model import PrimaryTransformer
from trainer import (
    get_device, train_primary, generate_oof_predictions, batch_predict,
)
from meta_model import (
    train_meta_model, tune_threshold, apply_signal_gate,
    compute_position_sizes, evaluate_combined,
)

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/pipeline.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
#  TRAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_train(tf: str = PRIMARY_TF):
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    log.info(f"\n{'═'*20}")
    log.info(f"  XAU/USD TRANSFORMER + META-LABEL — TRAINING ({tf})")
    log.info(f"{'═'*20}\n")

    device = get_device()

    # ── 1. Load & label ───────────────────────────────────────────────────
    csv_path = DATA_FILES[tf]
    df, feat_cols = load_and_prepare(csv_path)

    # ── 2. Time-split ─────────────────────────────────────────────────────
    df_train, df_val, df_test = time_split(df)

    # ── 3. Fit scaler on train only ───────────────────────────────────────
    scaler     = fit_scaler(df_train, feat_cols)
    df_train_s = apply_scaler(df_train, feat_cols, scaler)
    df_val_s   = apply_scaler(df_val,   feat_cols, scaler)
    df_test_s  = apply_scaler(df_test,  feat_cols, scaler)

    # Save scaler for inference
    import joblib
    joblib.dump(scaler, "models/scaler.pkl")
    log.info("  Scaler saved → models/scaler.pkl")

    n_features = len(feat_cols)
    log.info(f"  n_features = {n_features}")

    # ── 4. PyTorch datasets ───────────────────────────────────────────────
    train_ds = SlidingWindowDataset(df_train_s, feat_cols)
    val_ds   = SlidingWindowDataset(df_val_s,   feat_cols)
    test_ds  = SlidingWindowDataset(df_test_s,  feat_cols)
    log.info(f"  Datasets — train={len(train_ds):,}  "
             f"val={len(val_ds):,}  test={len(test_ds):,}")

    # ── 5. Train primary transformer ──────────────────────────────────────
    log.info("\n── STAGE 1: PRIMARY TRANSFORMER ──────────────────────────")
    primary_model = PrimaryTransformer(n_features=n_features).to(device)
    history = train_primary(primary_model, train_ds, val_ds, device)
    pd.DataFrame(history).to_csv(RESULT_DIR / "training_history.csv", index=False)

    # Reload best checkpoint
    ckpt = torch.load(PRIMARY_MODEL_PATH, map_location=device)
    primary_model.load_state_dict(ckpt["model_state"])

    # ── 6. Out-of-fold predictions on TRAIN set ───────────────────────────
    log.info("\n── OOF PREDICTIONS FOR META-LABEL CONSTRUCTION ──────────")
    oof_probs, oof_preds, valid_mask = generate_oof_predictions(
        df_train_s, feat_cols, n_features, device
    )

    # Align valid mask (rows in df_train_s that have OOF predictions)
    df_train_oof  = df_train_s.iloc[valid_mask].reset_index(drop=True)
    oof_probs_v   = oof_probs[valid_mask]
    oof_preds_v   = oof_preds[valid_mask]

    # ── 7. Build meta features + labels ───────────────────────────────────
    log.info("\n── META-LABEL CONSTRUCTION ───────────────────────────────")
    X_meta_train, y_meta_train = build_meta_features(
        df_train_oof, feat_cols, oof_probs_v, oof_preds_v
    )

    # Validation: run primary model to get val probs
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE * 2,
                            shuffle=False, num_workers=NUM_WORKERS)
    val_probs, val_preds_s1, val_true_s1 = batch_predict(
        primary_model, val_loader, device
    )

    X_meta_val, y_meta_val = build_meta_features(
        df_val_s.iloc[SEQ_LEN - 1:].reset_index(drop=True),
        feat_cols,
        val_probs,
        val_preds_s1,
    )

    # ── 8. Train meta-label model ─────────────────────────────────────────
    log.info("\n── STAGE 2: META-LABEL MODEL ────────────────────────────")
    meta_model = train_meta_model(X_meta_train, y_meta_train,
                                  X_meta_val,   y_meta_val)

    # ── 9. Tune threshold on validation set ───────────────────────────────
    log.info("\n── THRESHOLD TUNING ─────────────────────────────────────")
    meta_val_probs = meta_model.predict_proba(X_meta_val)[:, 1]

    # Next-bar returns for Sharpe computation
    val_returns = (
        df_val_s["Close"].diff(1).shift(-1)
        .iloc[SEQ_LEN - 1:].reset_index(drop=True).values
    )
    val_returns = val_returns[:len(meta_val_probs)]

    if AUTO_TUNE_THRESH:
        best_tau, sweep_df = tune_threshold(
            meta_val_probs, y_meta_val, val_returns, val_preds_s1
        )
        sweep_df.to_csv(RESULT_DIR / "threshold_sweep.csv", index=False)
    else:
        best_tau = META_THRESHOLD
        log.info(f"  Using fixed threshold τ = {best_tau}")

    # Save tau
    pd.Series({"tau": best_tau}).to_json("models/tau.json")

    # ── 10. Evaluate on test set ──────────────────────────────────────────
    log.info("\n── FINAL EVALUATION ON TEST SET ─────────────────────────")
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE * 2,
                             shuffle=False, num_workers=NUM_WORKERS)
    test_probs_s1, test_preds_s1, test_true_s1 = batch_predict(
        primary_model, test_loader, device
    )

    df_test_aligned = df_test_s.iloc[SEQ_LEN - 1:].reset_index(drop=True)
    X_meta_test, y_meta_test = build_meta_features(
        df_test_aligned, feat_cols, test_probs_s1, test_preds_s1
    )

    meta_test_probs = meta_model.predict_proba(X_meta_test)[:, 1]
    test_returns    = (
        df_test_s["Close"].diff(1).shift(-1)
        .iloc[SEQ_LEN - 1:].reset_index(drop=True).values
    )
    test_returns = test_returns[:len(meta_test_probs)]

    report = evaluate_combined(
        test_preds_s1, test_true_s1,
        meta_test_probs, y_meta_test,
        test_returns, best_tau,
    )
    report.to_csv(RESULT_DIR / "evaluation_report.csv")
    log.info(f"  Report saved → {RESULT_DIR / 'evaluation_report.csv'}")

    # ── Save all artefacts metadata ───────────────────────────────────────
    meta_info = {
        "timeframe":   tf,
        "n_features":  n_features,
        "feat_cols":   feat_cols,
        "tau":         best_tau,
        "seq_len":     SEQ_LEN,
    }
    import json
    with open("models/meta_info.json", "w") as f:
        json.dump({k: v for k, v in meta_info.items() if k != "feat_cols"}, f, indent=2)
    pd.Series(feat_cols).to_csv("models/feat_cols.csv", index=False, header=False)

    log.info("\n✓ Training pipeline complete.")
    return primary_model, meta_model, best_tau, report


# ─────────────────────────────────────────────────────────────────────────────
#  INFERENCE  (single bar / streaming)
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(new_data_csv: str,
                  primary_model_path: Path = PRIMARY_MODEL_PATH,
                  meta_model_path:    Path = META_MODEL_PATH):
    """
    Given a CSV of new bars (same feature schema as training data),
    produce trading signals with position sizes.
    """
    import json, joblib
    import lightgbm as lgb

    device = get_device()

    # Load artefacts
    with open("models/meta_info.json") as f:
        meta_info = json.load(f)
    feat_cols  = pd.read_csv("models/feat_cols.csv", header=None)[0].tolist()
    tau        = meta_info.get("tau", META_THRESHOLD)
    n_features = meta_info["n_features"]
    scaler     = joblib.load("models/scaler.pkl")

    ckpt = torch.load(primary_model_path, map_location=device)
    primary_model = PrimaryTransformer(n_features=n_features).to(device)
    primary_model.load_state_dict(ckpt["model_state"])
    primary_model.eval()

    meta_booster = lgb.Booster(model_file=str(meta_model_path))

    # Load & preprocess new data
    df, _ = load_and_prepare(Path(new_data_csv))
    df    = apply_scaler(df, feat_cols, scaler)

    infer_ds = SlidingWindowDataset(df, feat_cols)
    loader   = DataLoader(infer_ds, batch_size=BATCH_SIZE * 2,
                          shuffle=False, num_workers=NUM_WORKERS)

    with torch.no_grad():
        all_probs, all_preds = [], []
        for X, _ in loader:
            probs = primary_model.predict_proba(X.to(device)).cpu().numpy()
            all_probs.append(probs)
            all_preds.append(probs.argmax(axis=1))

    s1_probs = np.concatenate(all_probs)
    s1_preds = np.concatenate(all_preds)

    df_aligned = df.iloc[SEQ_LEN - 1:].reset_index(drop=True)
    X_meta, _  = build_meta_features(df_aligned, feat_cols, s1_probs, s1_preds)
    meta_probs  = meta_booster.predict(X_meta)

    signals = apply_signal_gate(s1_preds, meta_probs, tau)
    sizes   = compute_position_sizes(meta_probs, signals, tau)

    # Build output DataFrame
    label_names = {0: "SHORT", 1: "FLAT", 2: "LONG"}
    result = pd.DataFrame({
        "UTC":             df_aligned["UTC"].values if "UTC" in df_aligned else range(len(signals)),
        "primary_signal":  [label_names[p] for p in s1_preds],
        "meta_confidence": np.round(meta_probs, 4),
        "final_signal":    [label_names[s] for s in signals],
        "position_size":   np.round(sizes, 4),
    })

    out_path = RESULT_DIR / "inference_signals.csv"
    result.to_csv(out_path, index=False)
    log.info(f"Inference complete → {out_path}")
    log.info(f"\n{result.tail(10).to_string()}")
    return result


# ─────────────────────────────────────────────────────────────────────────────
#  CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="XAU/USD Transformer + Meta-Label Pipeline"
    )
    parser.add_argument("--mode", choices=["train", "eval", "inference"],
                        default="train")
    parser.add_argument("--tf",   default=PRIMARY_TF,
                        help="Timeframe: 1min | 3min | 5min")
    parser.add_argument("--data", default=None,
                        help="Path to new CSV for inference mode")
    args = parser.parse_args()

    if args.mode == "train":
        run_train(tf=args.tf)

    elif args.mode == "inference":
        if args.data is None:
            parser.error("--data is required for inference mode")
        run_inference(args.data)