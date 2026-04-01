"""
trainer.py  —  Training engine for the Primary Transformer.

Includes:
  - Standard train/val loop with early stopping
  - Class-weighted loss (handles imbalance in XAU/USD labels)
  - CosineAnnealingLR schedule
  - Out-of-fold (OOF) prediction generation for meta-label construction
  - Checkpoint saving / loading
"""

import logging
import time
from pathlib import Path
from typing import Tuple, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, ConcatDataset
from sklearn.model_selection import KFold

from config import (
    BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, PATIENCE, LR_T_MAX,
    OOF_FOLDS, DEVICE, NUM_WORKERS, SEED, PRIMARY_MODEL_PATH,
    NUM_CLASSES,
)
from pipeline_data import SlidingWindowDataset, LABEL_MAP
from model import PrimaryTransformer

log = logging.getLogger(__name__)


def get_device() -> torch.device:
    if DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    log.info("CUDA not available — using CPU.")
    return torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
#  CLASS WEIGHTS  (inverse frequency to handle label imbalance)
# ─────────────────────────────────────────────────────────────────────────────

def compute_class_weights(dataset: SlidingWindowDataset,
                          device: torch.device) -> torch.Tensor:
    labels = np.array([dataset.y_all[dataset.indices[i]] for i in range(len(dataset))])
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float32)
    weights = 1.0 / (counts + 1e-6)
    weights /= weights.sum()
    weights *= NUM_CLASSES          # scale so mean weight ≈ 1
    log.info(f"  Class weights: {weights.tolist()}")
    return torch.tensor(weights, dtype=torch.float32, device=device)


# ─────────────────────────────────────────────────────────────────────────────
#  SINGLE EPOCH
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model:     PrimaryTransformer,
              loader:    DataLoader,
              optimizer: Optional[torch.optim.Optimizer],
              criterion: nn.CrossEntropyLoss,
              device:    torch.device,
              train:     bool = True
              ) -> Tuple[float, float]:
    """Returns (avg_loss, accuracy)."""
    model.train() if train else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss   = criterion(logits, y)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item() * y.size(0)
            preds       = logits.argmax(dim=1)
            correct    += (preds == y).sum().item()
            total      += y.size(0)

    return total_loss / total, correct / total


# ─────────────────────────────────────────────────────────────────────────────
#  FULL TRAIN / VAL LOOP
# ─────────────────────────────────────────────────────────────────────────────

def train_primary(model:       PrimaryTransformer,
                  train_ds:    SlidingWindowDataset,
                  val_ds:      SlidingWindowDataset,
                  device:      torch.device,
                  save_path:   Path = PRIMARY_MODEL_PATH) -> dict:
    """
    Full training loop with early stopping.
    Returns dict of training history.
    """
    torch.manual_seed(SEED)

    class_weights = compute_class_weights(train_ds, device)
    criterion     = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=LR_T_MAX, eta_min=LR * 0.05
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  num_workers=NUM_WORKERS,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE * 2,
                              shuffle=False, num_workers=NUM_WORKERS,
                              pin_memory=True)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_loss  = float("inf")
    patience_count = 0

    log.info(f"Training on {device} — {len(train_ds):,} train / {len(val_ds):,} val samples")
    log.info(f"Model parameters: {model.count_parameters():,}")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, optimizer,
                                    criterion, device, train=True)
        va_loss, va_acc = run_epoch(model, val_loader,   None,
                                    criterion, device, train=False)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(va_acc)

        elapsed = time.time() - t0
        log.info(f"Epoch {epoch:03d}/{EPOCHS}  "
                 f"tr_loss={tr_loss:.4f}  va_loss={va_loss:.4f}  "
                 f"tr_acc={tr_acc:.3f}  va_acc={va_acc:.3f}  "
                 f"lr={scheduler.get_last_lr()[0]:.6f}  {elapsed:.1f}s")

        if va_loss < best_val_loss:
            best_val_loss  = va_loss
            patience_count = 0
            torch.save({"model_state": model.state_dict(),
                        "epoch": epoch,
                        "val_loss": va_loss,
                        "val_acc": va_acc},
                       save_path)
            log.info(f"  ✓ Saved best model → {save_path}")
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                log.info(f"Early stopping triggered at epoch {epoch}.")
                break

    # Reload best weights
    ckpt = torch.load(save_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    log.info(f"Best val_loss={best_val_loss:.4f} at epoch {ckpt['epoch']}")
    return history


# ─────────────────────────────────────────────────────────────────────────────
#  OUT-OF-FOLD PREDICTIONS  (for meta-label construction)
# ─────────────────────────────────────────────────────────────────────────────

def generate_oof_predictions(df_train,
                             feat_cols: List[str],
                             n_features: int,
                             device: torch.device,
                             n_folds: int = OOF_FOLDS
                             ) -> Tuple[np.ndarray, np.ndarray]:
    """
    K-fold cross-validation on the training set.
    Returns out-of-fold softmax probabilities and argmax predictions
    — these are used to build the meta-labels WITHOUT leakage.

    The SlidingWindowDataset is rebuilt per fold to respect temporal order:
    we always use earlier data to predict later data within the fold.
    (TimeSeriesSplit rather than random KFold for stricter validation.)

    Returns:
        oof_probs : [N_train_valid, 3]   softmax probabilities
        oof_preds : [N_train_valid]      predicted class indices
    """
    from sklearn.model_selection import TimeSeriesSplit
    from pipeline_data import SlidingWindowDataset, apply_scaler, fit_scaler, LABEL_MAP

    log.info(f"Generating OOF predictions with {n_folds}-fold TimeSeriesSplit ...")

    # We work on raw indices of df_train
    n       = len(df_train)
    tscv    = TimeSeriesSplit(n_splits=n_folds)

    # Allocate output arrays indexed over ROWS of df_train
    # (each row corresponds to the last bar in a potential window)
    oof_probs = np.full((n, NUM_CLASSES), np.nan, dtype=np.float32)
    oof_preds = np.full(n, -1, dtype=np.int64)

    for fold, (tr_idx, va_idx) in enumerate(tscv.split(np.arange(n)), 1):
        log.info(f"  Fold {fold}/{n_folds} — train={len(tr_idx):,}  val={len(va_idx):,}")

        fold_train_df = df_train.iloc[tr_idx].reset_index(drop=True)
        fold_val_df   = df_train.iloc[va_idx].reset_index(drop=True)

        # Fit scaler on fold train only
        scaler         = fit_scaler(fold_train_df, feat_cols)
        fold_train_df  = apply_scaler(fold_train_df, feat_cols, scaler)
        fold_val_df    = apply_scaler(fold_val_df,   feat_cols, scaler)

        tr_ds  = SlidingWindowDataset(fold_train_df, feat_cols)
        va_ds  = SlidingWindowDataset(fold_val_df,   feat_cols)

        # Build a fresh model for this fold
        fold_model = PrimaryTransformer(n_features=n_features).to(device)

        # Train fold model (fewer epochs for speed — early stopping still active)
        _fold_epochs_save = EPOCHS
        import config; config.EPOCHS = max(20, EPOCHS // 2)

        class_weights = compute_class_weights(tr_ds, device)
        criterion     = nn.CrossEntropyLoss(weight=class_weights)
        optimizer     = torch.optim.AdamW(fold_model.parameters(),
                                          lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler     = torch.optim.lr_scheduler.CosineAnnealingLR(
                            optimizer, T_max=max(10, config.EPOCHS // 3))

        tr_loader = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,
                               num_workers=NUM_WORKERS, pin_memory=True)
        va_loader = DataLoader(va_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                               num_workers=NUM_WORKERS, pin_memory=True)

        best_fold_loss  = float("inf")
        best_fold_state = None
        p_count         = 0

        for ep in range(1, config.EPOCHS + 1):
            run_epoch(fold_model, tr_loader, optimizer, criterion, device, True)
            va_loss, _ = run_epoch(fold_model, va_loader, None, criterion, device, False)
            scheduler.step()
            if va_loss < best_fold_loss:
                best_fold_loss  = va_loss
                best_fold_state = {k: v.cpu().clone()
                                   for k, v in fold_model.state_dict().items()}
                p_count = 0
            else:
                p_count += 1
                if p_count >= PATIENCE:
                    break

        config.EPOCHS = _fold_epochs_save
        fold_model.load_state_dict(best_fold_state)
        fold_model.eval()

        # Predict on validation fold
        all_probs, all_preds = [], []
        with torch.no_grad():
            for X, _ in va_loader:
                probs = fold_model.predict_proba(X.to(device)).cpu().numpy()
                all_probs.append(probs)
                all_preds.append(probs.argmax(axis=1))

        fold_probs = np.concatenate(all_probs, axis=0)
        fold_preds = np.concatenate(all_preds, axis=0)

        # Map back: va_ds indices correspond to rows va_idx[seq_len-1:]
        # (first seq_len-1 rows of each fold slice have no complete window)
        from config import SEQ_LEN
        valid_va_rows = va_idx[SEQ_LEN - 1:]
        if len(valid_va_rows) > len(fold_probs):
            valid_va_rows = valid_va_rows[:len(fold_probs)]
        elif len(valid_va_rows) < len(fold_probs):
            fold_probs = fold_probs[:len(valid_va_rows)]
            fold_preds = fold_preds[:len(valid_va_rows)]

        oof_probs[valid_va_rows] = fold_probs
        oof_preds[valid_va_rows] = fold_preds

        log.info(f"  Fold {fold} val_loss={best_fold_loss:.4f}")

    # Remove rows where OOF predictions weren't generated (first fold's warmup)
    valid_mask = oof_preds != -1
    log.info(f"  OOF coverage: {valid_mask.sum():,} / {n:,} rows "
             f"({valid_mask.mean():.1%})")

    return oof_probs, oof_preds, valid_mask


# ─────────────────────────────────────────────────────────────────────────────
#  BATCH INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def batch_predict(model:  PrimaryTransformer,
                  loader: DataLoader,
                  device: torch.device
                  ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model on a DataLoader.
    Returns (probs [N,3], preds [N], true_labels [N]).
    """
    model.eval()
    all_probs, all_preds, all_true = [], [], []
    for X, y in loader:
        probs = model.predict_proba(X.to(device)).cpu().numpy()
        all_probs.append(probs)
        all_preds.append(probs.argmax(axis=1))
        all_true.append(y.numpy())
    return (np.concatenate(all_probs),
            np.concatenate(all_preds),
            np.concatenate(all_true))
