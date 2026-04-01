"""
config.py  —  Central configuration for XAU/USD Transformer + Meta-Label pipeline.
All paths, hyperparameters, and flags live here. Edit this file only.
"""

from pathlib import Path

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
BASE_DIR   = Path(__file__).parent
DATA_DIR   = BASE_DIR / "data"
MODEL_DIR  = BASE_DIR / "models"
LOG_DIR    = BASE_DIR / "logs"
RESULT_DIR = BASE_DIR / "results"

for d in [DATA_DIR, MODEL_DIR, LOG_DIR, RESULT_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Merged feature CSVs produced by your feature engineering step
DATA_FILES = {
    "1min": DATA_DIR / "features_1min.csv",
    "3min": DATA_DIR / "features_3min.csv",
    "5min": DATA_DIR / "features_5min.csv",
}

# Primary timeframe used for training (3min recommended — less noise)
PRIMARY_TF = "3min"

# ─────────────────────────────────────────────
#  FEATURE COLUMNS
#  Adjust if your CSV column names differ.
# ─────────────────────────────────────────────
OHLCV_COLS = ["Open", "High", "Low", "Close", "Volume"]

# All engineered feature columns (subset of what your FE script produces).
# The pipeline will auto-detect these by dropping OHLCV + timestamp columns.
EXCLUDE_COLS = ["UTC", "timestamp", "datetime", "Open", "High", "Low", "Close", "Volume",
                "label_primary", "label_meta"]

# ─────────────────────────────────────────────
#  LABELLING
# ─────────────────────────────────────────────
LABEL_METHOD = "triple_barrier"     # "triple_barrier" | "fixed_horizon"
HORIZON_BARS = 3                    # bars ahead for fixed-horizon label
TP_MULT      = 1.5                  # take-profit  = TP_MULT  × ATR(14)
SL_MULT      = 1.0                  # stop-loss    = SL_MULT  × ATR(14)
MAX_HOLD     = 12                   # max bars to hold before time-exit
FLAT_BAND    = 0.0003               # |return| < FLAT_BAND → label 0 (flat)
                                    # only used in fixed_horizon mode

# ─────────────────────────────────────────────
#  SEQUENCE / DATASET
# ─────────────────────────────────────────────
SEQ_LEN      = 60                   # lookback window in bars
STEP_SIZE    = 1                    # stride between windows (1 = every bar)
TRAIN_RATIO  = 0.70
VAL_RATIO    = 0.15
# test = remaining 0.15 — always the most recent data (time-ordered)

# ─────────────────────────────────────────────
#  PRIMARY TRANSFORMER
# ─────────────────────────────────────────────
D_MODEL      = 128                  # embedding dimension
N_HEADS      = 8                    # attention heads  (D_MODEL % N_HEADS == 0)
N_LAYERS     = 4                    # encoder layers
D_FF         = 256                  # feedforward inner dim
DROPOUT      = 0.1
NUM_CLASSES  = 3                    # up / flat / down

# Training
BATCH_SIZE   = 512
EPOCHS       = 60
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 10                   # early-stopping patience (val loss)
LR_T_MAX     = 30                   # CosineAnnealingLR period

PRIMARY_MODEL_PATH = MODEL_DIR / "primary_transformer.pt"

# ─────────────────────────────────────────────
#  OUT-OF-FOLD SETTINGS (for meta-label gen)
# ─────────────────────────────────────────────
OOF_FOLDS    = 5

# ─────────────────────────────────────────────
#  META-LABEL MODEL  (LightGBM)
# ─────────────────────────────────────────────
META_MODEL_PATH = MODEL_DIR / "meta_lgbm.txt"

LGBM_PARAMS = {
    "objective":        "binary",
    "metric":           "binary_logloss",
    "boosting_type":    "gbdt",
    "n_estimators":     800,
    "learning_rate":    0.02,
    "num_leaves":       63,
    "max_depth":        -1,
    "min_child_samples": 40,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq":     5,
    "reg_alpha":        0.1,
    "reg_lambda":       0.1,
    "class_weight":     "balanced",
    "n_jobs":           -1,
    "random_state":     42,
    "verbose":          -1,
}

# ─────────────────────────────────────────────
#  INFERENCE GATE
# ─────────────────────────────────────────────
META_THRESHOLD   = 0.55             # p(correct) >= this → trade
# Set to None to auto-tune on validation set (recommended)
AUTO_TUNE_THRESH = True

# ─────────────────────────────────────────────
#  POSITION SIZING
# ─────────────────────────────────────────────
# Scale position size by meta-model confidence
USE_META_SIZING  = True
BASE_LOT         = 1.0              # base lot size (normalised)
MAX_LOT_MULT     = 2.0              # cap multiplier

# ─────────────────────────────────────────────
#  MISC
# ─────────────────────────────────────────────
SEED         = 42
DEVICE       = "cuda"               # "cuda" | "cpu"  — auto-fallback handled in code
NUM_WORKERS  = 4                    # DataLoader workers