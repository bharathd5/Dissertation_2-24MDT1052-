"""
model.py  —  Primary Transformer encoder for XAU/USD direction classification.

Architecture:
  Input  [B, T, F]
    → Linear projection  → [B, T, D_MODEL]
    → Positional encoding (learned)
    → N × TransformerEncoderLayer  (pre-norm, causal mask optional)
    → CLS-token pooling  (first token aggregates sequence)
    → MLP head  → [B, NUM_CLASSES]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import D_MODEL, N_HEADS, N_LAYERS, D_FF, DROPOUT, NUM_CLASSES, SEQ_LEN


# ─────────────────────────────────────────────────────────────────────────────
#  LEARNED POSITIONAL ENCODING
# ─────────────────────────────────────────────────────────────────────────────

class LearnedPositionalEncoding(nn.Module):
    """
    Learned position embeddings — more flexible than sinusoidal for
    financial time series where temporal patterns are non-stationary.
    """
    def __init__(self, seq_len: int, d_model: int, dropout: float):
        super().__init__()
        self.pos_embed = nn.Embedding(seq_len, d_model)
        self.dropout   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        T      = x.size(1)
        pos    = torch.arange(T, device=x.device)          # [T]
        x      = x + self.pos_embed(pos).unsqueeze(0)      # [B, T, D]
        return self.dropout(x)


# ─────────────────────────────────────────────────────────────────────────────
#  TRANSFORMER ENCODER BLOCK  (pre-LN for training stability)
# ─────────────────────────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """
    Pre-LayerNorm TransformerEncoderLayer.
    Pre-LN is more stable than post-LN on financial data with noisy gradients.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(
            embed_dim   = d_model,
            num_heads   = n_heads,
            dropout     = dropout,
            batch_first = True,     # expects [B, T, D]
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ff    = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor,
                attn_mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed,
                                attn_mask=attn_mask,
                                need_weights=False)
        x = x + attn_out

        # Feed-forward with residual
        x = x + self.ff(self.norm2(x))
        return x


# ─────────────────────────────────────────────────────────────────────────────
#  FULL PRIMARY TRANSFORMER
# ─────────────────────────────────────────────────────────────────────────────

class PrimaryTransformer(nn.Module):
    """
    Full classification transformer for XAU/USD direction prediction.

    Forward pass returns logits [B, NUM_CLASSES].
    Call .predict_proba(x) to get softmax probabilities.

    Parameters
    ----------
    n_features  : number of input features (auto-set from data)
    d_model     : embedding dimension
    n_heads     : number of attention heads
    n_layers    : number of transformer blocks
    d_ff        : feed-forward hidden dimension
    dropout     : dropout rate
    num_classes : 3  (short / flat / long)
    seq_len     : lookback window length (for positional encoding)
    """

    def __init__(self,
                 n_features:  int,
                 d_model:     int   = D_MODEL,
                 n_heads:     int   = N_HEADS,
                 n_layers:    int   = N_LAYERS,
                 d_ff:        int   = D_FF,
                 dropout:     float = DROPOUT,
                 num_classes: int   = NUM_CLASSES,
                 seq_len:     int   = SEQ_LEN):
        super().__init__()

        self.d_model = d_model

        # ── Input projection ──────────────────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
        )

        # ── CLS token ────────────────────────────────────────────────────
        # Prepended to sequence; its final hidden state is used for classification
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # ── Positional encoding  (seq_len + 1 for CLS) ───────────────────
        self.pos_enc = LearnedPositionalEncoding(seq_len + 1, d_model, dropout)

        # ── Transformer blocks ────────────────────────────────────────────
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        self.norm_out = nn.LayerNorm(d_model)

        # ── Classification head ───────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x : [B, T, F]
        returns logits : [B, num_classes]
        """
        B = x.size(0)

        # Project features → d_model
        x = self.input_proj(x)                              # [B, T, D]

        # Prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)              # [B, 1, D]
        x   = torch.cat([cls, x], dim=1)                    # [B, T+1, D]

        # Positional encoding
        x = self.pos_enc(x)                                 # [B, T+1, D]

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        x = self.norm_out(x)

        # CLS token output → classification head
        cls_out = x[:, 0, :]                                # [B, D]
        logits  = self.head(cls_out)                        # [B, C]
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Returns softmax probabilities [B, num_classes]."""
        self.eval()
        return F.softmax(self.forward(x), dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)