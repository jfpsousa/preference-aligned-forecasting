"""
Forecasting models.

* ``TPOForecaster``   -- the shared Transformer backbone used by the MSE baseline,
  the Asymmetric-Loss baseline, the Oracle and Temporal-DPO (manuscript Sec. 7.1:
  4 layers, 8 heads, model dimension 256, feed-forward 1024, dropout 0.1,
  approx. 3.3M parameters). It outputs a Gaussian per horizon step (mean and
  variance) because Temporal-DPO needs a likelihood pi(y_h | context).
* ``PatchTST`` and ``Informer`` -- independent re-implementations from the
  published descriptions (Nie et al., ICLR 2023; Zhou et al., AAAI 2021), sized
  to the same budget. They are not the original authors' code.

All models map a context window (batch, L, C) to a dict with
``mean`` (batch, H) and ``var`` (batch, H) for the target channel 0.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================================================
# Shared pieces
# ==========================================================================

class _PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div[:pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ==========================================================================
# PatchTST
# ==========================================================================

class PatchTST(nn.Module):
    """
    Patch-based transformer forecaster (Nie et al., ICLR 2023).

    The two defining ideas of the paper are implemented here:

    1. **Patching.** The input series is split into overlapping subseries of
       length ``patch_len`` taken with stride ``stride``. Each patch becomes one
       token, so attention operates over ~L/S tokens instead of L, and each
       token carries local semantic content rather than a single time step.
    2. **Channel independence.** Every input channel is processed by the same
       backbone as a separate univariate series, with no cross-channel
       attention. Channels are folded into the batch dimension.

    Forecasts are produced by flattening the encoded patches and applying a
    linear head, as in the paper (no autoregressive decoding).
    """

    def __init__(
        self,
        input_dim: int,
        context_length: int,
        forecast_horizon: int,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.context_length = context_length
        self.forecast_horizon = forecast_horizon
        self.patch_len = patch_len
        self.stride = stride

        # Number of patches after replication-padding the tail by one stride,
        # following the paper's end-padding scheme.
        self.num_patches = (max(context_length, patch_len) - patch_len) // stride + 2

        self.patch_embed = nn.Linear(patch_len, d_model)
        self.pos = _PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, norm_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False)
        self.head = nn.Linear(d_model * self.num_patches, forecast_horizon)

    def _patch(self, x: torch.Tensor) -> torch.Tensor:
        """(B*C, L) -> (B*C, num_patches, patch_len)."""
        # Replication-pad the end by one stride so the final window is complete.
        x = F.pad(x.unsqueeze(1), (0, self.stride), mode="replicate").squeeze(1)
        patches = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        if patches.size(1) < self.num_patches:      # pad short inputs
            deficit = self.num_patches - patches.size(1)
            patches = torch.cat([patches, patches[:, -1:].repeat(1, deficit, 1)], dim=1)
        return patches[:, : self.num_patches]

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, context_length, input_dim)
        Returns:
            dict with ``forecasts`` of shape (batch, forecast_horizon, input_dim),
            matching TemporalTransformer's interface.
        """
        b, _, c = x.shape
        # Channel independence: fold channels into the batch.
        z = x.permute(0, 2, 1).reshape(b * c, -1)          # (B*C, L)

        # Instance normalisation (RevIN-style), per the paper.
        mu = z.mean(dim=-1, keepdim=True)
        sigma = z.std(dim=-1, keepdim=True) + 1e-5
        z = (z - mu) / sigma

        tok = self.dropout(self.pos(self.patch_embed(self._patch(z))))
        enc = self.encoder(tok)                             # (B*C, P, D)
        out = self.head(enc.reshape(enc.size(0), -1))       # (B*C, H)

        out = out * sigma + mu                              # de-normalise
        forecasts = out.reshape(b, c, self.forecast_horizon).permute(0, 2, 1)
        return {"forecasts": forecasts, "uncertainties": torch.zeros_like(forecasts)}


# ==========================================================================
# Informer
# ==========================================================================

class _ProbSparseAttention(nn.Module):
    """
    ProbSparse self-attention (Zhou et al., AAAI 2021, Section 3.1).

    Only the ``u = c * ln(L_Q)`` queries with the highest sparsity measure
    attend fully; the rest are replaced by the mean of the values, which is the
    paper's approximation for near-uniform attention rows. This is what gives
    Informer its O(L log L) complexity.
    """

    def __init__(self, d_model: int, num_heads: int, factor: int = 5,
                 dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.h = num_heads
        self.dk = d_model // num_heads
        self.factor = factor
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q_in: torch.Tensor, kv_in: Optional[torch.Tensor] = None
                ) -> torch.Tensor:
        kv_in = q_in if kv_in is None else kv_in
        b, lq, _ = q_in.shape
        lk = kv_in.size(1)

        q = self.q_proj(q_in).view(b, lq, self.h, self.dk).transpose(1, 2)
        k = self.k_proj(kv_in).view(b, lk, self.h, self.dk).transpose(1, 2)
        v = self.v_proj(kv_in).view(b, lk, self.h, self.dk).transpose(1, 2)

        u = min(lq, max(1, int(self.factor * math.ceil(math.log(max(lk, 2))))))
        sample = min(lk, max(1, int(self.factor * math.ceil(math.log(max(lq, 2))))))

        # Sparsity measure M(q, K) ~= max_j (q.k_j) - mean_j (q.k_j), estimated
        # on a random subset of keys as in the paper.
        idx = torch.randint(lk, (sample,), device=q.device)
        scores_sample = torch.einsum("bhqd,bhkd->bhqk", q, k[:, :, idx]) / math.sqrt(self.dk)
        sparsity = scores_sample.max(dim=-1).values - scores_sample.mean(dim=-1)
        top = sparsity.topk(u, dim=-1).indices                       # (B, H, u)

        q_top = torch.gather(
            q, 2, top.unsqueeze(-1).expand(-1, -1, -1, self.dk))     # (B, H, u, dk)
        scores = torch.einsum("bhud,bhkd->bhuk", q_top, k) / math.sqrt(self.dk)
        attn = self.dropout(torch.softmax(scores, dim=-1))
        ctx_top = torch.einsum("bhuk,bhkd->bhud", attn, v)

        # Non-selected queries take the mean of V (the paper's lazy fill).
        ctx = v.mean(dim=2, keepdim=True).expand(-1, -1, lq, -1).clone()
        ctx.scatter_(2, top.unsqueeze(-1).expand(-1, -1, -1, self.dk), ctx_top)

        ctx = ctx.transpose(1, 2).reshape(b, lq, self.h * self.dk)
        return self.out(ctx)


class _DistillLayer(nn.Module):
    """Self-attention distilling: conv + max-pool halving the sequence length."""

    def __init__(self, d_model: int):
        super().__init__()
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.norm = nn.BatchNorm1d(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x.transpose(1, 2))
        y = F.elu(self.norm(y))
        return F.max_pool1d(y, kernel_size=3, stride=2, padding=1).transpose(1, 2)


class _InformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float,
                 factor: int):
        super().__init__()
        self.attn = _ProbSparseAttention(d_model, num_heads, factor, dropout)
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.n1(x + self.drop(self.attn(x)))
        return self.n2(x + self.ff(x))


class Informer(nn.Module):
    """
    Informer (Zhou et al., AAAI 2021).

    Implements the three contributions of the paper: ProbSparse self-attention,
    self-attention distilling between encoder layers, and a generative-style
    decoder that emits the whole horizon in one forward pass rather than
    autoregressively.
    """

    def __init__(
        self,
        input_dim: int,
        context_length: int,
        forecast_horizon: int,
        d_model: int = 256,
        num_heads: int = 8,
        num_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        factor: int = 5,
        label_len: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.forecast_horizon = forecast_horizon
        # Length of the known prefix the decoder is seeded with.
        self.label_len = label_len or max(1, context_length // 2)

        self.enc_embed = nn.Linear(input_dim, d_model)
        self.dec_embed = nn.Linear(input_dim, d_model)
        self.pos = _PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)

        self.enc_layers = nn.ModuleList([
            _InformerEncoderLayer(d_model, num_heads, d_ff, dropout, factor)
            for _ in range(num_layers)
        ])
        # One distilling step between consecutive encoder layers.
        self.distil = nn.ModuleList([
            _DistillLayer(d_model) for _ in range(max(0, num_layers - 1))
        ])

        self.dec_self = _ProbSparseAttention(d_model, num_heads, factor, dropout)
        self.dec_cross = _ProbSparseAttention(d_model, num_heads, factor, dropout)
        self.dn1 = nn.LayerNorm(d_model)
        self.dn2 = nn.LayerNorm(d_model)
        self.dn3 = nn.LayerNorm(d_model)
        self.dec_ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.projection = nn.Linear(d_model, input_dim)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch, context_length, input_dim)
        Returns:
            dict with ``forecasts`` of shape (batch, forecast_horizon, input_dim).
        """
        b = x.size(0)

        enc = self.dropout(self.pos(self.enc_embed(x)))
        for i, layer in enumerate(self.enc_layers):
            enc = layer(enc)
            if i < len(self.distil):
                enc = self.distil[i](enc)

        # Generative decoder input: known label prefix + zero placeholders.
        label = x[:, -self.label_len:, :]
        placeholder = torch.zeros(
            b, self.forecast_horizon, self.input_dim, device=x.device, dtype=x.dtype)
        dec_in = torch.cat([label, placeholder], dim=1)

        dec = self.dropout(self.pos(self.dec_embed(dec_in)))
        dec = self.dn1(dec + self.dec_self(dec))
        dec = self.dn2(dec + self.dec_cross(dec, enc))
        dec = self.dn3(dec + self.dec_ff(dec))

        forecasts = self.projection(dec)[:, -self.forecast_horizon:, :]
        return {"forecasts": forecasts, "uncertainties": torch.zeros_like(forecasts)}


# ==========================================================================
# Shared Temporal-DPO backbone
# ==========================================================================

class TPOForecaster(nn.Module):
    """
    Transformer encoder over patches of the multivariate context window.

    The context (L time steps, C channels) is cut into patches of ``patch_len``
    steps with stride ``stride``; each patch (patch_len * C values) is linearly
    embedded into one token. Tokens pass through a standard Transformer encoder,
    are mean-pooled, and two heads emit the per-horizon Gaussian mean and
    variance of the target channel.
    """

    def __init__(self, input_dim: int, context_length: int, forecast_horizon: int,
                 d_model: int = 256, num_heads: int = 8, num_layers: int = 4,
                 d_ff: int = 1024, dropout: float = 0.1,
                 patch_len: int = 16, stride: int = 8):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        self.forecast_horizon = forecast_horizon
        self.num_patches = (context_length - patch_len) // stride + 1
        self.embed = nn.Linear(patch_len * input_dim, d_model)
        self.pos = _PositionalEncoding(d_model)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False)
        self.mean_head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, forecast_horizon))
        self.var_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, forecast_horizon), nn.Softplus())

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        b, _, c = x.shape
        p = x.unfold(1, self.patch_len, self.stride)          # (B, P, C, patch)
        p = p[:, : self.num_patches].reshape(b, self.num_patches, -1)
        z = self.encoder(self.drop(self.pos(self.embed(p))))
        pooled = z.mean(dim=1)
        return {"mean": self.mean_head(pooled),
                "var": self.var_head(pooled) + 1e-4}


class _ChannelZeroAdapter(nn.Module):
    """Expose PatchTST / Informer through the {mean, var} interface."""

    def __init__(self, net: nn.Module):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        f = self.net(x)["forecasts"][:, :, 0]
        return {"mean": f, "var": torch.ones_like(f)}


def build_model(kind: str, input_dim: int, context_length: int, horizon: int,
                arch: dict) -> nn.Module:
    """Construct a model of the given kind with the shared architecture budget."""
    common = dict(d_model=arch["d_model"], num_heads=arch["num_heads"],
                  num_layers=arch["num_layers"], d_ff=arch["d_ff"],
                  dropout=arch["dropout"])
    if kind == "tpo":
        return TPOForecaster(input_dim, context_length, horizon,
                             patch_len=arch["patch_len"], stride=arch["stride"],
                             **common)
    if kind == "patchtst":
        return _ChannelZeroAdapter(PatchTST(input_dim, context_length, horizon,
                                            patch_len=arch["patch_len"],
                                            stride=arch["stride"], **common))
    if kind == "informer":
        return _ChannelZeroAdapter(Informer(input_dim, context_length, horizon,
                                            **common))
    raise ValueError(f"unknown model kind {kind!r}")
