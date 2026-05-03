"""Perceiver-IO encoder.

Cross-attention from spike tokens into a learned latent array, followed by
self-attention within the latent array. This is the architectural core of POYO
and what makes the model scale gracefully across sessions with different
neuron counts.

Implementation notes for Claude Code:
    - Use einops throughout (`einops.einsum`, `rearrange`).
    - Use `torch.nn.functional.scaled_dot_product_attention` (SDPA) for the inner
      attention; it dispatches to FlashAttention 2 when supported.
    - In Phase 2, the cross-attention will be replaced with a custom Triton kernel
      that exploits the temporal sparsity of spike events.

Shapes:
    spike_tokens: (B, E, D)  variable E per batch element, padded with mask
    latents:      (B, L, D)  fixed L per config

Reference:
    Jaegle et al. 2021, "Perceiver IO: A General Architecture for Structured
    Inputs & Outputs"
"""

from __future__ import annotations

from typing import cast

import torch
from einops import rearrange
from torch import nn

from cortex.models.config import CortexConfig


class CrossAttentionBlock(nn.Module):
    """Cross-attention from input tokens into the latent array."""

    def __init__(self, config: CortexConfig) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.cross_attn_heads
        self.head_dim = config.latent_dim // config.cross_attn_heads

        if self.head_dim * self.num_heads != config.latent_dim:
            raise ValueError("latent_dim must be divisible by cross_attn_heads")

        self.q_proj = nn.Linear(config.latent_dim, config.latent_dim, bias=False)
        self.kv_proj = nn.Linear(config.hidden_dim, 2 * config.latent_dim, bias=False)
        self.out_proj = nn.Linear(config.latent_dim, config.latent_dim, bias=False)

        self.norm_q = nn.LayerNorm(config.latent_dim)
        self.norm_kv = nn.LayerNorm(config.hidden_dim)

    def forward(
        self,
        latents: torch.Tensor,
        tokens: torch.Tensor,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Args:
            latents:    (B, L, D_lat)
            tokens:     (B, E, D_hid)
            token_mask: (B, E) bool, True where valid
        Returns:
            (B, L, D_lat)
        """
        q = self.q_proj(self.norm_q(latents))
        kv = self.kv_proj(self.norm_kv(tokens))
        k, v = kv.chunk(2, dim=-1)

        q = rearrange(q, "b l (h d) -> b h l d", h=self.num_heads)
        k = rearrange(k, "b e (h d) -> b h e d", h=self.num_heads)
        v = rearrange(v, "b e (h d) -> b h e d", h=self.num_heads)

        attn_mask = None
        if token_mask is not None:
            # broadcast (B, E) -> (B, 1, 1, E) for SDPA
            attn_mask = token_mask[:, None, None, :]

        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = rearrange(out, "b h l d -> b l (h d)")
        return cast(torch.Tensor, latents + self.out_proj(out))


class SelfAttentionBlock(nn.Module):
    """Self-attention within the latent array."""

    def __init__(self, config: CortexConfig) -> None:
        super().__init__()
        self.config = config
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim

        self.qkv_proj = nn.Linear(config.latent_dim, 3 * config.latent_dim, bias=False)
        self.out_proj = nn.Linear(config.latent_dim, config.latent_dim, bias=False)

        self.norm_attn = nn.LayerNorm(config.latent_dim)
        self.norm_mlp = nn.LayerNorm(config.latent_dim)

        mlp_hidden = int(config.latent_dim * config.mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(config.latent_dim, mlp_hidden, bias=False),
            nn.GELU(),
            nn.Linear(mlp_hidden, config.latent_dim, bias=False),
        )

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """Args:
            latents: (B, L, D)
        Returns:
            (B, L, D)
        """
        x = self.norm_attn(latents)
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = rearrange(q, "b l (h d) -> b h l d", h=self.num_heads)
        k = rearrange(k, "b l (h d) -> b h l d", h=self.num_heads)
        v = rearrange(v, "b l (h d) -> b h l d", h=self.num_heads)

        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn_out = rearrange(attn_out, "b h l d -> b l (h d)")

        latents = latents + self.out_proj(attn_out)
        latents = latents + self.mlp(self.norm_mlp(latents))
        return latents


class PerceiverEncoder(nn.Module):
    """Cross-attention into latents, then N self-attention blocks."""

    def __init__(self, config: CortexConfig) -> None:
        super().__init__()
        self.config = config

        # Learned latent array
        self.latents = nn.Parameter(torch.randn(config.num_latents, config.latent_dim) * 0.02)

        self.cross_attn = CrossAttentionBlock(config)
        self.self_attn_blocks = nn.ModuleList(
            [SelfAttentionBlock(config) for _ in range(config.num_layers)]
        )
        self.final_norm = nn.LayerNorm(config.latent_dim)

    def forward(self, tokens: torch.Tensor, token_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Args:
            tokens:     (B, E, D_hid)
            token_mask: (B, E) bool, True where valid
        Returns:
            latents: (B, L, D_lat)
        """
        batch_size = tokens.shape[0]
        latents = self.latents.unsqueeze(0).expand(batch_size, -1, -1)

        latents = self.cross_attn(latents, tokens, token_mask)
        for block in self.self_attn_blocks:
            latents = block(latents)
        return cast(torch.Tensor, self.final_norm(latents))
