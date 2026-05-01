"""Model configuration. Strongly typed via Pydantic.

Three reference sizes are exposed at the bottom:
- Cortex-XS: ~5M params  (5 layers,  256 dim, 4 heads, 128 latents)
- Cortex-S:  ~25M params (7 layers,  512 dim, 8 heads, 256 latents)
- Cortex-M:  ~80M params (11 layers, 768 dim, 12 heads, 384 latents)
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class CortexConfig(BaseModel):
    """Configuration for the Cortex model family.

    All fields are required. No silent defaults that drift between calls.
    """

    # Architecture
    hidden_dim: int = Field(..., gt=0, description="Embedding/hidden dimension")
    num_layers: int = Field(..., gt=0, description="Number of transformer layers")
    num_heads: int = Field(..., gt=0, description="Number of attention heads")
    head_dim: int = Field(..., gt=0, description="Per-head dimension")
    mlp_ratio: float = Field(4.0, gt=0, description="MLP hidden = mlp_ratio * hidden_dim")
    dropout: float = Field(0.0, ge=0, lt=1)

    # Perceiver cross-attention
    num_latents: int = Field(..., gt=0, description="Size of latent array")
    latent_dim: int = Field(..., gt=0, description="Latent array hidden dimension")
    cross_attn_heads: int = Field(..., gt=0)

    # Tokenization
    max_neurons: int = Field(..., gt=0, description="Max neuron index across all sessions")
    max_time_bins: int = Field(..., gt=0, description="Max time bin index per window")
    spike_value_buckets: int = Field(8, gt=0, description="Quantization buckets for spike count")

    # Decoder heads
    behavior_dim: int = Field(..., gt=0, description="Output dim of behavior decoder (e.g., 2 for x/y velocity)")
    use_masked_spike_head: bool = True

    # Numeric
    dtype: str = Field("bfloat16", pattern="^(float32|float16|bfloat16)$")

    # Phase 2: enable Triton kernels (requires CUDA; silently falls back to PyTorch on MPS/CPU)
    use_kernels: bool = False

    @model_validator(mode="after")
    def _check_head_consistency(self) -> CortexConfig:
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(f"hidden_dim ({self.hidden_dim}) must be divisible by num_heads ({self.num_heads})")
        if self.head_dim * self.num_heads != self.hidden_dim:
            raise ValueError(
                f"head_dim ({self.head_dim}) * num_heads ({self.num_heads}) "
                f"must equal hidden_dim ({self.hidden_dim})"
            )
        return self


# ── Reference configs ──────────────────────────────────────────────────────────

CORTEX_XS = CortexConfig(
    hidden_dim=256,
    num_layers=5,
    num_heads=4,
    head_dim=64,
    num_latents=128,
    latent_dim=256,
    cross_attn_heads=4,
    max_neurons=512,
    max_time_bins=1024,
    behavior_dim=2,
)

CORTEX_S = CortexConfig(
    hidden_dim=512,
    num_layers=7,
    num_heads=8,
    head_dim=64,
    num_latents=256,
    latent_dim=512,
    cross_attn_heads=8,
    max_neurons=512,
    max_time_bins=1024,
    behavior_dim=2,
)

CORTEX_M = CortexConfig(
    hidden_dim=768,
    num_layers=11,
    num_heads=12,
    head_dim=64,
    num_latents=384,
    latent_dim=768,
    cross_attn_heads=12,
    max_neurons=512,
    max_time_bins=1024,
    behavior_dim=2,
)
