"""Model configuration. Strongly typed via Pydantic.

Three reference sizes are exposed at the bottom:
- Cortex-XS: ~5M params  (4 layers,  128 dim, 2 heads)
- Cortex-S:  ~25M params (6 layers,  256 dim, 4 heads)
- Cortex-M:  ~80M params (8 layers,  384 dim, 6 heads)
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
    hidden_dim=128,
    num_layers=4,
    num_heads=2,
    head_dim=64,
    num_latents=64,
    latent_dim=128,
    cross_attn_heads=2,
    max_neurons=512,
    max_time_bins=1024,
    behavior_dim=2,
)

CORTEX_S = CortexConfig(
    hidden_dim=256,
    num_layers=6,
    num_heads=4,
    head_dim=64,
    num_latents=128,
    latent_dim=256,
    cross_attn_heads=4,
    max_neurons=512,
    max_time_bins=1024,
    behavior_dim=2,
)

CORTEX_M = CortexConfig(
    hidden_dim=384,
    num_layers=8,
    num_heads=6,
    head_dim=64,
    num_latents=256,
    latent_dim=384,
    cross_attn_heads=6,
    max_neurons=512,
    max_time_bins=1024,
    behavior_dim=2,
)
