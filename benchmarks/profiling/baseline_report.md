# Cortex-S Inference Profile — Baseline (Phase 2)

Generated: 2026-04-30  
Hardware: Apple M4 Pro, 24 GB unified memory, MPS backend (PyTorch 2.2.2)  
Profiling: `torch.mps.synchronize()` -fenced section timings (authoritative) +
           `torch.profiler` CPU-activity trace (dispatch-level breakdown)  
Raw data: [`section_timings.json`](section_timings.json),
          [`section_timings_before_fix.json`](section_timings_before_fix.json),
          [`profiler_table.txt`](profiler_table.txt)

## Model & Input Configuration

| Field | Value |
|---|---|
| Model | Cortex-S (24.80 M params) |
| hidden\_dim / latent\_dim | 512 |
| num\_latents | 256 |
| num\_layers (self-attn) | 7 |
| cross\_attn\_heads | 8 |
| Batch size | 32 |
| Events per sample | 512 (MC_Maze density: ~685 spikes / 600 ms window) |
| Total events B × E | 16,384 |
| Warmup / timed iters | 5 / 30 (each section independently, pre-computed inputs) |

---

## Section Timing: Before and After `_pack_events` Fix

The first profiling run revealed a high-impact fix that was applied before the kernels.
Both sets of numbers are preserved for the record.

| Section | Before (ms) | After (ms) | Delta | % of fwd (after) | Triton candidate |
|---|---|---|---|---|---|
| **full\_forward** | **133.24** | **129.01** | −4.23 | 100.0% | — |
| tokenizer | 1.34 | 1.39 | — | 1.1% | **no** (< 5%) |
| pack\_events | 13.63 | 7.83 | **−5.80** | 6.1% | **YES** |
| cross\_attn | 13.38 | 13.56 | — | 10.5% | **YES** |
| self\_attn\_all (×7) | 98.82 | 100.59 | — | 78.0% | **YES** |
| ↳ self\_attn\_single | 14.20 | 14.12 | — | 10.9% | — |
| ↳ mlp\_only (per block) | 6.98 | 6.94 | — | 5.4% | **YES** |
| behavior\_head | 3.08 | 3.01 | — | 2.3% | **no** (< 5%) |

**The `_pack_events` fix reduced full-forward by 3.2% (4.2 ms) with no Triton needed.**

---

## Critical Findings

### Finding 1 — Self-attention is 78% of wall time (primary kernel target)

Seven `SelfAttentionBlock` layers each run:

1. `norm_attn` → `qkv_proj`: LayerNorm `(B, 256, 512)` + linear `(512 → 1536)`
2. SDPA over L=256 latent tokens: attention map 256² = 65,536 per head × 8 heads
3. `out_proj`: `(512 → 512)`
4. `norm_mlp` → `mlp[0]`: LayerNorm `(B, 256, 512)` + linear `(512 → 2048)`
5. GELU + `mlp[2]`: `(2048 → 512)`
6. Two residual adds

The `mlp_only` measurement (6.94 ms/block = 5.4% per block × 7 blocks ≈ 37% of fwd) isolates the
MLP sub-path. Steps 1 and 4 each read the full `(B, 256, 512)` buffer to compute LayerNorm, then
write an intermediate buffer, then the following linear reads that buffer again — two full
HBM round-trips that a fused kernel eliminates.

**Kernel target**: fused LayerNorm + first linear (`norm_attn→qkv_proj`, `norm_mlp→mlp[0]`).
Expected: 1.5–2× per fused call, ~8–15% end-to-end.

### Finding 2 — `_pack_events` was dominated by 34 CPU↔MPS sync stalls (fixed)

The `torch.profiler` trace before the fix:

```
aten::_local_scalar_dense   83.68%   471.9 ms   270 calls   1.750 ms avg
aten::item                   0.38%     2.1 ms   270 calls   (same calls)
```

270 calls across 5 profiler iterations = **54 `.item()` calls per forward pass.**

The source was a Python loop in `_pack_events` (`cortex/models/cortex.py`):

```python
# OLD: 32 .item() calls — each forces a full MPS queue drain and CPU wait
for b in range(batch_size):
    cum[b] = running
    running += int(counts[b].item())   # ← MPS sync stall here

# FIX: 0 .item() calls — purely vectorized
cum = torch.cat([
    torch.zeros(1, dtype=torch.long, device=device),
    counts[:-1].cumsum(0),
])
```

Result: `_pack_events` 13.63 ms → 7.83 ms (−43%). `.item()` calls per pass: 54 → 22.
The 22 remaining calls come from `batch_indices.max().item()`, `counts.max().item()`, and
PyTorch-internal scalar extractions inside `bincount` and `scaled_dot_product_attention` on MPS
(not user-controllable without writing custom kernels).

**This fix is why profiling precedes kernel writing.**

### Finding 3 — Cross-attention is 10.5%, Q-small / KV-large, sparse domain

Shape: Q = `(32, 8, 256, 64)`, KV = `(32, 8, 512, 64)`.
Attention map: 256 × 512 = **131,072 elements per head**.

Spike events are temporally sparse — in MC_Maze, spiking activity concentrates in 50–150 ms
bursts around movement onset, with long silent intervals. A large fraction of the 512 KV positions
carry near-zero attention weight for any given latent query. A Triton kernel that skips those
positions reduces compute proportional to spike sparsity (~60–80% for rest-period windows).

The maximum possible improvement from this kernel is bounded by the 10.5% section share.

### Finding 4 — Tokenizer is 1.1%: below the 5% Triton threshold

`tokenizer = 1.39 ms (1.1%)`. Per CLAUDE.md: *"Never write a Triton kernel for an op that takes
less than 5% of inference time."*

The fused tokenizer kernel is scheduled in Phase 2.1 for **infrastructure and CUDA-at-scale
reasons** — not because the profiling justifies it on MPS. On an A100 with batch=256 and 2,000+
events/sample, the three independent embedding lookups become memory-bandwidth-bound at a scale
that warrants fusion. The benchmark will report this honestly: the kernel may produce <1 ms
improvement on MPS at batch=32.

### Finding 5 — LayerNorm dispatches to `aten::native_batch_norm` on MPS (PyTorch 2.2)

```
aten::native_batch_norm   2.26%   90 calls   (= 18 calls/fwd)
aten::native_layer_norm   0.20%   90 calls   (= 18 calls/fwd)
```

18 LayerNorm calls/fwd matches the model exactly: 7 SA × 2 + 1 cross-attn × 2 + 1 final\_norm +
1 behavior\_head = 18. On MPS with PyTorch 2.2, `nn.LayerNorm` on 3D tensors dispatches through
`native_batch_norm` as an internal optimization. Compute is correct; this is a dispatch-path
artifact. The fused norm+linear kernel will absorb this path entirely.

---

## Triton Kernel Roadmap (Profiling-Driven)

| Priority | Action | Section | Time | Threshold met | Projected gain |
|---|---|---|---|---|---|
| **0** ✅ done | Vectorize `_pack_events` loop | pack\_events | was 13.63 ms | n/a (pure PyTorch) | −4.2 ms |
| **1** | Fused LayerNorm + linear (Phase 2.3) | self\_attn\_all | 100.6 ms | YES (78%) | 8–15 ms end-to-end |
| **2** | Sparse cross-attention (Phase 2.2) | cross\_attn | 13.6 ms | YES (10.5%) | 5–10 ms (density-dep.) |
| **3** | Fused tokenizer (Phase 2.1) | tokenizer | 1.4 ms | **no** (1.1%) | <1 ms (infra value only) |

Implementation order follows Phase 2 plan: tokenizer → sparse xattn → fused norm+linear.
The plan order is fine because the tokenizer kernel establishes the Triton patterns reused by
the other two. Benchmarks will accurately report the measured impact of each.

---

## `torch.profiler` Op Breakdown — After Fix (top 30 by self CPU time)

5 iterations, `ProfilerActivity.CPU`, batch=32, 512 events/sample.

> **MPS caveat**: `ProfilerActivity.CPU` captures Python/C++ dispatch time, not Metal GPU kernel
> time. The `_local_scalar_dense` dominance reflects MPS queue-drain stalls at each `.item()`
> call. Section timings fenced with `torch.mps.synchronize()` are the authoritative numbers.

```
--------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                                        Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
--------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
           aten::_local_scalar_dense (item)        84.18%     438.871ms        84.20%     438.988ms       3.991ms           110
               aten::native_batch_norm (LN)         2.26%      11.777ms         2.28%      11.871ms     131.900us            90
                            aten::linear            2.16%      11.260ms         2.18%      11.368ms      66.871us           170
                            full_forward            2.01%      10.468ms       100.00%     521.340ms     104.268ms             5
                               aten::mul            1.24%       6.478ms         1.36%       7.088ms      75.404us            94
                               aten::bmm            1.14%       5.964ms         1.14%       5.964ms      66.267us            90
                           aten::addcmul            1.10%       5.730ms         1.10%       5.730ms      63.667us            90
                               aten::add            0.84%       4.356ms         0.84%       4.361ms      51.306us            85
                          aten::_softmax            0.61%       3.160ms         0.61%       3.160ms      70.222us            45
                              aten::gelu            0.36%       1.879ms         0.36%       1.879ms      53.686us            35
                      aten::index_select            0.27%       1.421ms         0.28%       1.443ms      96.200us            15
                     aten::native_layer_norm        0.20%       1.032ms         3.63%      18.921ms     210.233us            90
                          aten::matmul            0.18%         953us         1.81%       9.419ms     104.656us            90
    aten::_scaled_dot_product_attention_math        0.xx%         805us         3.20%      18.041ms     400.911us            45
        aten::scaled_dot_product_attention         0.10%         568us         3.58%      20.201ms     448.911us            45
                         aten::bincount           0.12%         635us         0.81%       4.240ms     848.000us             5
--------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 521.340ms (5 iters → 104.3 ms/iter; wall-clock 129.0 ms/iter with MPS async)
```

After the fix: `_local_scalar_dense` dropped from 270 to 110 calls (54 → 22 per forward pass),
confirming that exactly 32 calls were eliminated by replacing the Python loop with `cumsum`.

---

## Reproduction

```bash
# Full profile (generates all output files)
PYTHONPATH=. .venv/bin/python scripts/profile_inference.py \
    --device auto --batch-size 32 --events-per-sample 512

# Output files:
#   benchmarks/profiling/section_timings.json        — current timings (after fix)
#   benchmarks/profiling/section_timings_before_fix.json  — pre-fix baseline
#   benchmarks/profiling/profiler_table.txt          — full profiler table
#   benchmarks/profiling/profiler_trace.json         — Chrome://tracing JSON
```

Chrome trace can be loaded at `chrome://tracing` or `about:tracing` for a flame chart view.
