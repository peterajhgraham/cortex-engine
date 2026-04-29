# Phase 2 Kernel Benchmarks

> Populated by `python -m cortex.kernels.bench_tokenizer` and the equivalent
> for sparse_xattn and fused_rmsnorm.

## Hardware

TBD (e.g., A100-80GB)

## Fused Spike Tokenizer

Comparison of `fused_tokenizer` (Triton) vs `fused_tokenizer_reference` (PyTorch eager).

| E (events) | D (hidden) | Reference (ms) | Triton (ms) | Speedup |
|---|---|---|---|---|
| 1024 | 128 | TBD | TBD | TBD |
| 4096 | 256 | TBD | TBD | TBD |
| 16384 | 256 | TBD | TBD | TBD |
| 65536 | 384 | TBD | TBD | TBD |

### Notes

- Optimal `(BLOCK_E, BLOCK_D)` after autotune: TBD
- The kernel becomes bandwidth-bound at TBD events; further improvement requires...

## Sparse Cross-Attention

(populate from `cortex.kernels.bench_sparse_xattn`)

| L | E | Reference (ms) | Triton (ms) | Speedup | Sparsity |
|---|---|---|---|---|---|
| 128 | 1024 | TBD | TBD | TBD | TBD |
| 128 | 4096 | TBD | TBD | TBD | TBD |
| 128 | 16384 | TBD | TBD | TBD | TBD |

### Roofline analysis

(insert plot. Compute bound? Memory bound? At what input size does the regime change?)

## Fused RMSNorm + Linear

| Batch | Hidden | Reference (ms) | Triton (ms) | Speedup |
|---|---|---|---|---|
| TBD | TBD | TBD | TBD | TBD |

## Reproducibility

```bash
make bench-kernels
```

Raw JSON in `benchmarks/kernels/*.json`.
