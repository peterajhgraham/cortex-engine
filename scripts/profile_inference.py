"""Cortex-S inference profiler — MPS / CPU.

Measures wall-clock time for each model section using explicit sync barriers,
then collects a torch.profiler trace for op-level breakdown.

Sections timed:
    full_forward      — complete model.forward()
    tokenizer         — 3-way embedding lookup
    pack_events       — sort + scatter into dense (B, E, D) tensor
    cross_attn        — Perceiver cross-attention (tokens → latents)
    self_attn_all     — all N self-attention + MLP blocks combined
    self_attn_single  — one self-attention + MLP block
    behavior_head     — output cross-attention to behavior scalars

Usage:
    python scripts/profile_inference.py [--device auto] [--batch-size 32]
                                        [--events-per-sample 512]

Outputs:
    benchmarks/profiling/section_timings.json
    benchmarks/profiling/profiler_table.txt
    benchmarks/profiling/baseline_report.md  (summary)
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from cortex.models.config import CORTEX_S
from cortex.models.cortex import CortexModel, _pack_events
from cortex.utils.device import select_device
from cortex.utils.logging import configure_logging, get_logger

configure_logging(level="WARNING", json=False)
log = get_logger(__name__)

OUT_DIR = Path("benchmarks/profiling")


# ── Sync helpers ──────────────────────────────────────────────────────────────


def sync(device: torch.device) -> None:
    """Block until all pending work on device is complete."""
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()
    # CPU is always synchronous


def timed_loop(fn: Any, args: tuple, device: torch.device, n_warmup: int = 5, n_timed: int = 30) -> float:
    """Return mean wall-clock time in ms over n_timed calls after n_warmup warmup calls."""
    for _ in range(n_warmup):
        with torch.no_grad():
            fn(*args)
    sync(device)

    t0 = time.perf_counter()
    for _ in range(n_timed):
        with torch.no_grad():
            fn(*args)
    sync(device)

    return (time.perf_counter() - t0) / n_timed * 1000.0


# ── Synthetic input generation ────────────────────────────────────────────────


def make_inputs(
    batch_size: int,
    events_per_sample: int,
    device: torch.device,
    cfg: Any = CORTEX_S,
) -> dict[str, torch.Tensor]:
    """Synthetic spike events matching production shapes.

    MC_Maze: 137 heldin neurons, 600ms window (120 bins at 5ms).
    Typical density: ~5 spikes/neuron/window → ~685 events/sample.
    """
    rng = torch.Generator()
    rng.manual_seed(42)

    n_events = batch_size * events_per_sample
    neuron_ids = torch.randint(0, min(137, cfg.max_neurons), (n_events,), generator=rng)
    time_bins  = torch.randint(0, 120, (n_events,), generator=rng)
    values     = torch.randint(0, cfg.spike_value_buckets, (n_events,), generator=rng)

    # Assign events to batch elements (uniform distribution)
    batch_indices = torch.arange(n_events) % batch_size

    return {
        "neuron_ids":    neuron_ids.to(device),
        "time_bins":     time_bins.to(device),
        "values":        values.to(device),
        "batch_indices": batch_indices.to(device),
    }


# ── Section-level timing ──────────────────────────────────────────────────────


def run_section_timings(
    model: CortexModel,
    inp: dict[str, torch.Tensor],
    device: torch.device,
    n_timed: int = 30,
) -> dict[str, float]:
    """Time each model section independently. Returns ms per call."""
    cfg = model.config
    tokenizer = model.tokenizer
    encoder   = model.encoder
    beh_head  = model.behavior_head

    # Pre-compute intermediates once (outside timers) so each section
    # receives correct inputs without including prior-section cost.
    with torch.no_grad():
        flat_tokens = tokenizer(inp["neuron_ids"], inp["time_bins"], inp["values"])
        tokens, mask = _pack_events(flat_tokens, inp["batch_indices"])
        batch_size = tokens.shape[0]
        latents_init = encoder.latents.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        latents_after_cross = encoder.cross_attn(latents_init, tokens, mask)

    timings: dict[str, float] = {}

    # ── 1. Full forward ───────────────────────────────────────────────────────
    timings["full_forward"] = timed_loop(
        model,
        (inp["neuron_ids"], inp["time_bins"], inp["values"], inp["batch_indices"]),
        device,
        n_timed=n_timed,
    )

    # ── 2. Tokenizer ─────────────────────────────────────────────────────────
    timings["tokenizer"] = timed_loop(
        tokenizer,
        (inp["neuron_ids"], inp["time_bins"], inp["values"]),
        device,
        n_timed=n_timed,
    )

    # ── 3. _pack_events ───────────────────────────────────────────────────────
    timings["pack_events"] = timed_loop(
        _pack_events,
        (flat_tokens, inp["batch_indices"]),
        device,
        n_timed=n_timed,
    )

    # ── 4. Cross-attention ────────────────────────────────────────────────────
    timings["cross_attn"] = timed_loop(
        encoder.cross_attn,
        (latents_init, tokens, mask),
        device,
        n_timed=n_timed,
    )

    # ── 5. All self-attention blocks ──────────────────────────────────────────
    def run_all_self_attn(latents: torch.Tensor) -> torch.Tensor:
        for block in encoder.self_attn_blocks:
            latents = block(latents)
        return latents

    timings["self_attn_all"] = timed_loop(
        run_all_self_attn,
        (latents_after_cross,),
        device,
        n_timed=n_timed,
    )

    # ── 6. Single self-attention block ─────────────────────────────────────────
    timings["self_attn_single"] = timed_loop(
        encoder.self_attn_blocks[0],
        (latents_after_cross,),
        device,
        n_timed=n_timed,
    )

    # ── 7. MLP only (within a self-attn block) ────────────────────────────────
    block0 = encoder.self_attn_blocks[0]
    latents_post_attn = latents_after_cross  # approximate (avoids timing self-attn)

    def run_mlp_only(x: torch.Tensor) -> torch.Tensor:
        return x + block0.mlp(block0.norm_mlp(x))

    timings["mlp_only"] = timed_loop(
        run_mlp_only,
        (latents_post_attn,),
        device,
        n_timed=n_timed,
    )

    # ── 8. Behavior head ──────────────────────────────────────────────────────
    latents_final = encoder.final_norm(latents_after_cross)
    timings["behavior_head"] = timed_loop(
        beh_head,
        (latents_final,),
        device,
        n_timed=n_timed,
    )

    return timings


# ── torch.profiler trace ──────────────────────────────────────────────────────


def run_profiler_trace(
    model: CortexModel,
    inp: dict[str, torch.Tensor],
    device: torch.device,
    out_dir: Path,
) -> str:
    """Capture a torch.profiler trace. Returns formatted key-averages table."""
    activities = [ProfilerActivity.CPU]

    # Warmup outside the profiler context
    with torch.no_grad():
        for _ in range(3):
            model(inp["neuron_ids"], inp["time_bins"], inp["values"], inp["batch_indices"])
    sync(device)

    with profile(activities=activities, record_shapes=True, with_stack=False) as prof:
        with torch.no_grad():
            for _ in range(5):
                with record_function("full_forward"):
                    model(inp["neuron_ids"], inp["time_bins"], inp["values"], inp["batch_indices"])
        sync(device)

    # Export JSON trace
    trace_path = out_dir / "profiler_trace.json"
    prof.export_chrome_trace(str(trace_path))

    # Key-averages table (top 30 by self CPU time)
    table = prof.key_averages().table(
        sort_by="self_cpu_time_total",
        row_limit=30,
        max_name_column_width=60,
    )

    table_path = out_dir / "profiler_table.txt"
    table_path.write_text(table)
    print(table)
    return table


# ── Report writer ─────────────────────────────────────────────────────────────


def write_report(
    timings: dict[str, float],
    profiler_table: str,
    device: torch.device,
    batch_size: int,
    events_per_sample: int,
    cfg: Any,
    out_path: Path,
) -> None:
    """Write the baseline profiling report in Markdown."""

    full = timings["full_forward"]
    n_layers = cfg.num_layers

    def pct(ms: float) -> str:
        return f"{ms / full * 100:5.1f}%"

    def row(name: str, ms: float, note: str = "") -> str:
        p = pct(ms)
        flag = " ← **candidate**" if ms / full > 0.05 else ""
        return f"| {name:<26} | {ms:7.2f} ms | {p} |{flag} {note} |"

    # Derived numbers
    self_attn_per_layer = timings["self_attn_all"] / n_layers
    # overhead = full - (tokenizer + pack_events + cross_attn + self_attn_all + behavior_head)
    accounted = (
        timings["tokenizer"]
        + timings["pack_events"]
        + timings["cross_attn"]
        + timings["self_attn_all"]
        + timings["behavior_head"]
    )
    other_ms = max(full - accounted, 0.0)

    device_str = str(device)
    if device.type == "mps":
        device_str = f"Apple M-series MPS (synchronize-fenced)"

    total_events = batch_size * events_per_sample
    latent_size = cfg.num_latents
    hidden_dim  = cfg.hidden_dim
    latent_dim  = cfg.latent_dim

    # Bottleneck analysis
    candidates = {k: v for k, v in timings.items() if v / full > 0.05}
    candidate_lines = "\n".join(
        f"- **{k}** ({v:.2f} ms, {v/full*100:.1f}%)"
        for k, v in sorted(candidates.items(), key=lambda x: -x[1])
        if k != "full_forward"
    )

    # Triton kernel justification
    kernel_plan = []
    if timings.get("self_attn_all", 0) / full > 0.05:
        kernel_plan.append(
            "**Fused RMSNorm + linear projection** (Phase 2.3): "
            f"Each self-attention block calls LayerNorm twice and has 4 linear projections. "
            f"Fusing norm+first-linear reduces memory bandwidth by ~2× at this shape. "
            f"Applies to {n_layers} self-attention blocks + cross-attention."
        )
    if timings.get("cross_attn", 0) / full > 0.05:
        kernel_plan.append(
            "**Sparse cross-attention** (Phase 2.2): "
            f"Cross-attention has Q size ({latent_size}, {latent_dim}) and KV size "
            f"({total_events // batch_size}, {hidden_dim}). "
            f"Spike events are temporally sparse; a Triton kernel exploiting this "
            f"sparsity reduces compute and memory vs dense SDPA."
        )
    if timings.get("tokenizer", 0) / full > 0.05:
        kernel_plan.append(
            "**Fused spike tokenizer** (Phase 2.1): "
            f"Three separate embedding lookups + two additions. "
            f"A single fused kernel eliminates two intermediate write-backs to HBM."
        )
    if not kernel_plan:
        kernel_plan.append(
            "_No single section exceeds 5% of inference time in isolation — "
            "the bottleneck is distributed. See op-level breakdown for finer targets._"
        )

    report = f"""\
# Cortex-S Inference Profile — Baseline (Phase 2)

Generated: {_now()}
Hardware: {_hardware_desc(device)}
Profiling method: Manual section timing with `{_sync_fn(device)}` barriers + `torch.profiler` CPU trace

## Configuration

| Field | Value |
|---|---|
| Model | Cortex-S (24.80 M params) |
| hidden_dim | {hidden_dim} |
| latent_dim | {latent_dim} |
| num_latents | {latent_size} |
| num_layers (self-attn) | {n_layers} |
| cross_attn_heads | {cfg.cross_attn_heads} |
| Batch size | {batch_size} |
| Events per sample | {events_per_sample} (typical MC_Maze density) |
| Total events (B × E) | {total_events:,} |

## Section Timing Table

> All times are mean over 30 synchronization-fenced iterations. Sections are timed
> independently with pre-computed inputs to avoid cumulative overhead.
> **Candidate** = section accounts for >5% of full-forward wall-clock (Triton threshold).

| Section                    |     Time    |   % of fwd | Notes |
|----------------------------|-------------|------------|-------|
{row("full_forward", timings["full_forward"], "end-to-end model.forward()")}
{row("tokenizer", timings["tokenizer"], "3× embedding lookup + 2× add")}
{row("pack_events", timings["pack_events"], "argsort + scatter → (B,E,D)")}
{row("cross_attn", timings["cross_attn"], "tokens(E,D)→latents(L,D) xattn")}
{row("self_attn_all", timings["self_attn_all"], f"all {n_layers} SA blocks + MLP")}
{row(f"  self_attn_single (×{n_layers})", self_attn_per_layer, "1 SA block avg")}
{row("  mlp_only (per block)", timings["mlp_only"], "norm + 2×linear + GELU")}
{row("behavior_head", timings["behavior_head"], "output cross-attention")}
{row("other / framework", other_ms, "layer norms, expand, final_norm")}

## Bottleneck Summary

Sections exceeding the 5% Triton threshold:

{candidate_lines if candidate_lines else "_No section individually exceeds 5%. See op table._"}

### Key observations

1. **Self-attention dominates** (if `self_attn_all` > 50%): {n_layers} stacked SA blocks each contain
   a QKV projection ({latent_dim}→3×{latent_dim}), SDPA over {latent_size} tokens, output proj,
   and a 4× MLP. This is expected for a transformer at this depth/width.

2. **`_pack_events` is pure Python-on-CPU**: The scatter loop iterates over `batch_size={batch_size}`
   elements in Python. On MPS this serializes CPU↔GPU synchronization. At batch=32 with small
   event counts this may be measurable; at continuous-batching scales (batch≥128) it becomes
   a hard bottleneck regardless of GPU speed.

3. **Cross-attention is Q-small, KV-large**: Q is ({latent_size}, {latent_dim}), KV is
   ({events_per_sample}, {hidden_dim}) per sample. The attention map is
   {latent_size}×{events_per_sample} = {latent_size * events_per_sample:,} elements per head.
   With temporal sparsity (spikes cluster in short bursts), a sparse kernel can skip
   zero-attention-weight slots.

4. **Tokenizer is memory-bandwidth-bound**: Three independent embedding lookups each read
   {hidden_dim}×4 bytes per event, then discard the intermediate tensors. A fused kernel
   eliminates two HBM round-trips.

## Triton Kernel Opportunity Analysis

The three Phase 2 kernels map directly to the bottleneck profile:

{chr(10).join(f"{i+1}. {line}" for i, line in enumerate(kernel_plan))}

### Projected impact (pre-implementation estimates)

| Kernel | Target section | Expected speedup | Memory reduction |
|---|---|---|---|
| Fused tokenizer | `tokenizer` | 2–3× | 2× (elim. 2 intermediate tensors) |
| Sparse cross-attn | `cross_attn` | 2–4× (at high sparsity) | proportional to sparsity |
| Fused RMSNorm+linear | `self_attn_all` MLP path | 1.5–2× per block | ~30% fewer HBM reads |

These are pre-benchmark estimates. Actual speedup will be measured in Phase 2 per the
"every claim has a benchmark" principle. If a kernel does not achieve >10% end-to-end
improvement, it will be documented as such.

## `torch.profiler` Op-Level Breakdown

Top 30 ops by self CPU time, 5 iterations with `record_function` annotation:

```
{profiler_table.strip()}
```

> **MPS caveat**: `torch.profiler` captures CPU dispatch time, not Metal GPU kernel time.
> The self-CPU-time numbers reflect Python/C++ dispatch overhead. Section timings above
> (fenced with `torch.mps.synchronize()`) are the authoritative latency numbers.

## Reproduction

```bash
python scripts/profile_inference.py --device auto --batch-size {batch_size} --events-per-sample {events_per_sample}
# Outputs: benchmarks/profiling/section_timings.json
#          benchmarks/profiling/profiler_table.txt
#          benchmarks/profiling/profiler_trace.json  (Chrome trace)
#          benchmarks/profiling/baseline_report.md   (this file)
```
"""

    out_path.write_text(report)
    print(f"\nReport written to {out_path}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _now() -> str:
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M")

def _hardware_desc(device: torch.device) -> str:
    if device.type == "mps":
        return f"Apple Silicon MPS — {platform.node()} ({platform.processor() or 'arm'})"
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    return f"CPU — {platform.processor()}"

def _sync_fn(device: torch.device) -> str:
    if device.type == "mps":
        return "torch.mps.synchronize()"
    if device.type == "cuda":
        return "torch.cuda.synchronize()"
    return "(CPU, no sync needed)"


# ── Main ──────────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--events-per-sample", type=int, default=512,
                        help="Spike events per batch element. Typical MC_Maze density ~512.")
    parser.add_argument("--n-iters", type=int, default=30,
                        help="Timed iterations per section (after 5 warmup).")
    args = parser.parse_args()

    device = select_device(preference=args.device)
    print(f"Device: {device}  ({_hardware_desc(device)})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    model = CortexModel(CORTEX_S).to(device).eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"Cortex-S  {n_params:.2f}M params")

    inputs = make_inputs(args.batch_size, args.events_per_sample, device, CORTEX_S)
    total_events = args.batch_size * args.events_per_sample
    print(f"Input: batch={args.batch_size}, events/sample={args.events_per_sample}, "
          f"total events={total_events:,}")
    print()

    # ── Section timings ───────────────────────────────────────────────────────
    print("Timing sections (warmup=5, timed=30) ...")
    timings = run_section_timings(model, inputs, device, n_timed=args.n_iters)

    print("\n── Section Timings ──────────────────────────────────")
    full = timings["full_forward"]
    for k, v in timings.items():
        bar = "█" * int(v / full * 40)
        print(f"  {k:<26}  {v:7.2f} ms  {v/full*100:5.1f}%  {bar}")

    timings_path = OUT_DIR / "section_timings.json"
    timings_path.write_text(json.dumps({
        "device": str(device),
        "batch_size": args.batch_size,
        "events_per_sample": args.events_per_sample,
        "n_timed_iters": args.n_iters,
        "timings_ms": timings,
    }, indent=2))
    print(f"\nTimings saved: {timings_path}")

    # ── torch.profiler trace ──────────────────────────────────────────────────
    print("\nRunning torch.profiler trace (5 iterations) ...")
    profiler_table = run_profiler_trace(model, inputs, device, OUT_DIR)

    # ── Write report ──────────────────────────────────────────────────────────
    write_report(
        timings=timings,
        profiler_table=profiler_table,
        device=device,
        batch_size=args.batch_size,
        events_per_sample=args.events_per_sample,
        cfg=CORTEX_S,
        out_path=OUT_DIR / "baseline_report.md",
    )


if __name__ == "__main__":
    main()
