"""Streaming KV cache.

Why this exists
---------------
LLM serving uses paged KV caches to handle variable-length sequences without
fragmentation. We need the same trick for streaming neural decoders, but with a
twist: the "context" is a sliding window of recent spike events rather than a
growing prompt. Pages are evicted by age, not by sequence completion.

Architecture
------------
    - Block-based: cache is a fixed pool of pages, each holding K spikes worth
      of (k, v) tensors for one session
    - Page table: maps (session_id, time_window) -> page_id
    - LRU eviction when the pool is exhausted
    - Per-page metadata: last_used_at, session_id, window_start_ms

Implementation TODO list for Claude Code (Phase 3):
    [ ] PageTable with hash-based lookup
    [ ] PagePool with pre-allocated tensor of shape (num_pages, page_size, num_layers, num_heads, head_dim)
    [ ] LRU eviction policy
    [ ] Allocation API: get_or_create_page(session_id, window_start_ms)
    [ ] Read API: gather pages for a (session_id, window_range) into a contiguous batch tensor
    [ ] Metrics: page_hit_rate, eviction_rate
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PageMeta:
    page_id: int
    session_id: str
    window_start_ms: int
    last_used_at: float


class StreamingKVCache:
    """Paged KV cache for streaming inference. Skeleton; fill in Phase 3."""

    def __init__(self, num_pages: int, page_size: int, num_layers: int, num_heads: int, head_dim: int) -> None:
        self.num_pages = num_pages
        self.page_size = page_size
        # TODO Phase 3.K1: allocate tensor pool
        # TODO Phase 3.K2: page table
        # TODO Phase 3.K3: LRU bookkeeping
