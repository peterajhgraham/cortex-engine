"""Streaming KV cache for spike event encoder outputs.

What is cached
--------------
The Perceiver cross-attention computes K/V projections from spike event tokens.
In streaming BCI inference, consecutive decode windows overlap heavily:

    window_ms = 600ms, stride_ms = 50ms → 91.6% overlap

Events in the overlapping region appear in both consecutive windows and their
K/V projections are identical.  Caching them eliminates redundant computation.

This implementation caches the **spike token embeddings** (tokenizer output,
shape (E, D)) keyed by (session_id, time_bin_start).  The attention K/V
projections are derived from these embeddings at forward time; in Phase 4 they
can be cached directly once the encoder supports partial recomputation.

Architecture
------------
The cache is divided into fixed-size pages.  Each page holds `page_size` spike
event embeddings for a contiguous time range within one session.  This mirrors
the vLLM paged-attention pattern: fixed page size avoids fragmentation while
supporting variable event counts.

    PagePool : pre-allocated tensor of shape
               (num_pages, page_size, hidden_dim) — all pages in one block
    PageTable: dict (session_id, bin_start) → page_id
    LRU queue: deque of page_ids ordered by last access

When a new page is needed and the pool is full, the least-recently-used page
is evicted regardless of session.

API
---
    cache = StreamingKVCache(num_pages=128, page_size=64, hidden_dim=512)
    page = cache.get_or_create(session_id, bin_start)
    cache.write(page, events_tensor)           # fill from tokenizer output
    events_cached = cache.read(session_id, bin_start)  # None on miss
    cache.evict_session(session_id)            # cleanup on session end

Thread safety
-------------
The cache is NOT thread-safe.  It is accessed only from the single-threaded
InferenceWorker executor.  If multiple workers are added, use per-worker
caches or add locking.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass

import torch

from cortex.serve.metrics import KV_CACHE_HIT_RATE, KV_CACHE_PAGES_USED
from cortex.utils.logging import get_logger

log = get_logger(__name__)


# ── Page metadata ─────────────────────────────────────────────────────────────


@dataclass
class PageMeta:
    """Metadata for one cache page."""

    page_id: int
    session_id: str
    bin_start: int  # first time-bin index this page covers
    n_events: int  # number of valid events written into this page
    last_used_at: float  # monotonic timestamp, for LRU


# ── Cache implementation ──────────────────────────────────────────────────────


class StreamingKVCache:
    """Paged embedding cache for streaming spike event inference.

    Args:
        num_pages:  Total pages in the pool.  Each page can hold up to
                    page_size event embeddings.
        page_size:  Maximum events per page (must be >= max events per time tile).
        hidden_dim: Embedding dimension D (must match SpikeTokenizer output).
        dtype:      Storage dtype for the pool tensor.
        device:     Where the pool lives (should match the model device).
    """

    def __init__(
        self,
        num_pages: int = 128,
        page_size: int = 64,
        hidden_dim: int = 512,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
    ) -> None:
        self.num_pages = num_pages
        self.page_size = page_size
        self.hidden_dim = hidden_dim
        self.device = torch.device(device)

        # Pre-allocated tensor pool: (num_pages, page_size, hidden_dim)
        self._pool = torch.zeros(num_pages, page_size, hidden_dim, dtype=dtype, device=self.device)

        # Page table: (session_id, bin_start) → page_id
        self._table: dict[tuple[str, int], int] = {}

        # Page metadata: page_id → PageMeta
        self._meta: dict[int, PageMeta] = {}

        # Free list: page IDs not currently in use
        self._free: list[int] = list(range(num_pages))

        # LRU order: most-recently-used at the right (OrderedDict as ordered set)
        self._lru: OrderedDict[int, None] = OrderedDict()

        # Rolling hit/miss counters for metrics
        self._hits = 0
        self._misses = 0

        log.info(
            "kv_cache_init",
            num_pages=num_pages,
            page_size=page_size,
            hidden_dim=hidden_dim,
            pool_mb=round(self._pool.numel() * self._pool.element_size() / 1e6, 1),
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def read(self, session_id: str, bin_start: int) -> torch.Tensor | None:
        """Return the cached embedding page for (session_id, bin_start), or None.

        Updates LRU order on hit.
        """
        page_id = self._table.get((session_id, bin_start))
        if page_id is None:
            self._misses += 1
            self._update_hit_rate()
            return None

        # Touch LRU
        self._lru.move_to_end(page_id)
        meta = self._meta[page_id]
        meta.last_used_at = time.monotonic()

        self._hits += 1
        self._update_hit_rate()

        n = meta.n_events
        return self._pool[page_id, :n, :]  # (n, hidden_dim)

    def write(
        self,
        session_id: str,
        bin_start: int,
        embeddings: torch.Tensor,  # (E, hidden_dim) — tokenizer output
    ) -> None:
        """Write embeddings into a new page, evicting LRU if necessary.

        If E > page_size, only the first page_size events are cached (the
        remainder require a separate page for the next time tile).
        Silently overwrites an existing page for (session_id, bin_start).
        """
        E = embeddings.shape[0]
        n = min(E, self.page_size)

        # Re-use existing page if present
        page_id = self._table.get((session_id, bin_start))
        if page_id is None:
            page_id = self._allocate()
            self._table[(session_id, bin_start)] = page_id
            self._meta[page_id] = PageMeta(
                page_id=page_id,
                session_id=session_id,
                bin_start=bin_start,
                n_events=0,
                last_used_at=time.monotonic(),
            )
            self._lru[page_id] = None

        self._pool[page_id, :n, :] = (
            embeddings[:n].detach().to(dtype=self._pool.dtype, device=self.device)
        )
        self._meta[page_id].n_events = n
        self._meta[page_id].last_used_at = time.monotonic()
        self._lru.move_to_end(page_id)
        KV_CACHE_PAGES_USED.set(len(self._meta))

    def evict_session(self, session_id: str) -> int:
        """Free all pages belonging to session_id.  Returns count freed."""
        keys = [k for k in self._table if k[0] == session_id]
        for key in keys:
            page_id = self._table.pop(key)
            self._free_page(page_id)
        KV_CACHE_PAGES_USED.set(len(self._meta))
        if keys:
            log.debug("session_evicted", session_id=session_id, pages=len(keys))
        return len(keys)

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def pages_used(self) -> int:
        return len(self._meta)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _allocate(self) -> int:
        """Return a free page_id, evicting LRU if necessary."""
        if self._free:
            return self._free.pop()

        # Evict least-recently-used page
        lru_id, _ = self._lru.popitem(last=False)
        self._free_page(lru_id)

        if not self._free:
            raise RuntimeError("KV cache pool exhausted — this should never happen after eviction")
        return self._free.pop()

    def _free_page(self, page_id: int) -> None:
        """Return page_id to the free list and remove its table entry."""
        meta = self._meta.pop(page_id, None)
        if meta is not None:
            self._table.pop((meta.session_id, meta.bin_start), None)
        self._lru.pop(page_id, None)
        self._free.append(page_id)

    def _update_hit_rate(self) -> None:
        KV_CACHE_HIT_RATE.set(self.hit_rate)
