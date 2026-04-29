"""Prometheus metrics for the inference server.

Naming follows the Prometheus convention: lowercase_with_underscores, units in
the metric name (e.g., _seconds, _bytes), counters end in _total.
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram

# Request flow
REQUEST_COUNTER = Counter(
    "cortex_requests_total",
    "Total inference requests received",
    ["endpoint"],
)

REQUEST_LATENCY = Histogram(
    "cortex_request_latency_seconds",
    "End-to-end request latency (queue + inference + serialization)",
    ["endpoint"],
    buckets=(0.001, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5),
)

# Queue and scheduler
QUEUE_DEPTH = Gauge("cortex_queue_depth", "Pending requests in scheduler queue")
QUEUE_WAIT = Histogram(
    "cortex_queue_wait_seconds",
    "Time a request waits in the scheduler queue before inference begins",
    buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1),
)

# Inference
INFERENCE_LATENCY = Histogram(
    "cortex_inference_latency_seconds",
    "Time spent inside the inference worker (no queueing)",
    buckets=(0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.05, 0.1),
)
BATCH_SIZE = Histogram(
    "cortex_batch_size",
    "Size of each batch executed by the inference worker",
    buckets=(1, 2, 4, 8, 16, 32, 64),
)

# Resources
GPU_MEMORY_USED = Gauge("cortex_gpu_memory_used_bytes", "GPU memory currently allocated")
GPU_UTILIZATION = Gauge("cortex_gpu_utilization_percent", "Estimated GPU utilization")

# Cache
KV_CACHE_PAGES_USED = Gauge("cortex_kv_cache_pages_used", "Currently allocated KV cache pages")
KV_CACHE_HIT_RATE = Gauge("cortex_kv_cache_hit_rate", "Rolling KV cache hit rate")

# Errors
ERROR_COUNTER = Counter(
    "cortex_errors_total",
    "Total errors by type",
    ["error_type"],
)
