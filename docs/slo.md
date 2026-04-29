# Service Level Objectives

This document defines the SLOs for Cortex-Engine and the methodology for
detecting and responding to SLO burn.

## Why SLOs

A real-time BCI inference system has hard latency budgets imposed by physiology:
neural decoders must respond fast enough that the closed-loop interaction feels
natural to the user. We define SLOs that reflect those constraints and operate
the system against them.

## Defined SLOs

| SLO | Target | Measurement window | Error budget |
|---|---|---|---|
| **Latency** | 99% of requests complete in <50ms | Rolling 5 minutes | 1% per window |
| **Availability** | 99.9% of requests succeed (non-5xx) | Rolling 30 days | 30 minutes/month |
| **Throughput capacity** | Sustain 1000 req/s without degradation | Steady state | n/a (capacity, not SLO) |

The 50ms latency SLO is set higher than the 30ms target so we have headroom for
operational drift before customer impact.

## How SLOs Are Measured

All SLOs are measured from Prometheus metrics scraped from the inference server:

- Latency: `histogram_quantile(0.99, rate(cortex_request_latency_seconds_bucket[5m]))`
- Availability: `1 - (rate(cortex_errors_total[5m]) / rate(cortex_requests_total[5m]))`
- Throughput: `rate(cortex_requests_total[5m])`

## SLO Burn

Burn rate measures how fast we are consuming the error budget. Defined as:

    burn_rate = error_rate / SLO_threshold

Where `error_rate` is the rate of SLO violations in the current window and
`SLO_threshold` is `1 - SLO_target` (e.g., 0.001 for a 99.9% SLO).

A burn rate of 1.0 means we are consuming budget exactly at the rate that lasts
the full month. A burn rate of 14.4 means the entire monthly budget would be
spent in 2 hours.

### Multi-window, multi-burn-rate alerting

Following Google SRE Workbook patterns, we alert at two windows:

- **Fast burn** (page): 14.4x rate over 1 hour — would exhaust monthly budget in <2h
- **Slow burn** (ticket): 6x rate over 6 hours — would exhaust monthly budget in 5 days

Both rules live in `ops/docker/alerts.yml`.

## Response Procedure

When an SLO alert fires:

1. Check the Grafana dashboards: traffic spike, latency regression, GPU saturation, queue depth
2. Identify the proximate cause from the dashboard signals
3. Mitigate before debugging: scale out, throttle the noisy client, roll back recent deployment
4. Once mitigated, root cause and write up

## Error Budget Policy

If we burn through 100% of a monthly budget:

- Feature work pauses
- All capacity goes to reliability fixes until budget recovers above 50%

This is the lever that prevents reliability and feature velocity from chronically
trading off in favor of features.
