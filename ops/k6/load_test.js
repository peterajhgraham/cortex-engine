// k6 load test for Cortex-Engine.
//
// Three scenarios:
//   1. constant load at moderate rate (baseline behavior)
//   2. ramping load (find saturation point)
//   3. spike test (resilience under burst)
//
// SLO check: 99% of requests under 30ms.

import http from "k6/http";
import { check } from "k6";
import { Trend } from "k6/metrics";

const BASE = __ENV.CORTEX_BASE || "http://cortex-engine:8080";

const latencyTrend = new Trend("cortex_e2e_latency_ms", true);

export const options = {
  scenarios: {
    constant_load: {
      executor: "constant-arrival-rate",
      rate: 100,
      timeUnit: "1s",
      duration: "60s",
      preAllocatedVUs: 50,
      maxVUs: 200,
      exec: "decode",
    },
    ramping_load: {
      executor: "ramping-arrival-rate",
      startRate: 50,
      timeUnit: "1s",
      stages: [
        { duration: "30s", target: 100 },
        { duration: "60s", target: 500 },
        { duration: "60s", target: 1000 },
        { duration: "30s", target: 100 },
      ],
      preAllocatedVUs: 100,
      maxVUs: 500,
      exec: "decode",
      startTime: "70s",
    },
  },
  thresholds: {
    cortex_e2e_latency_ms: ["p(99)<30"],
    http_req_failed: ["rate<0.001"],
  },
};

function makeEvents(count) {
  const events = [];
  for (let i = 0; i < count; i++) {
    events.push({
      neuron_id: Math.floor(Math.random() * 256),
      time_bin: Math.floor(Math.random() * 120),
      value: Math.floor(Math.random() * 8),
    });
  }
  return events;
}

export function decode() {
  const payload = JSON.stringify({
    request_id: `${__VU}-${__ITER}`,
    session_id: `session-${__VU}`,
    events: makeEvents(50 + Math.floor(Math.random() * 100)),
  });

  const t0 = Date.now();
  const res = http.post(`${BASE}/decode`, payload, {
    headers: { "Content-Type": "application/json" },
  });
  latencyTrend.add(Date.now() - t0);

  check(res, {
    "status 200": (r) => r.status === 200,
    "has behavior": (r) => r.json("behavior") !== undefined,
  });
}
