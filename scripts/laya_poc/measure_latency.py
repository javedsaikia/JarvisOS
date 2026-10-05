#!/usr/bin/env python3
"""Measure Laya latency on this Mac (MPS vs CPU). Uses English checkpoint only."""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

QUESTIONS = {
    "intent": {
        "type": "choice",
        "instructions": "Which Bella tool domain should handle this request?",
        "criteria": {
            "calendar": "schedule, events, agenda, meetings",
            "email": "inbox, gmail, send mail, read email",
            "spotify": "play music, pause, skip track",
            "location": "weather, maps, nearby places, where am I",
            "screen": "what is on screen, read screen, how to click",
            "shell": "run command, terminal",
            "chat": "general conversation, no tool action",
        },
    },
    "needs_tool": {
        "type": "noul",
        "instructions": "Does the user expect Bella to perform a tool action?",
    },
}

STATE = {"body": "Can you add a meeting at 1pm today with the insurance agents?"}


def bench(device: str, n_warm: int = 3, n_run: int = 20) -> dict:
    import laya

    agent = laya.load("convaiinnovations/laya")
    if hasattr(agent, "to"):
        agent = agent.to(device)

    for _ in range(n_warm):
        agent.predict(STATE, QUESTIONS)

    singles: list[float] = []
    for _ in range(n_run):
        t0 = time.perf_counter()
        agent.predict(STATE, QUESTIONS)
        singles.append((time.perf_counter() - t0) * 1000)

    batch_times: list[float] = []
    for _ in range(5):
        t0 = time.perf_counter()
        for _ in range(10):
            agent.predict(STATE, QUESTIONS)
        batch_times.append((time.perf_counter() - t0) * 1000)

    return {
        "device": device,
        "checkpoint": "convaiinnovations/laya (english)",
        "single_ms_p50": statistics.median(singles),
        "single_ms_p95": sorted(singles)[int(len(singles) * 0.95) - 1],
        "single_ms_mean": statistics.mean(singles),
        "batch10_ms_mean": statistics.mean(batch_times),
        "batch10_ms_per_q": statistics.mean(batch_times) / 10,
    }


def main() -> int:
    import laya
    import torch

    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")

    print(f"laya {getattr(laya, '__version__', '?')}  torch {torch.__version__}")
    print(f"devices: {devices}\n")

    results = []
    for dev in devices:
        print(f"Benchmarking device={dev} …", flush=True)
        try:
            r = bench(dev)
            results.append(r)
            print(
                f"  single p50={r['single_ms_p50']:.1f} ms  "
                f"p95={r['single_ms_p95']:.1f} ms  "
                f"batch10={r['batch10_ms_mean']:.1f} ms "
                f"({r['batch10_ms_per_q']:.1f} ms/q)"
            )
        except Exception as e:
            print(f"  FAILED: {e}", file=sys.stderr)

    out = Path(__file__).parent / "latency_results.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"\nWrote {out}")
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
