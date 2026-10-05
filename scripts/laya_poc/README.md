# Laya × Bella POC

Isolated evaluation of [Laya](https://laya.convaiinnovations.com) as a System-1
router fallback. Does **not** modify `bella/.venv` — uses its own venv.

## Setup

```bash
python3 -m venv scripts/laya_poc/.venv
scripts/laya_poc/.venv/bin/pip install 'laya>=0.3.3'
```

## Run

```bash
# Latency (CPU + MPS on this Mac)
scripts/laya_poc/.venv/bin/python3 scripts/laya_poc/measure_latency.py

# Hierarchical schema token budgets
bella/.venv/bin/python3 scripts/laya_poc/validate_schema.py

# Zero-shot accuracy benchmark
PYTHONPATH=. scripts/laya_poc/.venv/bin/python3 scripts/laya_poc/benchmark_accuracy.py
```

## Results summary

See `FINETUNE_RECOMMENDATION.md` for the full verdict.

Integration lives in `bella/laya_router.py` behind `laya_enabled: false` in config.
