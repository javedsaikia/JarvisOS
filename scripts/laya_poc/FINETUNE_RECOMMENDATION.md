# Laya fine-tuning recommendation (Bella POC)

## POC results (this Mac, zero-shot, hierarchical 2-stage)

| Metric | Value |
|--------|-------|
| Dataset size | 90 utterances (routes matrix + miss paraphrases + Hindi + conversation log) |
| Regex miss subset | 28 utterances |
| Overall accuracy @ 0.85 threshold | **27.8%** (25/90) |
| Regex-miss accuracy @ 0.85 | **75.0%** (21/28) |
| Abstentions (low confidence) | **82/90** |

## Verdict: fine-tune before enabling in production

Overall accuracy is well below the 70% bar. The miss-path number (75%) looks
usable, but **91% of cases abstained** at the default 0.85 threshold because
the base English checkpoint ships with miscalibrated temperatures (Laya warns
about this at load time).

### Why fine-tuning is required

1. **Zero-shot base models score ~36% on typed-decisions** (Laya upstream docs).
   Bella's hierarchical schema is custom — the base model was not trained on it.
2. **Confidence gating is unusable without temperature fitting.** At 0.85,
   Laya abstains on almost everything; at lower thresholds it routes with
   over-confidence (documented failure mode on non-English scripts).
3. **Regex already handles the easy cases.** Laya only adds value on misses;
   those need domain-specific labels, not generic AG News / triage presets.

### Recommended fine-tuning path

1. **Export training data** from:
   - `checkpoints.py` `ROUTES` matrix (~40 labeled pairs)
   - Router miss paraphrases (see `benchmark_accuracy.py`)
   - `bella/memory/conversation_log.jsonl` (label user turns with backend)
   - Live miss logging once `laya_enabled` is on (log `regex=None` + corrected route)

2. **Target schema:** same 8 coarse + fine categories in `bella/laya_router.py`.

3. **Run upstream notebook:** `notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`
   (~4–5 h on 2×T4). Push result to a private Hugging Face repo or local path.

4. **Fit calibration temperatures** on a held-out Bella validation split (Laya docs:
   ECE drops from ~0.47 → ~0.08 after fitting).

5. **Re-run** `scripts/laya_poc/benchmark_accuracy.py` — target **≥70% overall**
   and **≥85% on regex misses** before setting `laya_enabled: true`.

### Latency on this Mac (English checkpoint, no fine-tune)

| Device | Single question p50 | 10 questions |
|--------|---------------------|--------------|
| CPU | 92.6 ms | 926 ms (92.6 ms/q) |
| MPS | 93.1 ms | 934 ms (93.4 ms/q) |

Two-stage hierarchical routing ≈ **185 ms** per miss — still faster than a Groq/Ollama
chat fallback (~500 ms–2 s), so latency is not the blocker; accuracy is.

### Integration status

`bella/laya_router.py` and `laya_enabled: false` config flag are wired in.
Leave disabled until a fine-tuned checkpoint is available.
