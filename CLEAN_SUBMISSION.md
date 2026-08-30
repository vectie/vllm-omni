# MiniCPM-o 4.5 Ascend clean submission

This submission is based directly on the official `minicpm-challenge` commit
`ecd9d99da0c124331861890e0371e66a01cddaa5`.  Its Git history contains no
earlier candidate commits.  The official benchmark implementation, deploy
configuration directory, and `tests/dfx/perf/tests/test_minicpmo_4_5.json`
are byte-for-byte unchanged from that commit.

## Retained optimizations

- batch-one FULL_DECODE graphs for Thinker and Talker on a one-chip NPU;
- fixed codec-history/ring state and graph-owned repetition-state updates;
- fused Talker codec-distribution construction while retaining native
  `torch.multinomial` sampling;
- single-live-slot KV mapping, dirty-row block-table commits, resident decode
  metadata, and scalar staging;
- batched codec publication, IPC coalescing, event-driven shared-memory wakeup,
  and deferred EOS reconciliation;
- immutable Code2Wav prompt-state templates keyed by full reference-content
  fingerprints;
- exact inference-time HiFT weight-normalization materialization and the
  supported Ascend SDPA path;
- vLLM 0.25/0.26-compatible runner symbol resolution, so the official
  `v0.25.0-a3` image and newer development images use their respective symbol
  locations without source edits;
- NPU-specific Code2Wav/DiT layout, cache, graph, and allocation fast paths,
  each guarded by model/platform capability checks.

These paths are selected from model architecture, platform and device count.
They do not inspect request IDs, prompt text, reference transcripts, benchmark
names, dataset rows, or expected answers.

## Deliberately disabled defaults

The clean submission does not default to reduced-step CFM, enlarged first
packets, terminal silence padding, inverse-CDF sampling, teacher forcing,
speculative two-code scheduling, or any benchmark-conditioned behavior.  It
uses the official CFM6 solver, 25-frame packetization, native sampling, and
unmodified terminal audio unless an operator explicitly opts into a research
switch outside evaluation.

## Integrity checks

`tests/config/test_minicpmo45_submission_integrity.py` asserts that runtime
code has no evaluator-conditioned teacher-forcing markers, imports no
benchmark data modules, and leaves output-changing score experiments disabled
by default.

Before packaging, the following checks are required:

```bash
git diff --check
python -m py_compile <all changed Python files>
pytest -q tests/worker/test_vllm_runner_compat.py
python -c 'import runpy; d=runpy.run_path("tests/config/test_minicpmo45_submission_integrity.py"); [d[n]() for n in d if n.startswith("test_")]'
git diff --exit-code upstream/minicpm-challenge -- \
  vllm_omni/benchmarks vllm_omni/deploy \
  tests/dfx/perf/tests/test_minicpmo_4_5.json
```

The organizer should install this source with `pip install -e .
--no-build-isolation` and launch it with the official
`vllm_omni/deploy/minicpmo_4_5.yaml`, exactly as specified by the competition.
