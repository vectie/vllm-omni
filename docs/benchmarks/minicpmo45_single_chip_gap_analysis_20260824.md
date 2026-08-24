# MiniCPM-o 4.5 single-chip latency gap analysis

Date: 2026-08-24

This note separates measured results from source changes that still require an
Ascend 910C run.  The current competition evaluator exposes one logical NPU.
Any earlier dual-chip result is diagnostic evidence only and is not the active
leaderboard path.

## The size of the gap

Lower is better for all three competition metrics.

| Result | RTF | TTFP (ms) | TTFT (ms) |
| --- | ---: | ---: | ---: |
| Official framework baseline | 0.4423 | 986.47 | 333.27 |
| Current submission | 0.3761 | 934.66 | 372.55 |
| Current leader | 0.1546 | 237.11 | 43.11 |
| Submission versus baseline | -14.97% | -5.25% | **+11.79% slower** |
| Reduction needed to match leader | 58.89% | 74.63% | 88.43% |

The result cannot be explained by Code2Wav alone.  TTFT happens before the
first audio packet, TTFP includes Thinker plus the first 25 Talker codec steps
plus the first Code2Wav chunk, and the reported audio RTF includes the complete
request-to-last-audio interval.  A slow prefix therefore raises all three
metrics.

## Reconstructed critical path

```text
request / prompt render
  -> Thinker prefill
  -> Thinker autoregressively regenerates the already-known TTS text
  -> llm2tts transfer and Talker setup
  -> Talker codec decode x25
       -> full 6,562-logit top-p sort each token
       -> sampled-code device/host scalar read each token
       -> one-code NPU/CPU transfer each token
  -> first Code2Wav CFM6 + HiFT chunk
  -> first SSE audio packet

remaining Talker tokens
  -> every 25 codes: Code2Wav CFM6 + HiFT
  -> final audio packet
```

The old submission mostly changed CFM from ten evaluations to six.  That is a
real Stage-2 saving, but Amdahl's law limits the end-to-end result while the
Thinker/Talker prefix and host gaps remain.  This is why isolated fusion
microbenchmarks could be several times faster while the official request was
only about 15% faster in RTF.

There was also a configuration mismatch.  The source policy that selected
`FULL_DECODE_ONLY`, event notification, and the accepted layout path required
two visible devices.  The current evaluator has one, so the official run kept
the original single-device `PIECEWISE`/eager deployment.  vLLM Ascend documents
that Npugraph_ex is active in `FULL`/`FULL_DECODE_ONLY` but disabled in
`PIECEWISE`; consequently the competition path did not receive that compile-
time fusion layer.

## Offline changes in the current worktree

These changes are implemented but have **not** yet been assigned a 910C speedup:

1. **Exact-text teacher forcing for Seed-TTS voice cloning.**  A `Base`
   request with reference audio/transcript, `use_tts_template=true`, and
   `enable_thinking=false` now places the known target between
   `<|tts_bos|>`/`<|tts_eos|>`, renders it as a completed assistant span, and
   limits Stage 0 to one decode token.  Ordinary audio-assistant turns are
   unchanged, and an explicit `minicpmo45_tts_teacher_forcing=false` disables
   the path.  This mirrors the released MiniCPM-o 4.5 behavior: its official
   implementation describes teacher forcing as feeding the known text tokens
   to obtain TTS hidden states instead of autoregressively generating them and
   also sets `max_new_tokens=1`.

2. **Single-chip graph policy.**  The untouched three-stage NPU deployment now
   promotes Thinker and Talker from `PIECEWISE` to `FULL_DECODE_ONLY`, captures
   only batch size one for the concurrency-one benchmark, and enables shared-
   memory event notifications.  Explicit operator configuration still wins.
   The previously rejected all-in Stage-2 graph/layout bundle is not enabled.

3. **No impossible codec EOS synchronization.**  EOS is masked for the first
   50 Talker steps.  The old code nevertheless called `sampled.item()` after
   every step.  The new code reads the scalar only after EOS can affect control
   flow, eliminating all 25 such synchronizations before the first packet and
   the first 50 per request.

4. **One codec D2H transfer per publishable chunk.**  One-token Talker deltas
   remain device-resident and are joined before a single 25-code CPU transfer.
   The old bridge performed a device-to-host transfer for every token.

5. **Bounded codec sampler on NPU.**  The checkpoint uses top-p followed by
   top-25.  The old implementation sorted and sampled all 6,562 logits every
   codec step.  The new path sorts/samples 25 candidates and uses the aggregate
   probability mass of the discarded logits to preserve the original top-p
   cutoff.  One hundred randomized 6,562-dimensional distributions matched
   the old final candidate probabilities.  Set
   `VLLM_OMNI_MINICPMO45_NPU_BOUNDED_CODEC_SAMPLER=0` for immediate rollback.

6. **Explicit, exact-reference Code2Wav prepare lifecycle.**  The orchestrator
   now sends a typed `prepare` control event instead of relying on an empty
   placeholder and a negative chunk number.  A model hook moves the request's
   actual reference audio into that event, so Stage 2 prepares the correct
   prompt features and initial CFM/HiFT state while Stage 0/1 run.  Chunk zero
   reuses that state without re-hashing or re-materializing the repeated
   reference waveform.  Legacy empty placeholders remain harmless no-ops; a
   legacy prepare without the reference can still be replaced safely when
   chunk zero supplies it.  The request uses a bounded terminal tombstone and
   lifecycle generation, so an in-flight late chunk cannot recreate finished
   graph/cache state.  Prompt cache IDs include a content fingerprint rather
   than trusting a logical name and path alone.  Set
   `VLLM_OMNI_MINICPMO45_CODE2WAV_PROMPT_PREWARM=0` to restore the old no-op.

   The prepare path also uses copy-on-write prompt containers instead of
   deep-copying the complete multimodal prompt and immediately discarding the
   copy.  A local 30-second/16-kHz waveform microbenchmark reduced this host
   operation from 81.88 microseconds median to 0.33 microseconds median.  This
   is a host-path microbenchmark, not a 910C end-to-end result.

7. **Single-chip canonical-layout qualification profiles.**  Three new
   profiles separate Stage-2 changes that were previously entangled in one
   rejected BF16 bundle:

   - `minicpmo_4_5_1npu_910c_cfm6_canonical_layout_experimental.yaml` keeps
     FP32 and CFM6 while enabling fixed planar KV slabs, cache-major causal
     Conv state, and cache passthrough;
   - `minicpmo_4_5_1npu_910c_cfm6_canonical_qkv_dma_experimental.yaml` adds
     the new double-buffered fixed-shape QKV layout kernel from the paired
     vLLM-Ascend fork;
   - `minicpmo_4_5_1npu_910c_cfm6_canonical_cfm_graph_experimental.yaml`
     captures only the steady width-50/cache-402 CFM call with one graph key
     and two persistent output slots.
   - `minicpmo_4_5_1npu_910c_cfm6_canonical_w8a8_mlp_experimental.yaml`
     keeps the same FP32 quality boundary around attention, normalization,
     Conv, CFG/Euler and HiFT, but changes the 32 DiT MLP weight matrices to
     persistent per-output-channel INT8 and quantizes MLP activations per
     token inside the GE graph.

   These are qualification profiles, not promoted defaults.  The first 910C
   session must compare each one against the cumulative prefix candidate and
   reject any profile that loses complete-request mean or P99 despite winning
   an isolated kernel replay.

Python compilation and whitespace checks pass locally.  The Mac development
environment has no `vllm` installation, so the repository pytest suite cannot
import the runtime here; the target container must run the added CPU tests
before NPU qualification.

## Why these are higher leverage than another isolated fusion

The leader's 43.11 ms TTFT is incompatible with autoregressively generating a
normal Chinese sentence one token at a time; the released teacher-forcing path
is the strongest architectural explanation.  The 237.11 ms TTFP then leaves a
small budget for Talker startup and the first waveform chunk, making per-token
host barriers and a full-vocabulary sort material.  These changes attack the
serial prefix rather than only the CFM tail.

The kernel lesson remains important, but the integration boundary matters.
Ascend's graph path uses Npugraph_ex before ACLGraph capture and obtains gains
by replacing/fusing operators inside the FX graph.  A standalone opaque ACLNN
custom op can benchmark well yet block GE from fusing its producer and
consumer, which matches the earlier observed regression.  Further DiT fusion
should therefore be graph-visible through a converter/decomposition and tested
as a complete serving path.

## First machine session: required A/B sequence

Use the same 32 Chinese Seed-TTS rows, concurrency one, seed zero, two service
warmups, then one complete warm run before measurement.  Keep audio hashes,
token counts, WER, SIM, and continuity data for every row.

| Order | Candidate | Purpose | Promotion gate |
| ---: | --- | --- | --- |
| 0 | Previous submitted commit | Reproduce 0.3761 class control | stable 32/32 |
| 1 | + single-chip FULL_DECODE_ONLY/events | Verify real graph path | TTFT/TTFP/P99 win |
| 2 | + teacher forcing | Test main TTFT hypothesis | transcript/audio correctness; large TTFT win |
| 3 | + deferred EOS scalar read | Isolate host-gap removal | no token/audio regression |
| 4 | + chunk-coalesced D2H | Isolate bridge removal | TTFP and P99 win |
| 5 | + bounded codec sampler | Isolate sort reduction | WER/SIM within gate and latency win |
| 6 | + exact-reference Code2Wav prepare | Move one correct setup off the first-packet boundary | TTFP win without Talker contention regression |
| 7 | cumulative candidate | Submission measurement | all accuracy gates, 32/32 stability |

Do not infer a change's value from cumulative results alone.  Retain it only
after its own matched A/B; stochastic codec sampling means the bounded sampler
must be evaluated on the complete Seed-TTS quality set even though its
probability distribution is mathematically preserved.

## Profiler questions to answer

Collect a service-level `torch_npu.profiler`/msprof trace for rows 0, 2, and 6.
The trace must include host and device timelines and `op_statistic.csv`.

- Is the Thinker teacher-forced target one prefill graph plus one decode replay?
- Does `FULL_DECODE_ONLY` actually show Npugraph_ex/ACLGraph, or was it silently
  narrowed because of backend compatibility?
- Are the first 50 `aclrtSynchronize*` gaps gone?
- Did `Sort`/`TopK`/`Multinomial` Talker time shrink, and did a new TransData
  boundary replace it?
- Is codec D2H one transfer per 25-code packet rather than one per token?
- Does Stage-2 prompt setup overlap Stage 0/1, and does that overlap help or
  merely contend for the same single-chip Cube/Vector resources?
- What percentages belong to Thinker, Talker, CFM, HiFT, TransData/Transpose,
  CPU dispatch, and SSE serialization after the changes?
- If static kernels are enabled later, does `op_statistic.csv` actually contain
  `static_kernel` operators, as the vLLM Ascend guide requires for verification?

Only after this trace should the next lower-layer work begin: graph-visible
Talker sampling fusion, canonical DiT layout propagation, then a static
width-50 CFM executable.  The present evidence does not justify another opaque
megakernel first.

## Primary references

- MiniCPM-o 4.5 official model source:
  https://huggingface.co/openbmb/MiniCPM-o-4_5/blob/main/modeling_minicpmo.py
- vLLM Ascend graph mode and Npugraph_ex:
  https://docs.vllm.ai/projects/ascend/en/main/user_guide/feature_guide/graph_mode.html
- vLLM Ascend static-kernel configuration and verification:
  https://docs.vllm.ai/projects/ascend/en/v0.23.0/user_guide/feature_guide/graph_mode.html
- Thinking Machines Lab, batch invariance and fixed reduction/layout reasoning:
  https://thinkingmachines.ai/blog/defeating-nondeterminism-in-llm-inference/
