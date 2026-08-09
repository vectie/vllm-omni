# MiniCPM-o 4.5 on Ascend 910C: official Seed-TTS comparison

Date: 2026-08-09; optimization update: 2026-08-10

This report compares the pinned LunaNexa vLLM-Omni candidate, an optimized
candidate built from it, and the competition's published Seed-TTS performance
baseline. It is a real 910C measurement. The candidate-to-optimized comparison
is a same-host, same-protocol code A/B; the organizer comparison is not a
same-binary A/B because its branch does not run against the newer
vLLM/vLLM-Ascend revisions installed in the supplied image.

## Optimized result

The optimized candidate now beats the published mean TTFT and audio TTFP. Its
whole-audio RTF is 1.51% above the published value, down from a 17.14% gap.

| Metric | Published baseline | Optimized, 3-run mean | Sample SD | Change vs published | Verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| TTFT | 333.27 ms | 315.59 ms | 1.86 ms | -5.31% | faster |
| Audio TTFP | 986.47 ms | 972.93 ms | 8.84 ms | -1.37% | faster |
| Whole-audio RTF | 0.4423 | 0.4490 | 0.0080 | +1.51% | slower |
| Per-chunk audio RTF | not published | 0.4821 | 0.0096 | n/a | measured |
| End-to-end latency | not published | 1884.27 ms | 31.85 ms | n/a | measured |

Lower is better for every metric in the table. All three runs completed 32/32
requests with no non-empty error and 100% streaming continuity. Each run
generated the same 136.96 seconds of audio.

The controlled change from the prior candidate is:

| Metric | Prior candidate | Optimized candidate | Improvement |
| --- | ---: | ---: | ---: |
| TTFT | 316.64 ms | 315.59 ms | 0.33% |
| Audio TTFP | 1034.80 ms | 972.93 ms | 5.98% |
| Whole-audio RTF | 0.5181 | 0.4490 | 13.35% |
| Per-chunk audio RTF | 0.5556 | 0.4821 | 13.23% |
| End-to-end latency | 2186.11 ms | 1884.27 ms | 13.81% |
| Request throughput | 0.4577 req/s | 0.5307 req/s | 15.96% |

Optimized per-run means:

| Run | TTFT | Audio TTFP | Whole-audio RTF | Per-chunk RTF | E2E |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 317.73 ms | 983.06 ms | 0.4581 | 0.4932 | 1920.95 ms |
| 2 | 314.39 ms | 968.94 ms | 0.4449 | 0.4771 | 1868.23 ms |
| 3 | 314.65 ms | 966.78 ms | 0.4439 | 0.4761 | 1863.62 ms |

## Implemented optimization

CosyVoice's timestep embedder rebuilt sinusoidal frequencies on CPU, copied
them to NPU, and reran the same two-layer MLP during every one of the ten CFM
steps for every streamed Code2Wav chunk. `BatchedToken2Wav` now computes those
immutable embeddings once per estimator/device/dtype/CFG-batch/timestep shape
and reuses them. Cache fill preserves the original per-step MLP batch shape and
Euler update order; the optimization does not reduce flow-matching steps,
change codec chunk geometry, or alter sampling parameters.

The complete targeted Code2Wav test file passes on the benchmark host: 35/35.
The generated-text, output-length, total-audio-frame, and total-audio-duration
signature is identical for all three prior and all three optimized runs:

```text
806793917d2bf1907f3523d6f3e16b50f527c88a146f746f059f95eb2913b2b3
```

This signature is structural parity, not a substitute for Seed-TTS WER and
speaker-similarity scoring.

## Rejected NPU graph experiment

NPU CFM graph replay remains disabled. The initial graph path failed because
the upstream timestep embedder copied a CPU tensor during capture. After the
cache removed that transfer, capture reached an unsupported internal-format
Conv2D. Setting `torch.npu.config.allow_internal_format=False` made capture
succeed, but a four-request smoke run regressed mean audio RTF from about 0.45
to 1.43, per-chunk RTF to 1.60, and raised logical-chip-1 HBM from about 45.4
GiB to 51.5 GiB. Growing attention-cache shapes are part of the graph key, so
the observed behavior is consistent with capture churn and cache eviction.

That experimental runtime change was reverted. A viable graph implementation
needs fixed/padded cache buckets and output-buffer reuse before retesting.

## Pre-optimization reference

Prior per-run means:

| Run | TTFT | Audio TTFP | Whole-audio RTF | Per-chunk RTF | E2E |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 322.22 ms | 1040.71 ms | 0.5027 | 0.5436 | 2114.49 ms |
| 2 | 313.31 ms | 1038.18 ms | 0.5181 | 0.5563 | 2186.24 ms |
| 3 | 314.38 ms | 1025.51 ms | 0.5335 | 0.5670 | 2257.60 ms |

The input- and output-length array hashes are identical across the three runs,
so the repetitions used the same deterministic 32-request sample:

```text
input_lens  de364db07f8d81c19507da8ad3cbd73a7ea68dccbbc8826d50c8146e63130cb6
output_lens 57eaf1101a747b229c047f0307b342a274b36c2cca9a2e88184ccb4c22745987
```

## Protocol

- Dataset: Seed-TTS English, 1,088 available rows
- Measured prompts: 32
- Warmups: 3 before each measured run
- Concurrency: 1
- Request rate: unlimited
- Oversampling: disabled
- Endpoint: `/v1/chat/completions`
- Modalities: text and audio
- Thinking: disabled
- TTS chat template: enabled
- Random seed: benchmark default, 0
- Candidate commit: `eae333b46073d250f4ddb8c6bc3a04637e6a2e5e`
- Organizer branch reference: `009b80d686febcf683fdbc2bcdf3ad752884641e`

The client used the local checkpoint path for tokenizer loading because the
host could not reach Hugging Face. Requests retained the served model name
`openbmb/MiniCPM-o-4_5`; this changes no request or server behavior.

The candidate server used:

```text
VLLM_ASCEND_ENABLE_CUSTOM_OPS=0
VLLM_OMNI_MINICPMO45_NPU_SDPA_BACKEND=auto
VLLM_OMNI_MINICPMO45_INITIAL_CODEC_CHUNK_FRAMES=25
VLLM_OMNI_MINICPMO45_CODEC_CHUNK_FRAMES=25
VLLM_OMNI_MINICPMO45_CODEC_LEFT_CONTEXT_FRAMES=3
VLLM_OMNI_MINICPMO45_TOKEN2WAV_N_TIMESTEPS=10
VLLM_OMNI_MINICPMO45_NPU_CFM_GRAPH=0
VLLM_OMNI_NPU_SYNC_BEFORE_DEVICE_EVENT=0
```

The installed CANN environment does not expose the custom-op package expected
by vLLM-Ascend, so custom ops were disabled and `fuse_norm_quant` was disabled
through the Ascend compilation config. These settings define this experiment;
numbers from a custom-op-capable image must be reported separately.

## Hardware topology

`npu-smi` reports one physical Ascend 910 card entry with two logical 64 GiB
chips. The candidate deploy config placed stage 0 on logical chip 0 and stages
1 and 2 on logical chip 1. During startup the main stage used about 48.5 GiB on
chip 0; the other stages used chip 1.

This is one physical-card deployment, but it is not a one-logical-chip result.
The published baseline's exact runtime image and topology were not reproduced,
so the comparison is the competition reference comparison, not a controlled
same-host baseline/candidate speedup claim.

## Why the organizer branch was not used as a measured baseline

The organizer branch was preserved as an immutable source snapshot and started
with only compatibility shims. It reached a healthy API, but the first real TTS
request exposed successive removed vLLM-Ascend APIs:

- `AscendConfig.profiling_chunk_config`
- `NPUGenerationModelRunner.pcp_size` and the old PCP manager API
- `AscendConfig.enable_async_exponential`

Earlier startup attempts also required lazy connector imports to avoid eager
Code2Wav inspection crashes and a `use_cp` to `use_dcp` compatibility update.
Continuing would require porting a substantial part of the NPU runners from the
candidate and would contaminate the organizer baseline with candidate-era code.
The failed attempts and logs remain isolated from valid results under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-official-ab-20260809
```

Valid result checksums:

```text
bf66aac027cac9428e9c5599d2689e76e5597813c38337ce057afe05b392bfdd  seed-tts-run1.json
5710f774758a785042f25f96abeae5de7374ac1d1ca019607ecab97dc9237d57  seed-tts-run2.json
6683709efa6e29348e8ad8536e53e29520dce4cb275f8aa40c5a62dd9907bc4d  seed-tts-run3.json
```

Optimized result checksums:

```text
352d6c0f44142bba80d4fca06948ddf6e4c8127a6043c88ef851a5f12f055877  optimized-run1.json
c54af74c5695dcba93fd010b8425db8271ddea8f939db8056c3d315268511988  optimized-run2.json
ffbda40990f738ce7d41f3e1db9e92a6dc1a1cf540162d112c5aee913f065ef2  optimized-run3.json
```

Optimized raw results and the healthy final server log are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-audio-opt-20260810
```

## Competition status and next experiment

This result still does not establish a competition pass. Seed-TTS WER and speaker
similarity were not evaluated, and the full Daily-Omni and Video-MME quality
gates were not run. The current local Video-MME extraction contains only ten
videos, while the official run requires all 2,700 questions.

The immediate next step is quality validation, not another default-on speed
change: export the fixed Seed-TTS manifest audio, run official WER and speaker
similarity, then run the complete Daily-Omni and Video-MME suites. Further
speed work should focus on static Code2Wav cache buckets/output-buffer reuse or
a custom-op-capable image, and remain off by default until it beats this profile
without exceeding the two-point quality budget.
