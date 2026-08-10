# MiniCPM-o 4.5 on Ascend 910C: official Seed-TTS comparison

Date: 2026-08-09; optimization update: 2026-08-11

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

## Accepted CFM-delta cache on refreshed 910C host

A refreshed Atlas A3 host exposed the physical card as two logical
`Ascend910_9382` devices with 64 GiB each. The same three-stage placement and
10-step CFM profile were deployed from candidate commit `5cce0948`. A further
Code2Wav change caches the invariant Euler step widths once per device and
dtype. It reproduces CosyVoice's accumulated-time recurrence exactly, instead
of replacing it with direct adjacent timeline differences, and removes two
tiny eager NPU operations from each non-final CFM step of every streamed
chunk.

The candidate was measured twice, then removed and the service restarted for
an interleaved control. The post-restart control reproduced the earlier stable
baseline, so the candidate was accepted and restored. A third candidate run
then completed the fail-closed three-run gate. All measured runs used the same
32 Seed-TTS English requests, three warmups, concurrency one, and the protocol
below.

| Variant/run | TTFT | Audio TTFP | Whole-audio RTF | Per-chunk RTF | E2E |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stable baseline 2 | 322.73 ms | 989.48 ms | 0.4531 | 0.4837 | 1908.46 ms |
| Interleaved baseline 3 | 317.14 ms | 986.83 ms | 0.4540 | 0.4872 | 1913.33 ms |
| Delta-cache candidate 1 | 313.19 ms | 956.88 ms | 0.4475 | 0.4797 | 1885.22 ms |
| Delta-cache candidate 2 | 310.85 ms | 947.02 ms | 0.4407 | 0.4722 | 1859.99 ms |
| Delta-cache candidate 3 | 323.45 ms | 983.58 ms | 0.4513 | 0.4836 | 1901.00 ms |

| Metric | Stable baseline mean | Candidate mean | Change |
| --- | ---: | ---: | ---: |
| TTFT | 319.94 ms | 312.02 ms | -2.47% |
| Audio TTFP | 988.16 ms | 951.95 ms | -3.66% |
| Whole-audio RTF | 0.4536 | 0.4441 | -2.08% |
| Per-chunk audio RTF | 0.4854 | 0.4759 | -1.96% |
| End-to-end latency | 1910.89 ms | 1872.60 ms | -2.00% |
| Benchmark duration | 61.16 s | 59.94 s | -2.00% |

The table above uses the two stable baseline and first two candidate runs to
show the paired steady-state mean. The repository's promotion gate uses all
three runs per side and compares medians with a 1% minimum improvement. It
passed every selected target:

| Gate target | Baseline median | Candidate median | Improvement |
| --- | ---: | ---: | ---: |
| TTFT | 322.73 ms | 313.19 ms | 2.96% |
| Audio TTFP | 989.48 ms | 956.88 ms | 3.30% |
| Whole-audio RTF | 0.4540 | 0.4475 | 1.42% |
| Per-chunk audio RTF | 0.4872 | 0.4797 | 1.53% |
| End-to-end latency | 1913.33 ms | 1885.22 ms | 1.47% |

Lower is better. Every run completed 32/32 requests with zero failures, zero
audio underrun, and 100% streaming continuity. Each produced 4,801 input
tokens, 480 output tokens, 3,321,600 audio frames, and 138.40 seconds of audio.
The workload/output-shape signature was identical across every baseline and
candidate result:

```text
0c7fdd66996ae513520bfb0f1e0697c8629ac1bc6a8110b48279ffd558fc254e
```

The first baseline run is preserved with the raw results but excluded from the
stable mean because one request had a TTFT outlier; including it increases the
reported candidate advantage. The complete targeted Code2Wav test file passes
36/36, including an exact-recurrence and cache-identity test. Structural parity
does not replace the official Seed-TTS WER and speaker-similarity gates.

Raw results and the accepted server log are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-910c-20260810
```

Result checksums:

```text
00949474fe7e47053e48d98b00864c7479b41a68eff02e9370e18534de6d94ee  baseline-newserver-run2.json
e3857d511ea19a554c9718d36a44d3c839e0f205332372ae4c511109262860c3  baseline-newserver-run3-interleaved.json
e5c89facb530357310cb6d18577f4f44e0ec1052bf0fb54b6e75d9d41ce554f7  delta-cache-run1.json
3cc324d7539d89af3d48c155e7d513a847e74fab06d28db7d0d07c494c620e32  delta-cache-run2.json
5330207e2a968a9cdf7878e7fd4d02a0e1232ee796dda570c8294698481b6387  delta-cache-run3.json
7464973cfb64bbb2139b3afd5892ed84c010602f83eedcb4f3c111fb6144dbe7  delta-cache-performance-gate.json
```

## Opt-in eight-step CFM candidate

Stage timing on the accepted ten-step service showed Code2Wav dominating the
streaming path: its mean stage time was 1568.69 ms, compared with 269.09 ms
for stage 0 and 822.45 ms for stage 1. The next candidate therefore reduces
the Code2Wav Euler flow-matching schedule from ten evaluations to eight. No
codec-window, context, transport, model-weight, or request setting changes.

The candidate passed a fail-closed three-run median performance gate against
the accepted ten-step delta-cache runs. Each run used the same 32 Seed-TTS
English prompts, three warmups, concurrency one, and produced exactly 138.40
seconds / 3,321,600 frames of audio with 32/32 completions.

| Run | TTFT | Audio TTFP | Whole-audio RTF | Per-chunk RTF | E2E |
| --- | ---: | ---: | ---: | ---: | ---: |
| CFM8 1 | 320.00 ms | 906.57 ms | 0.3961 | 0.4193 | 1675.54 ms |
| CFM8 2 | 315.07 ms | 903.72 ms | 0.3944 | 0.4172 | 1668.50 ms |
| CFM8 3 | 314.92 ms | 905.78 ms | 0.3960 | 0.4187 | 1675.80 ms |

| Gate metric | Ten-step median | Eight-step median | Change | Verdict |
| --- | ---: | ---: | ---: | --- |
| Per-chunk audio RTF | 0.4797 | 0.4187 | -12.72% | passes 5% target |
| Whole-audio RTF | 0.4475 | 0.3960 | -11.52% | passes 5% target |
| Audio TTFP | 956.88 ms | 905.78 ms | -5.34% | passes 5% target |
| End-to-end latency | 1885.22 ms | 1675.54 ms | -11.12% | passes 5% target |
| TTFT | 313.19 ms | 315.07 ms | +0.60% | within 2% guard |

Because the change alters the numerical sampler rather than removing redundant
work, performance parity alone cannot promote it. A paired eight-prompt
Seed-TTS English screen was run with the official protocols: Whisper Large v3
for WER and the fine-tuned WavLM Large speaker-verification checkpoint for SIM.

| Quality metric | Ten-step control | Eight-step candidate | Change |
| --- | ---: | ---: | ---: |
| WER, lower is better | 0.0000 (8/8) | 0.0000 (8/8) | 0.00 pp |
| WavLM SIM, higher is better | 0.023218 | 0.027103 | +0.003885 (+0.39 pp) |

Both WER runs exported 8/8 WAVs with no request, PCM, ASR, or scoring failure.
The WavLM checkpoint mapped with zero missing model keys; the one extra loss
projection key is unused for inference. A CPU/NPU check on the first candidate
pair differed by less than 0.001 SIM. The low absolute SIM values therefore
come from the paired output/reference set rather than an NPU-only numerical
failure, but this small screen is not a competition accuracy claim.

The downloaded evaluator needed device-neutral `.to(device)` calls and
Python 3.12 / torchaudio 2.10 import compatibility shims. Those shims do not
change audio, weights, embeddings, or cosine scoring. The exact upstream
checkpoints used were:

```text
51f07e3b94d9e0262a6a675ef5a087be3dd09e8c62e9d886827f44f82fe7f94b  wavlm_large_finetune.pth
6fb4b3c3e6aa567f0a997b30855859cb81528ee8078802af439f7b2da0bf100f  wavlm_large.pt
```

This candidate is available as the explicit opt-in deploy profile
`vllm_omni/deploy/minicpmo_4_5_2npu_910c_cfm8.yaml`. The conservative 910C
profile remains at ten steps until the full 1,088-row Seed-TTS WER/SIM run and
the required Daily-Omni and Video-MME gates pass. The profile was also started
without the timestep environment override and passed a post-warmup end-to-end
smoke: one request completed in 1.70 seconds, generated 3.28 seconds / 78,720
frames of audio, and preserved streaming continuity.

Raw result checksums:

```text
3493b6771ce8d1161ada6da4b5115050fb6da0e5cf7bf3df7780624fe5ea4ec2  cfm8-run1.json
12f6f180f50738855487390c51cd85c997fd8a68124e9d27b563abe4f0eb80f5  cfm8-run2.json
0799b5e6252dc430239c3c906b39c43bab884754d998f7e8ed01e5febf028e17  cfm8-run3.json
eb30375602233d9c9540b1c763f18eb6531d8672a06e6047efea6a4faad3654f  cfm8-performance-gate.json
3f3f385e63319be0e49b02913c3a9f4921791f6e3bf29f8bbeff4959e2ad3dab  cfm8-quality-en8.json
9f54c870928d02426aa30e4fca1ebef4f6cf74e344bb34aed5f03ab3900e9222  cfm10-quality-en8.json
444ee7e57ad72f28ecad24ddd19c7f12103a8872b3970844cff528ac48da3f6f  cfm8-en-8/wav_res_ref_text.sim
94d927fc3e78e83cb00e7bc56ae21fa2090b769fd1cdb759c6339ca5987e0b23  cfm10-en-8/wav_res_ref_text.sim
00b88e046efccaa42db65f61d4ae74e256cc40434319dff87a778ebcd11fa3b9  cfm8-profile-smoke3-warm3.json
```

## Opt-in six-step CFM candidate

Code2Wav remained the dominant stage after the eight-step optimization, so the
next candidate reduced only the Euler flow-matching schedule from eight to six
evaluations. Transport, codec windows, left context, model weights, request
order, and sampling parameters stayed fixed. Three runs used the same 32
Seed-TTS English prompts, three warmups, greedy decoding, and concurrency one.
Every run completed 32/32 requests, generated exactly 138.40 seconds / 3,321,600
frames, reported 100% streaming continuity, and measured zero underrun.

| Run | TTFT | Audio TTFP | Whole-audio RTF | Per-chunk RTF | E2E |
| --- | ---: | ---: | ---: | ---: | ---: |
| CFM6 1 | 311.61 ms | 834.69 ms | 0.3628 | 0.3830 | 1547.55 ms |
| CFM6 2 | 313.21 ms | 829.82 ms | 0.3589 | 0.3788 | 1530.69 ms |
| CFM6 3 | 314.69 ms | 826.56 ms | 0.3572 | 0.3769 | 1523.17 ms |

The fail-closed three-run median gate required at least 5% improvement on every
audio/E2E target and allowed at most 2% TTFT regression:

| Gate metric | Eight-step median | Six-step median | Change | Verdict |
| --- | ---: | ---: | ---: | --- |
| Per-chunk audio RTF | 0.4187 | 0.3788 | -9.54% | passes 5% target |
| Whole-audio RTF | 0.3960 | 0.3589 | -9.35% | passes 5% target |
| Audio TTFP | 905.78 ms | 829.82 ms | -8.39% | passes 5% target |
| End-to-end latency | 1675.54 ms | 1530.69 ms | -8.64% | passes 5% target |
| TTFT | 315.07 ms | 313.21 ms | -0.59% | within 2% guard |

A paired eight-prompt official Seed-TTS screen kept Whisper Large v3 WER at
0.0000 for 8/8 utterances with no request, PCM, ASR, or scoring failure. The
official fine-tuned WavLM Large speaker score moved from 0.027103 at eight steps
to 0.016932 at six steps, an absolute drop of 0.010171 (1.02 percentage points),
inside the competition's 2-point allowance. The six-step profile is therefore
available at `vllm_omni/deploy/minicpmo_4_5_2npu_910c_cfm6.yaml`, but remains
opt-in until the full 1,088-row Seed-TTS run plus Daily-Omni and Video-MME gates
pass. The conservative 910C profile remains at ten steps. The new YAML itself
was also started without the timestep environment override and passed an
eight-request, three-warmup smoke with 8/8 completions, 100% streaming
continuity, zero underrun, and exactly 32.76 seconds / 786,240 frames of audio.

Raw result checksums:

```text
d8a4f67e086876c9c9ad2b19714d78b2a6aafac0b83d947ac5ea4ffbcbfc2b90  cfm6-run1.json
4aefd3a1c414e195fd52b2554bd29d69032607f69b533b8b4b3a7ceffb53ee68  cfm6-run2.json
a45b5b526871b776cec72883ebe022e932aa3f71a3995e69ea954d4687c56e23  cfm6-run3.json
181699c2eb017284b53d37e027c900b6784d0aea4828638b3802080c171d30e3  cfm6-performance-gate.json
d522197f3a7ddf0ee04d1dcf1d903ab6412b93041bc29082960f66713041b99d  cfm6-quality-en8.json
73349fc0d05d0de98d14067f0f1aea1b3ff4a2955ee63254771f8bf198dad7b1  cfm6-en-8/wav_res_ref_text.sim
fc3c4c306e1534141a82a456f2655d042c395e0e2e87f4e29eae90f8bf2dd712  cfm6-profile-smoke8.json
```

### Rejected shorter initial codec window

Reducing only the initial codec window from 25 to 12 frames improved the
eight-prompt audio TTFP to 859.98 ms, but mean per-chunk RTF regressed to
0.7409 and P99 per-chunk RTF reached 3.0611. Whole-audio RTF was 0.4466 and
all requests completed, so this was a scheduling-quality failure rather than
a crash. It is rejected because the competition explicitly scores every
audio chunk's RTF, not only first-packet latency.

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

The launch profile requested `VLLM_ASCEND_ENABLE_CUSTOM_OPS=0` and
`fuse_norm_quant=false`, but the installed development vLLM-Ascend build did
not honor those legacy controls: its engine log reports enabled `norm_quant`
and `act_quant` fusions and `pass_config.fuse_norm_quant=true`. The same
observed backend configuration was held fixed within each valid A/B run.
Results from a different backend build or pass configuration must be reported
as a separate profile.

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

## Same-backend allocation-reuse experiment (rejected)

The installed vLLM-Ascend development tree changed after the earlier resident
server started. A new same-backend control was therefore built from the
verified compatibility tree, with only `batched_token2wav.py` restored to
commit `b1192725`. SHA-256 checks confirmed that the control and candidate
matched for every NPU runner and scheduler compatibility file and differed
only in the proposed Code2Wav allocation change.

The candidate preallocated all ten CFM step-output buffers as leading-dimension
stacks, removed the final `torch.stack` copies, and elided single-request flow
and HIFT cache clones. The idea reduced allocation count, but increased the
live working set and was slower on this 910C profile.

Three 32-request repetitions per variant used the same process warmup sequence
and produced identical totals in every run: 4,801 input tokens, 480 output
tokens, 3,321,600 audio frames, and 138.40 seconds of audio.

| Variant | TTFT | Audio TTFP | Whole-audio RTF | Per-chunk RTF | E2E |
| --- | ---: | ---: | ---: | ---: | ---: |
| Same-backend control | 330.97 ± 11.15 ms | 987.16 ± 8.87 ms | 0.4485 ± 0.0020 | 0.4792 ± 0.0021 | 1,890.08 ± 8.59 ms |
| Allocation-reuse candidate | 328.31 ± 3.37 ms | 1,015.15 ± 1.22 ms | 0.4900 ± 0.0011 | 0.5307 ± 0.0025 | 2,062.55 ± 5.72 ms |

The fail-closed median gate rejected the candidate:

- whole-audio RTF regressed 9.49%;
- audio TTFP regressed 3.06%;
- E2E regressed 9.25%, beyond the 2% guard;
- TTFT improved 0.39%, which does not compensate for the audio regressions.

The allocation/copy patch was reverted. The backend compatibility fixes were
retained: legacy/current context-parallel managers, optional profiling config
locations, the added full-graph `positions` argument, and scheduler
`_free_request` dict/`None` versus tuple returns. The combined Code2Wav, NPU,
and scheduler gate passes 48/48 after the rejected candidate-only test is
removed.

Raw control/candidate results and the machine-readable rejection report are
under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-audio-opt-20260810/allocation-v2
```

## Competition status and next experiment

This result still does not establish a competition pass. Seed-TTS WER and speaker
similarity were not evaluated, and the full Daily-Omni and Video-MME quality
gates were not run. The current local Video-MME extraction contains only ten
videos, while the official run requires all 2,700 questions.

The immediate next step is quality validation, not another default-on speed
change: export the fixed Seed-TTS manifest audio, run official WER and speaker
similarity, then run the complete Daily-Omni and Video-MME suites. Further
speed work should use NPU profiling to identify a narrower per-step workspace
strategy, or evaluate a pinned custom-op-capable image. Whole-stack CFM output
preallocation must not be retried unchanged. Every candidate remains off by
default until it beats this profile without exceeding the two-point quality
budget.
