# MiniCPM-o 4.5 on Ascend 910C: official Seed-TTS comparison

Date: 2026-08-09; optimization update: 2026-08-17

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
| --- | ---: | ---: | ---: | ---: |
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

## Opt-in fixed-width DiT MLP graph partition

The full Code2Wav graph could not tolerate the estimator's internal-format
Conv2D or growing attention-cache shapes. The narrower partition leaves
attention and convolution eager and compiles only the affine-free `norm2`,
modulation, two-layer MLP, gate, and residual expression. It runs only for
steady streaming chunks with CFG batch 2 and width 50; setup, final, and
mismatched shapes stay eager. One weight-parameterized TorchAir graph is
shared by all 16 DiT blocks. Compile and replay failures fall back to eager.

The installed TorchAir build also has an import-order defect when vLLM has
already registered `npu_define::broadcast`: its converter skips a local
`op_broadcast` alias that a later converter imports. The adapter repairs that
missing alias from the registered operator without changing torch-npu.
Startup logged a successful fixed-width compile, and the first live request
logged graph replay with no fallback.

Three candidate and three same-era control runs used 32 fixed Seed-TTS English
prompts, three warmups, concurrency one, greedy decoding, and identical CFM6
settings. Every run completed 32/32 requests with 100% streaming continuity
and exactly 138.40 seconds / 3,321,600 frames of audio.

| Run | TTFT | Audio TTFP | Whole-audio RTF | Per-chunk RTF | E2E |
| --- | ---: | ---: | ---: | ---: | ---: |
| CFM6 control 1 | 316.42 ms | 860.70 ms | 0.3825 | 0.4065 | 1628.55 ms |
| CFM6 control 2 | 312.66 ms | 823.82 ms | 0.3606 | 0.3798 | 1535.94 ms |
| CFM6 control 3 | 463.18 ms | 975.96 ms | 0.3923 | 0.4182 | 1691.61 ms |
| MLP graph 1 | 317.85 ms | 848.38 ms | 0.3653 | 0.3853 | 1553.95 ms |
| MLP graph 2 | 314.51 ms | 842.88 ms | 0.3607 | 0.3807 | 1537.23 ms |
| MLP graph 3 | 333.34 ms | 858.88 ms | 0.3672 | 0.3852 | 1554.48 ms |

The host produced latency spikes in control run 3, so the result uses the
predeclared three-run median rather than selecting the best run:

| Gate metric | CFM6 median | MLP graph median | Change |
| --- | ---: | ---: | ---: |
| Per-chunk audio RTF | 0.4065 | 0.3852 | -5.25% |
| P99 per-chunk audio RTF | 1.1587 | 1.1346 | -2.08% |
| Whole-audio RTF | 0.3825 | 0.3653 | -4.49% |
| Audio TTFP | 860.70 ms | 848.38 ms | -1.43% |
| End-to-end latency | 1628.55 ms | 1553.95 ms | -4.58% |
| TTFT | 316.42 ms | 317.85 ms | +0.45% |

The primary per-chunk metric clears the 5% target, P99 also improves, and TTFT
stays inside the 2% guard. The other audio/E2E metrics improve but do not all
clear 5%, so this is an opt-in incremental promotion rather than a new default.

An eight-prompt graph-path quality screen completed 8/8 requests and exported
all WAVs. Whisper Large v3 WER remained 0.0000 for 8/8 with no ASR or PCM
failure. Official fine-tuned WavLM Large SIM was 0.018103 versus CFM6's
0.016932, an improvement of 0.001171 (+0.12 percentage points). The profile is
`vllm_omni/deploy/minicpmo_4_5_2npu_910c_cfm6_dit_mlp_graph.yaml`. Environment
variables can override its switch or width for fail-closed operations. A
profile-only restart (without the graph environment switch) then completed an
8/8 smoke with the exact 1,197 input-token, 116 output-token, and 786,240-frame
signature; the service log confirmed graph compilation and replay. The full
1,088-row Seed-TTS, Daily-Omni, and Video-MME gates remain required.

Raw result checksums:

```text
cf6bdbf5b7356c6f4c05d6540793ddf7a0ba3c15290af7054663e28c045cc132  cfm6-same-era-control-run1.json
a11a68f8095ebc40ef1e490a6f4e10aca02a0f67bd525a95bfb4efd2ef6e55d4  cfm6-same-era-control-run2.json
e341a54432077b4c7209c40d36fce8048df62f269a5afd19f54918b838ca9856  cfm6-same-era-control-run3.json
d6e4de70dc62bd082c35f076fe1916fa2d53fd6cc44ccc3e53cc63f3b75a800a  cfm6-dit-mlp-graph-run1.json
153a7004fc6fea076b61680f5c751ed444eb67444ee62c7cdd6151985b4abb4f  cfm6-dit-mlp-graph-run2.json
c43a425190f8a3ee8053386942f68f92e73e58a131837e9e24c6f43d839b4577  cfm6-dit-mlp-graph-run3.json
5dc89bf3d864595ac84a3fca73c70aae7309569729561b54c236b078b7757b87  cfm6-dit-mlp-graph-quality-en8.json
46072dcc2a66ddea019decdeef11eff89e5addb820bfc627360fc70c3732d289  cfm6-dit-mlp-graph-en-8/wav_res_ref_text.sim
a4c294497a180caf079a4c73d60213ef5514a84ddf4cc7fb394202544bab3e84  cfm6-dit-mlp-graph-profile-smoke8.json
```

### Rejected shorter initial codec window

Reducing only the initial codec window from 25 to 12 frames improved the
eight-prompt audio TTFP to 859.98 ms, but mean per-chunk RTF regressed to
0.7409 and P99 per-chunk RTF reached 3.0611. Whole-audio RTF was 0.4466 and
all requests completed, so this was a scheduling-quality failure rather than
a crash. It is rejected because the competition explicitly scores every
audio chunk's RTF, not only first-packet latency.

## Rejected full-loop NPU graph experiment

NPU CFM graph replay remains disabled. The initial graph path failed because
the upstream timestep embedder copied a CPU tensor during capture. After the
cache removed that transfer, capture reached an unsupported internal-format
Conv2D. Setting `torch.npu.config.allow_internal_format=False` made capture
succeed, but a four-request smoke run regressed mean audio RTF from about 0.45
to 1.43, per-chunk RTF to 1.60, and raised logical-chip-1 HBM from about 45.4
GiB to 51.5 GiB. Growing attention-cache shapes are part of the graph key, so
the observed behavior is consistent with capture churn and cache eviction.

That full-loop runtime change was reverted. The narrower fixed-width MLP
partition described above avoids both failure modes; attention-cache bucketing
would still be required before expanding graph coverage around attention.

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

## Post-CFM6 910C profiling and rejected candidates

A fresh CFM6 service was profiled on the same Atlas A3 / 910C host after the
six-step profile was accepted. The stage-2 kernel trace attributed 451.6 ms
of device time as follows:

| Operator family | Device time | Share |
| --- | ---: | ---: |
| TransData | 98.33 ms | 21.77% |
| Transpose | 65.80 ms | 14.57% |
| MatMulV2 | 45.94 ms | 10.17% |
| LayerNormV3 | 42.03 ms | 9.31% |
| Mul | 35.97 ms | 7.97% |
| Add | 31.68 ms | 7.02% |
| ConcatD | 26.86 ms | 5.95% |
| Slice | 24.49 ms | 5.42% |
| FlashAttention | 16.64 ms | 3.69% |
| Conv2D | 15.53 ms | 3.44% |

The host trace also reported 2,973 `aclnnCat` calls taking 749 ms, 3,666
`aclnnAdd` calls taking 497 ms, 3,744 `aclnnAddmm` calls taking 381 ms, and
1,758 `aclnnAdds` calls taking 319 ms. The next major target is therefore the
DiT block's layout/construction and launch overhead, not FlashAttention or the
Conv2D arithmetic alone.

The exact benchmark protocol matters. In particular, MiniCPM-o's TTS template
kwargs must be nested in the HTTP extra body; the benchmark CLI's top-level
`--chat-template-kwargs` does not populate the custom Omni backend payload.
The validated request fragment is:

```bash
vllm bench serve --omni \
  --backend openai-chat-omni \
  --endpoint /v1/chat/completions \
  --model openbmb/MiniCPM-o-4_5 \
  --tokenizer /models/OpenBMB/MiniCPM-o-4_5 \
  --trust-remote-code \
  --dataset-name seed-tts \
  --dataset-path /benchmarks/seedtts_testset \
  --seed-tts-root /benchmarks/seedtts_testset \
  --seed-tts-locale en \
  --num-prompts 32 --num-warmups 3 \
  --max-concurrency 1 --request-rate inf --seed 0 --temperature 0 \
  --extra-body '{"modalities":["text","audio"],"chat_template_kwargs":{"use_tts_template":true,"enable_thinking":false}}'
```

Every valid 32-request run produced the same structural signature: 4,801
input tokens, 480 output tokens, 3,321,600 audio frames, and 138.40 seconds of
audio. A fresh three-run CFM6 control reproduced that signature:

| Run | TTFT | Audio TTFP | Whole-audio RTF | Per-chunk RTF | E2E |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fresh CFM6 1 | 313.68 ms | 824.42 ms | 0.3693 | 0.3948 | 1573.71 ms |
| Fresh CFM6 2 | 315.70 ms | 836.57 ms | 0.3731 | 0.3983 | 1588.51 ms |
| Fresh CFM6 3 | 450.54 ms | 965.92 ms | 0.4130 | 0.4295 | 1715.40 ms |

The third run contains a host-latency excursion, so promotion decisions use
the three-run median and a same-era control rather than selecting the best
run. The following candidates were rejected and fully reverted:

| Candidate | Per-chunk RTF | Whole RTF | Audio TTFP | E2E | Reason |
| --- | ---: | ---: | ---: | ---: | --- |
| Five CFM steps | +1.91% | +3.54% | +13.25% | +4.17% | Slower than fresh CFM6; no quality budget spent |
| Split K/V cache | +1.90% | +1.66% | +4.79% | +1.58% | More layout/view overhead than concatenation saved |
| Fused CFG/Euler expression | +1.16% | +1.38% | -0.15% | +1.38% | Small TTFP win, audio/E2E regression |
| In-place CFG/Euler update | +0.57% | +0.76% | -0.32% | +0.76% | Small TTFP win, audio/E2E regression |

The first post-profile kernel experiment was also rejected before service
integration. It fused the affine-free LayerNorm plus AdaLN shift/scale into
one Ascend Triton kernel and fused the final AdaLN gate plus residual into a
second kernel for the exact steady Code2Wav shape: BF16 `[2, 50, 512]` with
`[2, 1, 512]` conditioning. After 100 warmups, a 1,000-iteration synchronized
screen on the free logical 910C device measured:

| Exact-shape expression | Native eager | Triton prototype | Change |
| --- | ---: | ---: | ---: |
| LayerNorm + shift/scale | 47.74 us | 82.80 us | +73.43% |
| Gate + residual | 20.81 us | 85.80 us | +312.31% |

Lower is better. The LayerNorm path also differed from the native BF16 result
by 0.0625 maximum absolute and 0.00230 mean absolute because the fused kernel's
reduction order did not reproduce the stock operator. The prototype was
removed. This rules out small pointwise Triton replacements on this workload;
the next kernel attempt must cover a substantially larger, layout-aware DiT
boundary and eliminate enough `TransData`, transpose, and host launches to
amortize custom-kernel dispatch.

Full-loop NPU graph replay remains rejected. Native `NPUGraph` cannot capture
the estimator's internal-format Conv2D. Disabling internal formats allowed a
TorchAir graph to capture, but its first compilation took 108.5 seconds and
new streaming-cache shapes triggered 10--17 second recompilations. A viable
implementation needs fixed cache-shape buckets and an eager boundary around
the convolution path before it can be benchmarked again.

A 25/50-frame chunk experiment kept the first packet at 25 codec frames and
doubled only steady-state windows. Its eight-request screen preserved 786,240
frames / 32.76 seconds and continuity, improved whole-audio RTF by 10.08%,
audio TTFP by 2.04%, and E2E by 10.12%, but regressed mean per-chunk RTF by
12.41%, P99 per-chunk RTF by 0.25%, and TTFT by 2.26%. It was rejected because
the competition scores chunk delivery, not only aggregate completion time.

Raw result checksums:

```text
3f26e832766af342c2901b8a74faddc7bb4ab889fbd164cd4c86e84a2f20fb9d  cfm6-fresh-control-run1.json
955c87cbebceebfd019d3ed7a0db13ddd3121486e16b98f5cfd715b814c5f38e  cfm6-fresh-control-run2.json
52ede14e3c277d6bc4e09ef8616984c2dcde088dafd88f8b3249fe1535e05d28  cfm6-fresh-control-run3.json
7145110de7b5f12cc1d174c9ac2e0b47c5726200351da2373f15692942f38d38  cfm5-run1.json
8ac7f5b5477c5ba0043509abf541d951fe9e935d62f8fbf26f536ac19898be0c  cfm5-run2.json
93ffd114d255a0b2cd56d4f590be3a98e5da788e1d3bf74f56b22ab84f8a9004  cfm5-run3.json
3dfc1e81ca830dad3e86afe34293dd50d9e20e4129ca4e7905d4805bbda25350  cfm6-chunk50-smoke8.json
```

## Accepted Talker repetition-frequency cache

A focused Stage-1 trace captured 340 Talker codec steps. `Bincount` ran on AI
CPU 336 times for 48.14 ms (6.16% of traced device time, 143.3 us average).
The host trace also recorded 2,732 scalar reads. The longest reads immediately
followed `torch.multinomial`, while another synchronization occurred inside
`bincount`. This made the repetition histogram a narrower target than changing
the Talker transformer or its already-replaying decode graphs.

The accepted implementation stores the exact 16-code frequency vector per
request and advances it with device-native equality/add/subtract operations.
It does not change the seed, logits warpers, sampling order, EOS check, codec
history, or Code2Wav inputs. Focused CPU tests compare both the full penalty
and incremental window eviction against the former `torch.bincount` result.
On NPU 1, an isolated 1,000-iteration microbenchmark reduced the hot section
from 434.77 us mean / 412.48 us P50 to 254.46 us mean / 247.11 us P50, a 41.5%
mean reduction.

The synchronization was optimized separately and rejected. Skipping
`sampled.item()` while EOS was masked preserved structure but regressed the
three-run median whole RTF by 2.77%, chunk RTF by 2.66%, TTFP by 2.37%, and
E2E by 2.87%. The change was fully reverted before measuring the frequency
cache; shared Talker/Code2Wav pacing on logical NPU 1 is part of the measured
system behavior.

The cache candidate used the same CFM6 + fixed-width DiT MLP graph service,
32 fixed English prompts, three warmups, concurrency one, greedy outer-token
decoding, and nested MiniCPM-o TTS template body as the fresh control. Every
run completed 32/32 with 4,801 input tokens, 480 output tokens, 3,321,600
frames, 138.40 seconds of audio, and 100% streaming continuity.

| Run | TTFT | Audio TTFP | Whole-audio RTF | Per-chunk RTF | P99 chunk RTF | E2E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cache 1 | 319.42 ms | 829.88 ms | 0.3503 | 0.3699 | 1.1115 | 1488.39 ms |
| Cache 2 | 713.75 ms | 1225.09 ms | 0.4413 | 0.4691 | 1.9112 | 1887.05 ms |
| Cache 3 | 319.48 ms | 833.82 ms | 0.3520 | 0.3718 | 1.1049 | 1494.89 ms |

Run 2 is retained as a host/NPU latency excursion rather than discarded. The
predeclared three-run median remains favorable against the immediately prior
three-run control:

| Gate metric | Control median | Cache median | Change |
| --- | ---: | ---: | ---: |
| Per-chunk audio RTF | 0.3762 | 0.3718 | -1.16% |
| P99 per-chunk audio RTF | 1.1230 | 1.1115 | -1.03% |
| Whole-audio RTF | 0.3565 | 0.3520 | -1.24% |
| Audio TTFP | 835.97 ms | 833.82 ms | -0.26% |
| End-to-end latency | 1515.55 ms | 1494.89 ms | -1.36% |
| TTFT | 317.31 ms | 319.48 ms | +0.69% |

TTFT remains inside the 2% guard while all audio and E2E targets improve. An
eight-prompt Seed-TTS screen then completed 8/8 with the exact 1,197 input,
116 output, 786,240-frame, and 32.76-second signature. Whisper Large v3 WER
was 0.0000 for 8/8, with zero request, PCM, ASR, or export failures.

Raw result checksums:

```text
4fef770978c2f0512a8379ffbefab3371512fe2211fa33d2a695bdad97910b8d  talker-frequency-cache-run1.json
299632ce53f79e5afe3191ce818f8b58ea23da1240abc054972d988d617b4c85  talker-frequency-cache-run2.json
e4e32e787791daa3f458cca93c00079bab1b1f7ffffef96a9c3d6bf9a0dff52a  talker-frequency-cache-run3.json
85fb688b5e9dfac6a407a20ebcefe14fb4d97f7c214684e6e19c34677bbb1024  talker-frequency-cache-quality-en8.json
```

## Full Seed-TTS qualification of the accepted cache

The accepted cache profile completed the complete 1,088-row English Seed-TTS
manifest. All 1,088 requests and all 1,088 audio exports succeeded, with zero
request, PCM, ASR, or export failures. The run produced 163,442 input tokens,
16,562 output tokens, 118,310,400 PCM frames, and 4,929.6 seconds of audio.

| Metric | Full 1,088-row result |
| --- | ---: |
| Mean TTFT | 579.73 ms |
| Mean audio TTFP | 1,091.71 ms |
| Whole-audio RTF | 0.4085 |
| Per-chunk RTF | 0.4305 |
| P99 per-chunk RTF | 2.0963 |
| Mean E2E | 1,812.85 ms |
| Whisper Large v3 mean WER | 0.03327 |
| Whisper Large v3 median WER | 0.00000 |
| Official WavLM speaker SIM | 0.02903 |

The full run therefore promotes the Talker repetition-frequency cache past its
Seed-TTS quality gate. These absolute WER and SIM values describe this model,
prompt template, and evaluator combination; the competition comparison must
use the same evaluator and dataset revision for both framework baseline and
optimized service.

```text
2def4ce17d95d16e45fb375bcd02cf24b04121353700c9888bdbdeb076c7f5f8  talker-frequency-cache-quality-full-en-1088.json
8d22eea2f2bc3c96bd284f4c0918d917d084d8197d1206469abf36bb5fae5a27  wav_res_ref_text.sim
```

## Post-cache Talker profile and rejected sampling candidates

A second Stage-1-only profile captured exactly 340 Talker codec steps after the
cache landed. `Bincount` was absent. The largest remaining sampling-related
host-to-device queue costs were `Multinomial` (89.25 ms across 340 calls),
`Softmax` (7.10 ms), `Sort` (6.62 ms), `Topk` (5.98 ms), `Cumsum` (5.20 ms),
and masked fill (6.65 ms). Together, the Top-P/Top-K block accounted for about
31.55 ms of traced queue time and was the next bounded target.

An experimental adapter around `torch_npu.npu_top_k_top_p` reduced an isolated
FP32 microbenchmark mean from 446.93 us to 378.17 us (-15.38%). It was not
accepted. Exact mask parity passed all 98 FP32 and FP16 cases but failed 2 of
98 BF16 cases, which is the live Talker dtype. More importantly, a clean
reverse-control 3x32 service comparison showed a median per-chunk RTF
regression of 8.94%, whole-audio RTF regression of 9.17%, audio TTFP
regression of 6.96%, P99 chunk RTF regression of 6.97%, and E2E regression of
9.25%. TTFT also regressed 2.46%, beyond the 2% guard. All paired output and
chunk hashes differed.

The candidate's eight-row Seed-TTS screen completed with WER 0.0000. Its SIM
was 0.02053 versus 0.03446 for the matching first eight rows of the accepted
full run, a 1.39 percentage-point drop. Although that screen stayed inside the
competition's two-point accuracy allowance, it cannot compensate for BF16
parity loss and the measured serving regressions. The implementation and its
runtime flag were fully removed; this is a documented negative result, not a
dormant feature.

The next candidate reused vLLM-Ascend's exponential-race `random_sample`
implementation to replace `torch.multinomial`. It was tested on idle logical
NPU 1 with the live 6,562-token probability-vector shape and the required
scalar read, using 100 warmups plus 1,000 measured iterations. All 1,100 draws
from each path were valid, but the candidate was substantially slower:

| Sampler path | Mean | P50 | P99 |
| --- | ---: | ---: | ---: |
| `torch.multinomial` | 219.98 us | 210.29 us | 245.35 us |
| Ascend exponential race | 383.33 us | 351.67 us | 449.98 us |
| Change | +74.25% | +67.23% | +83.40% |

The model already performs the scalar read needed for EOS and request-state
updates, so the generic sampler's synchronization-avoidance rationale does not
translate into a win here. Its global-stream handoff, exponential fill, divide,
and argmax cost more than the live `multinomial` call at batch one. The opt-in
implementation and environment flag were removed before service A/B testing.

## Competition-exact Daily-Omni protocol audit

Early Daily-Omni screens were far below the organizer's 79.5% framework
baseline even though the 1 fps image/audio interleave itself matched MiniCPM.
The remaining mismatch was in the conversation contract. The client was still
using Daily-Omni's Qwen instruction, retained an empty system role, allowed 256
output tokens, and did not reproduce MiniCPM-o's chat-template arguments.

The OpenBMB OmniEvalKit `daily_omni` configuration instead uses its own strict
MCQ prompt, 128 output tokens, 64 frames at 1 fps, interleaved audio, and no
system role when the configured system prompt is empty. A later audit of the
competition's current vLLM-Omni deployment guide found one important adapter
difference: the competition request pins `enable_thinking=false` but does not
enable `use_tts_template`. It also pins Stage 0 to greedy decoding with
`repetition_penalty=1.2` and `max_tokens=128`; the repetition penalty must live
in the deploy YAML because the Omni stage owns it. The benchmark client now
matches that contract and keeps `modalities=["text"]`, so Talker and Code2Wav
do no irrelevant work.

Three concurrency-one screens on the same first 32 questions isolated the
conversation and deploy-contract changes:

| Protocol | Correct | Accuracy | HTTP / parse failures | Duration | Mean TTFT |
| --- | ---: | ---: | ---: | ---: | ---: |
| MiniCPM prompt, but template flags omitted | 23/32 | 71.875% | 0 / 0 | 39.65 s | 1,162.13 ms |
| Direct-HF-adapter template flags (`use_tts_template=true`) | 25/32 | 78.125% | 0 / 0 | 46.01 s | 1,398.70 ms |
| Competition contract, Stage 0 repetition penalty 1.2 | 27/32 | 84.375% | 0 / 0 | 63.58 s | 1,949.18 ms |

This is an evaluator-correctness fix, not a claimed model-speed improvement.
It prevents a fast but invalid benchmark from being promoted. The competition
screen result is `organizer-protocol-daily-32-c1.json` (SHA-256
`d9d8b1f8055962e39f715f5ff25f719dd21b4cc63cfe9d90f96da4b692fa332e`).

The official Hugging Face `qa.json` contains 1,197 rows over 684 videos
(SHA-256 `3210a45d42424c7d57c1b40a0b9aa2708fc02fab2364bf01fd7d16e1242e146b`).
The previous local conversion had 1,196 rows because it deduplicated one
intentional repeated `(video_id, question)` pair; all 684 video/audio assets
were already complete. Full qualification now uses the untouched official
1,197-row annotation rather than explaining away the mismatch.

## Thinker-only repeated-prefix candidate

Daily-Omni and Video-MME place media before the question and revisit the same
video for multiple questions. An opt-in 910C profile now enables vLLM prefix
caching and the uniform KV manager on Stage 0 only. Talker and Code2Wav retain
the inherited disabled setting. This can reuse both KV blocks and
vLLM-Omni's multimodal hidden-state prefix cache, but it also consumes pinned
host memory and must not be promoted before a fresh-process A/B measures
accuracy, cache hits, host/NPU memory, mean latency, and P99 latency. The
source order is intentionally retained; no benchmark-only grouping is used.

The benchmark now records OpenAI `prompt_tokens_details.cached_tokens` per
request, so a latency change is not attributed to caching without a real hit.
On a cache-empty, seeded-shuffle 32-row c1 run, the candidate recorded only
3/32 hits (9,728 cached prompt tokens). Against the matching accepted service,
duration regressed 3.25%, mean TTFT regressed 2.90%, and P99 TTFT regressed
29.83%, with exact 20/32 accuracy parity. A cache-empty c10-first run recorded
the same three hits. Against a cache-empty accepted reverse control it improved
throughput by 2.77% and mean E2E by 3.14%, but mean TTFT regressed 0.41% and
P99 TTFT regressed 8.42%.

The mechanism is useful for a different workload shape. On an explicit
unshuffled repeated-media c1 screen, 19/32 requests hit (45,184 cached tokens),
duration improved 42.47%, and mean TTFT improved 42.89%, with exact 27/32
accuracy parity. A deliberately pre-warmed c10 upper bound hit 32/32 and ran
12.93x faster. Those are session/repeated-media results, not competition
results. The candidate remains opt-in and is not promoted into the organizer
profile because the realistic cache-empty distribution fails the P99 guard.

## Full competition Daily-Omni 8K control

The original c4 + 8K competition control completed the untouched 1,197-row annotation
at concurrency 10:

| Metric | Result |
| --- | ---: |
| Accuracy | 937/1,197 = 78.279% |
| Organizer gate | >= 77.5% (pass) |
| HTTP successes / failures | 1,197 / 0 |
| Parse failures | 3 |
| Duration | 3,176.65 s |
| Throughput | 0.3768 req/s |
| Mean / P99 TTFT | 21.840 / 33.101 s |
| Mean / P99 E2E | 26.478 / 40.644 s |

The detailed result is
`organizer-protocol-daily-full-1197-c10.json` (SHA-256
`49fc5cba86f8a57e72cc43982fae11612eb04460d98a6f9ac4e6ad8a3f584678`).
It exactly matches the organizer reference count of 937 correct answers.

## Thinker c10 admission experiment

The full run exposed the actual performance bottleneck: the client submits ten
requests while Stage 0 admits and captures only four sequences. An isolated
profile raises Stage 0 `max_num_seqs` to 10 and adds decode graph capture sizes
8 and 10, while retaining the 8,192-token scheduler budget. On cache-empty
c10-first screens it completed in 78.90 and 78.39 seconds versus 98.04 seconds
for the accepted reverse control. The first comparison improved throughput by
24.25%, mean TTFT by 26.64%, P99 TTFT by 18.67%, mean E2E by 15.54%, and P99
E2E by 20.08%, with exact 20/32 accuracy parity.

The full 1,197-row run showed why the screen is not enough. c10 improved
duration by 4.41%, throughput by 4.61%, mean TTFT by 20.33%, and mean E2E by
4.29%. It passed the accuracy gate at 936/1,197 = 78.195%, a 0.084-point
change from the accepted run, with zero HTTP failures and the same three parse
failures. However, P99 TTFT regressed 2.39% and P99 E2E regressed 38.03%.
The detailed result is
`organizer-protocol-daily-full-1197-c10-thinker-c10.json` (SHA-256
`93c407d378e8a9111740d29ab4f2afd827e6be53c8a4a0e0e9d6bd487b1fa48f`).
c10 therefore remains an experiment rather than replacing the accepted
profile. The next admission point is c8, with the same 8,192-token budget and
decode graph coverage through batch eight.

The full c8 run improved duration by 4.74%, throughput by 4.97%, mean TTFT by
20.95%, P99 TTFT by 19.88%, and mean E2E by 4.61%. Accuracy passed at
938/1,197 = 78.363%, with zero HTTP failures and two parse failures. It also
reduced c10's P99 E2E from 56.101 seconds to 52.752 seconds, but that is still
a 29.79% regression from the accepted c4 run's 40.644 seconds. The detailed
result is `organizer-protocol-daily-full-1197-c10-thinker-c8.json` (SHA-256
`bc2ecc0b71cc8a3873c7b2cf53dabafbe53238543d683ef6823652947d57182d`).
c8 therefore remains an optional mean-TTFT/throughput profile. c6 is the next
bounded experiment for recovering the full-run E2E tail.

## Thinker c6 admission screen

The bounded c6 profile keeps the same 8,192-token Stage 0 scheduler budget and
adds only batch six to the accepted decode-graph shapes. On the same seeded,
cache-empty, shuffled 32-row concurrency-ten screen, it preserved exact 20/32
accuracy and completed all requests. Against the matching accepted c4 reverse
control it improved duration by 17.60%, throughput by 21.35%, mean TTFT by
19.53%, P99 TTFT by 7.66%, mean E2E by 14.65%, and P99 E2E by 14.20%.

| Metric | Accepted c4 | c6 | Change |
| --- | ---: | ---: | ---: |
| Duration | 98.035 s | 80.783 s | -17.60% |
| Throughput | 0.3264 req/s | 0.3961 req/s | +21.35% |
| Mean / P99 TTFT | 23.058 / 32.277 s | 18.555 / 29.806 s | -19.53% / -7.66% |
| Mean / P99 E2E | 27.973 / 39.132 s | 23.875 / 33.576 s | -14.65% / -14.20% |
| Accuracy | 20/32 | 20/32 | exact parity |

The detailed screen result is
`thinker-c6-candidate-shuffled-c10-first.json` (SHA-256
`4d7bef032e21c86caae8a525dfa7297578c17a050bceaa1d55b13ca371793cdf`).
The screen qualified c6 for the full 1,197-row run but did not by itself
promote the profile. The predeclared full-run guard was zero HTTP failures,
accuracy at or above 77.5%, and P99 E2E no more than 2% above the accepted c4
result (41.457 seconds).

The full run completed all 1,197 requests and improved accuracy from 937 to 942
correct (78.697%), with two parse failures. It improved duration by 5.38%,
throughput by 5.69%, mean TTFT by 13.98%, P99 TTFT by 13.35%, and mean E2E by
5.31%. P99 E2E nevertheless regressed 24.24%, from 40.644 to 50.495 seconds,
so c6 is not promoted. The detailed result is
`organizer-protocol-daily-full-1197-c10-thinker-c6.json` (SHA-256
`e540600ec58d5d01bd102525a9c50eace758cff7135916bddc3f388a48adfe62`).

Unlike the earlier full artifacts, this result contains aligned per-request
IDs and E2E values. They locate the regression after first token: output length
correlates 0.896 with post-TTFT time, while input length correlates only 0.060.
Only 29 responses exceeded eight tokens, but those valid `A. <option text>`
completions dominate P99. The two 18--19-token responses spent 42--50 seconds
after first token. The next bounded experiment therefore targets decode
admission using an exact replay of those 29 prompts; a larger prefill budget is
not a plausible fix for this tail.

## Thinker c5 final admission point

c5 is the only integer admission point between the accepted c4 profile and c6.
An exact replay corpus was built from the 29 c6 requests whose output exceeded
eight tokens (SHA-256
`425bdc7ad6cdb7d956b7b15ea43b1154dfc923d11554b6a9c2925ac4070b25ea`).
On this deliberately tail-heavy set, c5 preserved 23/29 accuracy and all 29
successful requests while reducing P99 E2E from 45.781 to 42.110 seconds
(-8.02%). It also reduced maximum post-TTFT time from 18.379 to 16.176 seconds.
The cost was 5.54% lower throughput, 16.19% worse mean TTFT, and 4.18% worse
mean E2E. The c6 and c5 replay artifacts have SHA-256 values
`3afac3bd162068c0e11070d0a1b8c63b4169cb6766696392778821ced17797ae`
and `fe7fa12d6c74b000145705e767592809e2c3017c262a04f24b9aa26b8dab5f75`.

The full c5 run completed all 1,197 requests and passed accuracy at 935/1,197 =
78.111%, with three parse failures. Against c4 it improved duration by 4.73%,
throughput by 4.96%, mean TTFT by 9.21%, P99 TTFT by 11.79%, and mean E2E by
4.69%. P99 E2E still regressed 12.02%, from 40.644 to 45.530 seconds, and
failed the same 41.457-second guard. The detailed result is
`organizer-protocol-daily-full-1197-c10-thinker-c5.json` (SHA-256
`4db216d305bb3484a1f725d10d17a6fab8c64c1940f9fa4f81385a30c049a671`).

The admission sweep is therefore closed: c5, c6, c8, and c10 all improve mean
performance but fail the predeclared full-run E2E-tail guard. c4 remains the
competition default. Further admission points would either duplicate an
already measured integer point or require an unsupported fractional policy.

## Qualified c4 + 16K Thinker prefill budget

With decode admission fixed at c4, the next candidate doubled only Stage 0's
`max_num_batched_tokens` from 8,192 to 16,384. On the matching seeded,
cache-empty 32-row concurrency-ten screen, it preserved 20/32 accuracy and all
32 successful requests. It improved duration by 21.18%, throughput by 26.88%,
mean/P99 TTFT by 14.35%/11.20%, and mean/P99 E2E by 21.88%/26.65%. The screen
artifact is `competition-c4-prefill16k-shuffled-c10-first.json` (SHA-256
`427276b727c155c612589e06e59b0a8f2c9e17b3db6861a4a699303a97ed4d8f`).

The full 1,197-row qualification passed every predeclared promotion condition:

| Metric | c4 + 8K control | c4 + 16K | Change |
| --- | ---: | ---: | ---: |
| Accuracy | 937/1,197 = 78.279% | 937/1,197 = 78.279% | exact aggregate parity |
| HTTP failures | 0 | 0 | pass |
| Parse failures | 3 | 4 | accuracy unaffected |
| Duration | 3,176.652 s | 3,002.472 s | -5.48% |
| Throughput | 0.3768 req/s | 0.3987 req/s | +5.80% |
| Mean / P99 TTFT | 21.840 / 33.101 s | 21.318 / 31.653 s | -2.39% / -4.38% |
| Mean / P99 E2E | 26.478 / 40.644 s | 25.020 / 40.289 s | -5.51% / -0.87% |

The 40.289-second P99 E2E is 1.168 seconds below the predeclared 41.457-second
guard. Per-request text and correctness are not bit-identical across the two
batching policies, as expected for load-sensitive inference, but the aggregate
accuracy is identical and remains 0.779 percentage points above the organizer
gate. The qualified artifact is
`organizer-protocol-daily-full-1197-c10-competition-prefill16k.json` (SHA-256
`156a53c9ee29d4927f75d75d386c88e88fa5265489a2ff73709b3dbc0bb1397e`).

The named `minicpmo_4_5_2npu_910c_cfm6_dit_mlp_graph_competition.yaml`
profile now carries c4 + 16K. The prior control remains reproducible as
`minicpmo_4_5_2npu_910c_cfm6_dit_mlp_graph_competition_prefill8k.yaml`, and
all measured admission and prefix-cache experiments continue to inherit that
8K replay base.

## Video-MME official-adapter screen

The first fail-closed real-service screen used the official 2,700-row parquet,
local organizer videos, MiniCPM frame packing, 96 frames, no audio or subtitle,
greedy decoding, 128 output tokens, and concurrency four. The adapter first
prewarmed 32 unique videos with four bounded workers. A cold cache completed in
71 seconds and persisted 96-frame JPEG sets outside the timed request section.

All 32 requests completed, 22 answers were correct (68.75%), and there were no
HTTP failures. This small screen is above the 67.0% organizer gate but is not a
substitute for the full 2,700-row accuracy run. Mean/P99 TTFT were
12.534/19.134 seconds and mean/P99 E2E were 14.278/21.343 seconds. The saved
artifact contains aligned request IDs, per-request TTFT/E2E values, and all 32
evaluation records. Its SHA-256 is
`4770f4becab903be7d34cf2c5e6f776763afdacae3693615b9b2a6849e62338d`.

## Full Video-MME gate

The qualified c4 + 16K competition profile then ran the complete official
2,700-row set over all 900 videos with the same adapter contract: MiniCPM frame
packing, 96 frames, no audio or subtitle, greedy 128-token text output, and
concurrency four. Four preprocessing workers populated the persistent frame
cache before timing; all 2,700 request objects were constructed before the
initial endpoint test and three-request warmup.

| Metric | Result |
| --- | ---: |
| Accuracy | 1,897/2,700 = 70.259% |
| Organizer gate | >= 67.0% (pass by 3.259 points) |
| HTTP successes / failures | 2,700 / 0 |
| Parse failures | 1 |
| Duration | 9,379.122 s |
| Throughput | 0.2879 req/s |
| Mean / median / P99 TTFT | 12.589 / 13.192 / 17.041 s |
| Mean / median / P99 E2E | 13.890 / 14.360 / 19.561 s |

The result contains 2,700 unique request IDs and aligned input lengths, output
lengths, TTFTs, E2Es, generated texts, errors, and evaluation records. Its
error array is empty and the service remained healthy after the 2.6-hour timed
run. The detailed artifact is
`videomme-official-full-2700-c4-prefill16k.json` (SHA-256
`ade02dd9f01f91e6c8d3b4e650642e6b23c5e7ef35ded28c79831ea7f5a28b4c`).
The 32-row screen's 68.75% was conservative; the full result improved it by
1.509 percentage points.

## Competition status and next experiment

The accepted cache has passed the full 1,088-row Seed-TTS WER and official
speaker-SIM gate. The strict final c4 + 16K profile was rerun over all 1,088
English rows: serving completed without request or export failure in 1,640.460
seconds, versus 1,973.520 seconds for the prior 8K qualification (-16.87%).
Its performance metrics were 335.71 ms mean TTFT, 826.68 ms mean audio TTFP,
0.3406 whole-audio RTF, 0.3584 per-chunk RTF, 1.100 P99 per-chunk RTF, and
1,506.87 ms mean E2E. Relative to the prior full qualification, these improve
by 42.1%, 24.3%, 16.6%, 16.7%, 47.5%, and 16.9%, respectively.

Whisper Large v3 then evaluated the persisted 1,088-WAV export offline, with
zero ASR failures: mean WER was 0.03693 and median WER was 0.00000. The mean
change from the prior 0.03327 result is +0.366 percentage points, inside the
two-point accuracy-loss budget. Exact official WavLM-Large SV scoring covered
all 1,088 pairs and produced 0.029247 SIM, slightly above the prior 0.029029.
The serving-side evaluator initially failed because a partial Whisper load
published its processor global before model loading succeeded; initialization
is now transactional, and the saved WAV export allowed fail-closed rescoring
without repeating inference.

```text
c5dafd1f54ae7ac4517694aeb134d03390227ea61de998f9b14f521584d45997  talker-frequency-cache-quality-full-en-1088-prefill16k.json
58736dee065df32727ee201e07a52616b101bc32924aefb6f4c9e6969753efba  seed-tts-official-wer-full-en-1088-prefill16k.json
c034d610078c8f0ce7473eb720e0a31aca87ed0a60f9067948e121ab2373224  wav_res_ref_text.sim
```

Daily-Omni passed the full 1,197-row organizer gate at
78.279%, and Video-MME passed the full 2,700-row gate at 70.259%. All three
specified benchmark gates have therefore passed with complete evaluated counts
and zero HTTP failures. Thinker c10 and c8 were not promoted because their
full-run P99 E2E regressed 38.03% and 29.79%, respectively. c6 also remains
experimental: it
improved mean and P99 TTFT but regressed full-run P99 E2E by 24.24%. c5 reduced
that regression to 12.02% but still failed the same guard. The subsequent c4 +
16K prefill candidate passed the full gate and is now the accepted competition
profile, with +5.80% throughput, -5.51% mean E2E, -4.38% P99 TTFT, -0.87% P99
E2E, and identical aggregate accuracy versus the c4 + 8K control.

Further speed work must start from the post-cache trace rather than retry the
rejected fused Top-P, exponential-race sampler, or whole-stack CFM
preallocation candidates. The next candidate should target a narrower
fixed-shape DiT partition or Ascend-specific layout/cache kernel, and must beat
the accepted service in a fresh-process three-run A/B before repeating the
full quality qualification.

## Rejected DiT AdaLN modulation cache

The next narrow candidate cached every DiT block's AdaLN modulation for the
six immutable CFM timestep embeddings. It removed 96 repeated SiLU/linear
projections from each steady-state audio chunk on the fixed-width NPU MLP graph
path and preserved the generic eager fallback. Focused MiniCPM-o tests passed
47/47, and the candidate service confirmed graph compilation and replay.

The exact 32-prompt Seed-TTS protocol ran three times on the still-resident
pre-change service and three times after a candidate restart. All six runs
completed 32/32 with zero failure and 100% streaming continuity. One control
host/NPU excursion was retained; the predeclared comparison uses medians.

| Gate metric | Control median | AdaLN cache median | Change |
| --- | ---: | ---: | ---: |
| TTFT | 330.22 ms | 321.04 ms | -2.78% |
| Audio TTFP | 815.94 ms | 838.87 ms | +2.81% |
| Per-chunk RTF | 0.3566 | 0.3755 | +5.31% |
| P99 per-chunk RTF | 1.1062 | 1.1150 | +0.80% |
| E2E | 1,431.11 ms | 1,517.71 ms | +6.05% |

The candidate was rejected and reverted. Saving small invariant outputs added
memory reads and layout pressure on the already graph-replaying block path;
the TTFT improvement cannot compensate for material audio and E2E regressions.
This result closes AdaLN precomputation as a speed target on the current 910C
stack.

```text
0106389be776aaec372265949ddbe49a82d410223365c793927b254e55c4e162  adaln-cache-control-run1.json
59707c4a0d22fba03c9527ec62de9152d67405cf6c29d1897c1ca07b246a1e77  adaln-cache-control-run2.json
88e1885ec32ff0966139fef1b0a59e0526d338d63a763f84b89db86c875fb0ac  adaln-cache-control-run3.json
76787dbe68f1b76fbc8cb62a9c5d8afd723404dfeda0b21e11e4d9298a9e6be0  adaln-cache-candidate-run1.json
bb47ef1dd4a8dea61b626bca35dc7c4c4545df8a292ff5e547ba7d19fb140b10  adaln-cache-candidate-run2.json
d953eed8508c6d61f93f1b28d580d3d93262c906a089bf7dd6137bcc910f7d08  adaln-cache-candidate-run3.json
```

## Rejected Ascend evaluation LayerNorm kernel

The next layout-oriented screen replaced the two eager, affine-free DiT
normalizations per block with torch-npu's inference-only
`npu_layer_norm_eval`. Before paying for another full service A/B, the exact
steady-state DiT shape (`[2, 50, 512]`, bfloat16) was measured for 1,000
iterations on the same 910C. The fused operator was bit-identical to
`nn.LayerNorm`, but it was materially slower:

| Kernel | Mean | P50 | P99 |
| --- | ---: | ---: | ---: |
| Native `nn.LayerNorm` | 42.02 us | 40.87 us | 61.20 us |
| `npu_layer_norm_eval` | 82.27 us | 82.26 us | 100.32 us |

That is a 95.8% mean regression, and the installed torch-npu version also
reports `npu_layer_norm_eval` as deprecated. The candidate was therefore
rejected at the kernel gate and fully reverted. A same-era accepted-service
control had already completed three 32-prompt runs with zero failures and the
expected 4,801 input tokens, 480 output tokens, 3,329,280 frames, and 138.72
seconds of audio in every run; those artifacts remain useful as the next
candidate's control:

```text
e545c631531453c7a522b156b3c8a2d302fd8a0f71a2440df0124087cb5f1137  layout-norm-control-run1.json
3ba4123f02b7b0120f25955d8fcb2b99fce169578435846dced9d7c7a798b5b0  layout-norm-control-run2.json
e99e9229525efae035291aaf0ace2eddc056297f32e6b6bf8bb8c595e3323315  layout-norm-control-run3.json
```

The same exact-shape kernel gate also rejected two explicit layout changes.
Materializing the `[2, 320, 50] -> [2, 50, 320]` transpose took 40.97 us with
`npu_transpose` versus 23.60 us for the native view. Including the following
320-to-512 input projection, native view-plus-linear took 56.99 us while an
explicit contiguous input took 73.18 us (+28.4%) and changed bfloat16
rounding. A real FRACTAL_NZ projection weight was also slower: 66.89 us versus
51.02 us for ND (+31.1%). Neither candidate entered service A/B.

## Rejected estimator concat output workspace

A narrower allocation candidate reused one 64 KiB output tensor for the
fixed-shape `[2, 320, 50]` estimator input. Unlike four manual copies,
`torch.cat(..., out=workspace)` preserved the native concat kernel and was
bit-identical. The isolated exact-shape screen improved mean concat time from
29.35 us to 24.22 us (-17.5%) and P99 from 43.28 us to 37.96 us (-12.3%).
Focused tests passed 48/48 and the service log confirmed that the accepted DiT
MLP graph still compiled and replayed.

The exact same-era 32-prompt protocol then ran three times for both variants.
Every run completed 32/32 with zero failures and the identical 4,801 input
tokens, 480 output tokens, 3,329,280 audio frames, and 138.72 seconds of audio.

| Gate metric | Control median | Concat-out median | Change |
| --- | ---: | ---: | ---: |
| TTFT | 323.66 ms | 323.39 ms | -0.08% |
| Audio TTFP | 844.60 ms | 850.64 ms | +0.71% |
| Per-chunk RTF | 0.3796 | 0.3808 | +0.31% |
| P99 per-chunk RTF | 1.1488 | 1.1731 | +2.11% |
| E2E | 1,532.69 ms | 1,530.83 ms | -0.12% |

The candidate failed the audio gate and was reverted. The isolated allocation
win did not survive the full six-step CFM schedule; retaining one output
buffer across steps likely adds dependency/lifetime pressure that outweighs
the allocator saving.

```text
c5d00b67e81927ab69e75b995ab235075eb071d736f74f080a06bbd08546200d  estimator-cat-out-candidate-run1.json
498ea19c6b179e1aabf50719dd740778a3e2d38d1fb6c27778ad06b8fad5be03  estimator-cat-out-candidate-run2.json
dac2be12dfe37e0148eac7f78a125c4a18ce081434f822af213373bade75ce0d  estimator-cat-out-candidate-run3.json
```

Two wider fixed-shape TorchAir partitions were then screened. Compiling the
native transpose-plus-linear expression failed in the installed converter: it
dropped or misinterpreted the transpose and attempted a matrix multiply with
K=50 against K=320. An algebraically equivalent einsum compiled in 9.77
seconds, but replay took 176.03 us versus 48.32 us eager (3.64x slower) and
introduced `3.43e-5` maximum fp32 drift. Finally, a graph containing the full
LayerNorm-scale-shift AdaLN input expression was bit-identical but replayed in
167.95 us versus 74.91 us eager (2.24x slower). Both were rejected at the
kernel gate.

Together these screens show that the remaining Stage 2 layout and launch
overhead cannot be removed profitably with the installed high-level torch-npu
or TorchAir primitives. The next speed candidate must be a purpose-built CANN
kernel/fusion spanning a materially larger DiT attention or convolution
boundary, with its own operator-level parity tests; another Python-level
buffer, layout cast, or small graph partition is not justified by the 910C
measurements.

## Rejected DiT MLP static-kernel candidate

The next experiment kept the accepted fixed-shape DiT MLP TorchAir partition
but enabled CANN's ACLNN static-shape kernel compiler only for that partition.
The switch was fail-closed and opt-in through
`npu_dit_mlp_static_kernel`; it did not change the global Thinker or Talker
compiler configuration. Focused server-side tests passed 3/3. At startup,
TorchAir confirmed `Starting static kernel compilation`, generated and
installed a static-kernel run package for the fixed `[2, 50, 512]` partition,
and then logged the normal compiled-partition confirmation.

An eight-prompt Seed-TTS smoke completed 8/8 with valid audio and 100%
streaming continuity. The first warmup paid an 87-second one-time static
materialization cost; subsequent warmups and all measured requests were
steady-state. Three matched 32-prompt runs were then collected for the
resident accepted control and for the candidate. Every run completed 32/32
with zero failures and exactly 4,801 input tokens, 480 output tokens,
3,329,280 audio frames, and 138.72 seconds of generated audio.

| Gate metric | Control median | Static-kernel median | Change |
| --- | ---: | ---: | ---: |
| Serving duration | 47.845 s | 49.539 s | +3.54% |
| Request throughput | 0.6688 req/s | 0.6460 req/s | -3.42% |
| Mean TTFT | 328.13 ms | 324.75 ms | -1.03% |
| P99 TTFT | 464.75 ms | 455.90 ms | -1.91% |
| Mean audio TTFP | 835.08 ms | 856.83 ms | +2.60% |
| P99 audio TTFP | 973.53 ms | 989.61 ms | +1.65% |
| Mean per-chunk RTF | 0.3710 | 0.3831 | +3.25% |
| P99 per-chunk RTF | 1.1232 | 1.1499 | +2.38% |
| Mean E2E | 1,494.55 ms | 1,547.56 ms | +3.55% |
| P99 E2E | 2,055.39 ms | 2,111.73 ms | +2.74% |

Lower is better for latency and RTF; higher is better for throughput. The
candidate improved TTFT, but regressed every audio-path target and exceeded
the 2% guard on mean TTFP, mean and P99 per-chunk RTF, and mean and P99 E2E.
It was rejected and the accepted service restored. This result also closes
static-kernel compilation as a profitable optimization for this already small
MLP partition: removing launch overhead from ten block calls does not repay
the static package's runtime cost on the six-step CFM schedule.

```text
3f23330d91d8c7c64db68c72767bc2ad68ababd38c65bb654b44d8cd36a325db  static-kernel-control-run1.json
ad1a230e7c0bf70a5a7e03d23365b9bf3168924ce162701ece2f3f80d7f29833  static-kernel-control-run2.json
539eb793d2fbcdd6eee9d7c96dc09e09c371f200f90ec4f2753690904db9c6c4  static-kernel-control-run3.json
4c32786bb2e064a472a04db273aaf07e1eec1ca3cf5997d1814e8bb59e5face4  static-kernel-candidate-run1.json
6f3def5cc93106d970ed6ea98df04d23a00745a9cf8fba09ce812311ce661d6f  static-kernel-candidate-run2.json
a4eb95ca0fe8adeb519ef9ae292d745fd46d343117e7e4ea48980f80ce4ce8ae  static-kernel-candidate-run3.json
6346c2d5da567b4a621d5b04126e9c5f6388fa2aad5526c7e8dfa8f7547a80e4  static-kernel-smoke8.json
```

## Rejected virtual-concat input projection

The next candidate implemented a native Triton-Ascend kernel that treated the
four `[B, C, T]` estimator inputs as one virtual K dimension and performed the
following 320-to-512 projection without materializing `torch.cat`. The kernel
was wired behind an opt-in, fail-closed MiniCPM-o switch and preserved the
accepted eager path as its fallback.

The exact steady-state shape (`B=2`, `T=50`, `K=320`, `N=512`, bfloat16) was
screened on the same idle 910C service host. Both a 16-program Cube tiling and
a coarser two-program tiling were tested after warmup. The coarser result is
shown with two stock alternatives:

| Projection path | Mean | P50 | P99 | Max abs delta |
| --- | ---: | ---: | ---: |
| Native concat + view + linear | 94.14 us | 94.74 us | 110.80 us | 0 |
| Triton virtual concat, coarse | 927.89 us | 926.96 us | 951.61 us | 0.5 |
| Four split native linears + sum | 214.03 us | 212.25 us | 241.73 us | 0.5 |

The first, finer Triton tiling was also slow at 583.85 us mean and 580.43 us
P50. Torch-npu's exposed `npu_linear` operator could not replace the native
path because it accepts only a 2-D input on this installed stack. The custom
kernel was 9.86x slower than the native expression, and both ways of changing
the accumulation order introduced bfloat16 drift. The candidate was rejected
at the operator gate, removed before deployment, and did not consume a service
A/B run. This closes virtual concatenation at this small projection boundary;
a profitable custom operator must fuse a materially larger attention or
convolution region and amortize its launch and layout cost.

## Rejected attention cache-width bucketing

The next operator gate tested whether padding the variable attention-cache
width to a small fixed bucket could unlock a more reusable 910C graph or
kernel. The screen used the exact scaled-dot-product attention expression and
representative widths observed in the MiniCPM-o audio path. Each case was
warmed and measured for 600 iterations on an otherwise idle device; the
padded lanes were masked so they could not affect the result.

| Live cache width | Native width | Fixed bucket | Bucketed width | Change before copy |
| ---: | ---: | ---: | ---: | ---: |
| 302 | 93.826 us | 384 | 105.845 us | +12.81% |
| 302 | 93.826 us | 512 | 101.442 us | +8.12% |
| 352 | 92.568 us | 384 | 104.411 us | +12.79% |
| 352 | 92.568 us | 512 | 100.999 us | +9.11% |
| 402 | 90.114 us | 512 | 98.002 us | +8.75% |

Lower is better. Every fixed bucket was slower by 8.1% to 12.8% before
counting the cache-padding copy, and the attention call occurs roughly 600
times per synthesized chunk. The candidate was therefore rejected before a
service run. Cache-width bucketing remains useful only if a future fused CANN
kernel eliminates both the padding materialization and enough adjacent
attention work to repay the extra lanes.

## Accepted Stage-0 duplex foreground scheduling

The next end-to-end candidate targets interactive tail latency under mixed
load rather than single-request model execution. The duplex runtime now sends
established interactive requests with a configured negative priority, while
ordinary batch requests retain priority zero. A bounded aging rule temporarily
promotes one background request after 30 seconds, then restores its original
priority after admission. The production MiniCPM-o duplex profile enables
priority scheduling only on Stage 0; Talker and Code2Wav remain FCFS because
their chunk-transfer queues require deque operations that vLLM's priority heap
does not expose.

Focused tests on the 910C environment passed 20/20. The first all-stage
candidate correctly exposed that queue-interface mismatch when Stage 2 failed
on `PriorityRequestQueue.remove`; narrowing the policy to Stage 0 removed the
failure without widening shared vLLM queue semantics.

The concurrency screen launched six deterministic text-only requests with
`max_tokens=384` 250 ms before committing the duplex audio. Both runs used
`max_num_seqs=2`. The duplex input and result were identical at the behavioral
boundary: two Stage-0 output tokens, the `listen` decision, no client errors,
and no generated audio. All six background requests completed with HTTP 200,
and the candidate service remained healthy after the run.

| Duplex contention metric | FCFS control | Stage-0 priority | Change |
| --- | ---: | ---: | ---: |
| Stage-0 TTFT | 18,526.24 ms | 7,182.85 ms | -61.23% |
| Duplex model decision | `listen` | `listen` | unchanged |
| Successful background requests | 6/6 | 6/6 | unchanged |
| Service health after screen | healthy | healthy | unchanged |

Lower TTFT is better. The optimization removes 11.34 seconds of interactive
queue delay but cannot preempt the two requests already executing, so its
remaining approximately seven-second delay is expected. The candidate's
subsequent idle, steady-state duplex request completed with a 109.47 ms
Stage-0 TTFT and the same `listen` decision.

This is an admission-latency optimization: it does not change model kernels,
sampling parameters, or the accepted Seed-TTS/Daily-Omni/Video-MME serving
profile, so it neither claims nor changes their throughput and accuracy
numbers. A stricter matched cold-start FCFS replay was attempted, but the
control process remained blocked in the GlusterFS `lock_page` path after
checkpoint loading and was stopped without recording benchmark data. The
reported comparison therefore remains a real mixed-load screen, not a new
official competition score.

## Accepted Stage-0 duplex foreground preemption

Priority admission still left the interactive request behind both background
requests already occupying `max_num_seqs=2`. The duplex profile now opts into
a token-boundary preemption hook: when all Stage-0 slots are occupied and a
request at the configured foreground priority is waiting, the scheduler
selects one lower-priority running victim and delegates to vLLM's existing
preemption lifecycle. vLLM frees the victim's KV blocks and resumes it by
recomputation; the policy does not introduce a second KV state machine.

The first restart reproduced the host's GlusterFS `lock_page` stall. The exact
19 GiB model directory was therefore copied to the host's local overlay; the
model index and Code2Wav flow configuration checksums matched the source. All
three stages then initialized in 166.62 seconds, and the following service
screen used that local checkpoint without changing model files or request
semantics.

Three repetitions each launched six deterministic 384-token text requests,
then committed the same 4.16-second mono PCM16 16 kHz duplex input. Every
duplex request made the same two-token `listen` decision with no error or
generated audio. All 18 background requests completed with HTTP 200, the
scheduler log recorded exactly one lower-priority victim per repetition, and
the service remained healthy.

| Run | Stage-0 duplex TTFT | Slowest background request |
| --- | ---: | ---: |
| Preemption 1 | 1,202.33 ms | 22.09 s |
| Preemption 2 | 135.88 ms | 20.96 s |
| Preemption 3 | 143.28 ms | 20.96 s |
| Three-run median | 143.28 ms | 20.96 s |

Lower is better. The 143.28 ms median is 98.01% lower than the prior
priority-only 7,182.85 ms screen and 99.23% lower than the original FCFS
18,526.24 ms screen. Even the first resident-session materialization run was
83.26% lower than priority-only admission. The background guard remains
bounded: the slowest request stayed near the three-wave, two-slot completion
envelope rather than starving, and bounded aging is still enabled.

The full focused scheduler and duplex-deployment suite passes 114/114 in the
910C environment. Because this changes only Stage-0 mixed-load admission and
uses the existing recompute path, it does not change ordinary competition
requests, model kernels, sampling, or the previously accepted Daily-Omni,
Seed-TTS, and Video-MME accuracy results.

## Accepted native-duplex control-token embedding cache

MiniCPM-o's persistent Stage-0 append path repeatedly injects the same small
set of unit and boundary tokens. The existing opt-in cache keeps those token
embeddings resident on the model device and returns the immutable tensor on
subsequent appends. It is now exposed as the typed duplex setting
`cache_control_embeddings` and enabled in the static-weight MiniCPM-o duplex
profile. The default remains false, and deployments with dynamic LoRA or other
embedding-weight mutation must not enable it.

An exact-shape 910C operator screen used the checkpoint vocabulary and hidden
dimensions (`151748 x 4096`, bfloat16) for 1,000 iterations after warmup:

| Operation | Native mean | Cached mean | Change | Max abs delta |
| --- | ---: | ---: | ---: | ---: |
| One control-token embedding | 35.36 us | 18.43 us | -47.88% | 0 |
| Three control embeddings plus concat | 95.95 us | 37.08 us | -61.35% | 0 |

The full duplex service then completed one warmup and three resident requests.
The warmup request paid graph/session materialization and is excluded. All
three measured requests returned `ok=true`, the same `listen` decision, and no
errors. Their Stage-0 TTFT values were 104.85, 97.09, and 98.71 ms: 100.21 ms
mean and 98.71 ms median. Against the prior resident no-cache screen at
109.47 ms, that is 8.45% lower mean and 9.83% lower median TTFT. The service
remained healthy after the runs.

This cache applies only to the native-duplex Stage-0 input builder. It cannot
change the ordinary competition profile, sampling, Daily-Omni accuracy,
Seed-TTS audio quality, or Video-MME results. Its promotion gate is therefore
exact tensor parity plus native-duplex behavioral and latency stability rather
than a rerun of unrelated ordinary-request suites.

## Rejected raw-tensor SHM, retained event notification

The accepted 910C profiles had enabled the `tensor-v1` shared-memory format
before a target-host connector proof existed. An exact connector A/B now
compares it with the ordinary msgpack-plus-SHM path. Each iteration created,
wrote, read, cloned, and unlinked a real POSIX segment. The steady payload
matched a 25-frame codec chunk plus three left-context codes; the first-chunk
case additionally carried six seconds (96,000 float32 samples) of reference
audio.

| Payload | Connector | Mean round trip | P50 | P99 | Segment bytes |
| --- | --- | ---: | ---: | ---: | ---: |
| Steady codec chunk | Serialized SHM | 279.54 us | 272.77 us | 359.00 us | 540 |
| Steady codec chunk | `tensor-v1` | 370.17 us | 356.81 us | 472.71 us | 705 |
| First chunk + reference | Serialized SHM | 1,076.55 us | 1,071.38 us | 1,170.70 us | 384,593 |
| First chunk + reference | `tensor-v1` | 1,105.09 us | 1,098.73 us | 1,219.78 us | 384,769 |

Lower is better. Raw tensors regressed the steady round trip 32.42% and the
large first-chunk round trip 2.65%, while increasing the segment size in both
cases. All MiniCPM-o 910C profiles now keep `raw_tensor_shm: false`. The
generic implementation and its tests remain available for payloads where a
future measurement shows a win.

Event notification was screened separately for 1,000 real segment transfers.
An immediate same-thread put/get control measured 163.97 us mean and 286.68 us
P99; Unix-datagram notification plus put/get measured 184.60 us mean and
272.42 us P99. The 20.63 us mean notification cost buys a 14.26 us lower P99
and, more importantly, replaces the receiver's production fallback of up to
one millisecond between failed reads. Notifications therefore remain enabled
for tail wakeup and idle-CPU behavior; only the raw serialization format is
rejected.

## Rejected post-cache kernel candidates

Several additional 910C operator screens were completed before changing the
stage topology. Each used the exact steady Code2Wav shapes and was rejected
before promotion when its win did not survive the relevant higher-level gate.

- Replacing the accepted TorchAir MLP partition with eager
  `torch_npu.npu_ffn` took 154.85 us, and graphing that operator took 162.47 us,
  versus 149.27 us for the accepted graph.
- Packing the three attention projections reduced isolated eager preparation
  from 119.40 us to 92.52 us. In the full three-run service A/B, however, mean
  TTFP regressed 0.38%, P99 TTFP regressed 0.82%, and P99 chunk RTF regressed
  1.55%. The candidate was reverted.
- Direct `npu_fusion_attention` replay was slower than native SDPA at every
  screened live width. Queue-amortized examples were 31.10 versus 37.48 us at
  width 202 and 35.32 versus 40.99 us at width 502. The two exposed inference
  attention-score variants were slower again at roughly 110--130 us.
- A Triton-Ascend kernel fusing packed-QKV splitting, Q/K affine norms, layout,
  and cache append preserved Q/V exactly and had a 1.53e-5 maximum K delta. It
  improved the original preparation path by 5--12%, but remained 7--13% slower
  than the already-rejected packed eager path.
- FRACTAL_Z reduced an isolated convolution from 64.52 to 27.26 us, but the
  complete two-convolution block regressed from 225.07 to 253.80 us because
  its consumers paid the format conversion. FRACTAL_NZ linear weights were
  also neutral to slower.

The installed torch-npu emits `npu_fusion_attention_v3`, while its bundled
TorchAir lacks that converter. An inference-only converter prototype mapped
the dropout-free BNSD call to GE `FlashAttentionScore`; an exact static full
attention partition then replayed in 123.91 us versus 254.60 us eager with
bit-identical outputs. It could not be used directly in serving: Seed-TTS
reference prompts produce variable initial cache lengths, so none of the four
warmed static lengths replayed. A dynamic graph specialized and recompiled for
every new length (4--29 seconds per shape), replayed in about 699 us versus
265 us eager, and produced NaNs. A fixed-capacity scatter-plus-mask graph was
also rejected: it took 1.72--1.86 ms versus 0.26--0.32 ms eager and introduced
nontrivial output/cache drift. This closes high-level attention graphing on the
installed stack; the 51% static-shape result remains useful evidence for a
future purpose-built CANN operator with a native variable-length interface.

## Accepted Thinker/Talker co-location and device-baseline fix

The accepted two-NPU layout placed Talker and Code2Wav together on NPU 1 even
though Thinker usually finishes the short TTS text response before most codec
and audio work. The new competition layout places Thinker and Talker on NPU 0
and gives Code2Wav exclusive use of NPU 1. Thinker memory utilization is 0.72
and Talker is 0.08. Startup measured 71,808 Thinker KV tokens and 71,040 Talker
KV tokens, covering the qualified c4/16K and c4/4K envelopes.

The first launch exposed an independent parallel-initialization race. When the
parent had no NPU visibility restriction, the runtime stored the baseline as
`None`; the resolver interpreted that as "read the current environment" after
the parallel diffusion initializer had temporarily selected NPU 1. Talker's
requested NPU 0 was consequently remapped onto NPU 1. The runtime now records
an unrestricted baseline explicitly as an empty string, and the resolver
preserves the physical IDs from the deployment. The corrected log reports
Stages 0/1 on NPU 0 and Stage 2 on NPU 1. The focused device-resolution and
stage-initialization suite passes 45/45.

Both variants were launched as fresh processes from the same source tree and
ran the exact 32-prompt Seed-TTS protocol three times. Every run completed
32/32 with zero failures, 100% streaming continuity, 4,801 input tokens, 480
output tokens, 3,329,280 frames, and 138.72 seconds of audio. The promotion
comparison uses the three-run median.

| Gate metric | Fresh control | Co-located Talker | Change |
| --- | ---: | ---: | ---: |
| Serving duration | 51.983 s | 47.327 s | -8.96% |
| Request throughput | 0.6156 req/s | 0.6762 req/s | +9.84% |
| Median TTFT | 333.54 ms | 321.25 ms | -3.68% |
| P99 TTFT | 460.37 ms | 459.56 ms | -0.18% |
| Mean audio TTFP | 858.67 ms | 849.54 ms | -1.06% |
| P99 audio TTFP | 983.39 ms | 970.25 ms | -1.34% |
| Mean per-chunk RTF | 0.3993 | 0.3692 | -7.55% |
| P99 per-chunk RTF | 1.1149 | 1.1195 | +0.42% |
| Mean E2E | 1,624.04 ms | 1,478.50 ms | -8.96% |
| P99 E2E | 2,193.40 ms | 1,902.56 ms | -13.26% |

Lower is better except for throughput. The sole regression is the 0.42% P99
chunk-RTF movement, inside the 2% performance guard. Stage placement and KV
capacity do not change weights, kernels, sampling, or request content, and the
identical structural signature confirms that the existing full Seed-TTS,
Daily-Omni, and Video-MME accuracy qualifications carry forward.

```text
104dd9236daea0659694cc1243d13c8c5b48c60da1d7ace24a8cccabbc2cecb7  fresh-control-run1.json
e82b009f8b840459d846d7a8c44a8e01ca9a75db1c8102ccbe60b222eef4f1ff  fresh-control-run2.json
df5406b0a4e8453dfd11ca7543d51fba7f891ae80da944727f559856bf29336d  fresh-control-run3.json
a87bdcce96ad1745f6617024712e1d6a5484a3467640f42080b311189cf34776  talker0-candidate-run1.json
b2ff698a7e0a6760eea4a0bd4820056e55bd0c751de881f827f33f80489ac3e1  talker0-candidate-run2.json
5329f607c9aa8b234f02fd13330fd71d62fae7bca9d480561f1f9c67c3f42ae9  talker0-candidate-run3.json
```

## Rejected single-request Code2Wav cache-state packing

The next Code2Wav candidate targeted the batch-of-one path used by the
concurrency-one competition workload. The control rebuilt batched flow and
HiFT cache tensors with seven `torch.cat` operations for every streamed chunk,
even when only one request was active. Two progressively narrower variants
were tested against a fresh control on the accepted Thinker/Talker co-located
CFM6 profile.

The first variant skipped both the input cats and the compact output-cache
clones. It was rejected immediately: retaining detached views kept their
larger backing tensors alive and increased allocator/lifetime pressure. The
three-run median serving duration and mean E2E both regressed by about 3.5%,
and mean audio TTFP regressed by 2.03%.

The second variant removed only the seven redundant input cats and preserved
the existing compact output clones. It passed an exact two-chunk state and
audio parity test and completed all three service runs, but its performance
result was mixed:

| Three-run median metric | Fresh control | Input-only candidate | Change | Direction |
| --- | ---: | ---: | ---: | --- |
| Serving duration | 46.416 s | 46.511 s | +0.20% | slower |
| Request throughput | 0.6894 req/s | 0.6880 req/s | -0.20% | slower |
| Mean E2E | 1,449.92 ms | 1,453.04 ms | +0.22% | slower |
| Median E2E | 1,472.86 ms | 1,469.72 ms | -0.21% | faster |
| P99 E2E | 1,969.68 ms | 1,917.62 ms | -2.64% | faster |
| Mean TTFT | 324.35 ms | 318.17 ms | -1.91% | faster |
| Median TTFT | 325.78 ms | 319.48 ms | -1.93% | faster |
| P99 TTFT | 458.48 ms | 463.08 ms | +1.00% | slower |
| Mean audio TTFP | 836.42 ms | 832.45 ms | -0.47% | faster |
| P99 audio TTFP | 998.92 ms | 980.67 ms | -1.83% | faster |
| Mean whole-audio RTF | 0.3418 | 0.3421 | +0.09% | slower |
| Median whole-audio RTF | 0.3325 | 0.3365 | +1.19% | slower |
| P99 whole-audio RTF | 0.4162 | 0.4118 | -1.05% | faster |

Lower is better except for throughput. Every run completed 32/32 requests
with zero failures and 100% streaming continuity, and produced the identical
4,801 input tokens, 480 output tokens, 3,329,280 audio frames, and 138.72
seconds of audio. The saved client artifacts did not contain usable
per-chunk-RTF samples (`null` for control and zero for the candidate), so that
metric was not used to promote the change.

The narrowed candidate's modest TTFT and tail-TTFP wins did not survive as a
throughput, mean-E2E, or whole-audio-RTF improvement. Both variants were
therefore removed from the runtime and deployment profile. This negative
result also sets the next Code2Wav boundary: avoid Python-side micro-caches
whose savings are below run-to-run service noise, and concentrate on measured
NPU graph/operator work or larger stage scheduling changes.

Raw results are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-single-state-20260812/results
```

Result checksums:

```text
888295934fedcb084985679912c22f8fb825b71658db4d9a1a9a9c43ff1708d3  control-run1.json
e2546312ae653137380b32a30f4d066268b53460380bb68d8f372fe92d6edef2  control-run2.json
7b4be878c00dad9a92aed0b54c764bd0c948d979d179feb71bd1423bd84c1cd8  control-run3.json
697e7e0bec5a52cf758430d74a3eb6f2bf3deeca56a04ea4544111c39186cbbf  candidate-run1.json
1e153f3a927754d9bdca17c035cd340ea3a50ea857dc96c34680fc8d8bb24470  candidate-run2.json
997493ac54045f1f143264cceb340f3f0fa1518c7d55818b56eee574c910f6a0  candidate-run3.json
35731529c63e48c4ed8063529038d8926b078515579d94f525d7555c591dfc47  input-only-run1.json
a2ca2a3f0b92d57f9287ebc787ee10af3219fe27b4d39714ad5a437dbd6de9b8  input-only-run2.json
ba5ef71705eff7c2e31d9c1fceb5279c8c7dfdd02ede290b9fc3c64e8efb7cb9  input-only-run3.json
```

## Rejected early Code2Wav prompt prewarm

This candidate tried to overlap Seed-TTS reference-audio prompt preparation
on NPU 1 with Talker's accumulation of its first 25 codec tokens on NPU 0.
The stage processor emitted one control-only payload as soon as the first
codec token arrived. Code2Wav prepared and cached the reference features but
did not create streaming state or run CFM/HiFT; the first real audio chunk
kept sequence number zero and followed the unchanged generation path.

The opt-in implementation passed 74 focused tests, including prompt-cache
reuse, first-chunk ordering, cleanup, and invalid-switch coverage. It was then
run three times against the same fresh control used for the narrowed
cache-packing experiment. Every candidate run completed 32/32 requests with
zero failures and produced the same 4,801 input tokens, 480 output tokens,
3,329,280 audio frames, and 138.72 seconds of audio.

| Three-run median metric | Fresh control | Prompt-prewarm candidate | Change | Direction |
| --- | ---: | ---: | ---: | --- |
| Serving duration | 46.416 s | 47.770 s | +2.92% | slower |
| Request throughput | 0.6894 req/s | 0.6699 req/s | -2.83% | slower |
| Mean E2E | 1,449.92 ms | 1,492.28 ms | +2.92% | slower |
| Median E2E | 1,472.86 ms | 1,486.04 ms | +0.90% | slower |
| P99 E2E | 1,969.68 ms | 2,002.92 ms | +1.69% | slower |
| Mean TTFT | 324.35 ms | 326.49 ms | +0.66% | slower |
| Median TTFT | 325.78 ms | 329.87 ms | +1.26% | slower |
| P99 TTFT | 458.48 ms | 468.08 ms | +2.09% | slower |
| Mean audio TTFP | 836.42 ms | 852.87 ms | +1.97% | slower |
| Median audio TTFP | 840.65 ms | 838.00 ms | -0.32% | faster |
| P99 audio TTFP | 998.92 ms | 991.09 ms | -0.78% | faster |
| Mean whole-audio RTF | 0.3418 | 0.3512 | +2.75% | slower |
| Median whole-audio RTF | 0.3325 | 0.3413 | +2.64% | slower |
| P99 whole-audio RTF | 0.4162 | 0.4272 | +2.65% | slower |

Lower is better except for throughput. The small median/tail TTFP movements
did not compensate for regressions in duration, throughput, mean TTFP, E2E,
TTFT tail, or whole-audio RTF. The additional cross-stage scheduling and IPC
work costs more than the prompt preparation it overlaps at concurrency one.
The candidate was therefore removed from both source and deployment state.

Raw results are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-prompt-prewarm-20260812/results
```

Result checksums:

```text
aaa02b02a5a0c1a87ffee2cb182f5c38893dd1cd9619fedde655ab9cb8629227  prompt-prewarm-run1.json
d529dc0388f18e230aa5626f18d6484202ed2d90dc73ffa387e5c5f2cdb7ac0d  prompt-prewarm-run2.json
43f7a57234642eddee62e7abd34b5624046586ab41b90c293cebe372d64f9f70  prompt-prewarm-run3.json
```

## Accepted prompt speaker-projection cache

The next candidate removed repeated NPU work without changing stage traffic.
MiniCPM-o previously normalized the immutable reference-speaker embedding and
ran `spk_embed_affine_layer` during prompt setup and again for every streamed
25-frame codec chunk. The prompt cache now stores that projected embedding
beside the existing speech tokens and mel features, expands it by batch, and
evicts it through the existing prompt lifecycle.

The implementation preserves the original autocast context and mathematical
order for each row. It changes no model weight, CFM step, random input,
sampling operation, codec token, or HiFT operation. The focused Code2Wav suite
passes 46/46, including an explicit assertion that two streamed chunks do not
repeat the projection.

Three fresh-process candidate runs used the same 32 fixed English prompts,
three warmups, concurrency one, CFM6 profile, and nested TTS request body as
the immediately preceding fresh control. Every run completed 32/32 with zero
failure and 100% streaming continuity, and produced the identical 4,801 input
tokens, 480 output tokens, 3,329,280 audio frames, and 138.72 seconds of audio.

| Three-run median metric | Fresh control | Speaker-cache candidate | Change |
| --- | ---: | ---: | ---: |
| Serving duration | 46.416 s | 45.954 s | -1.00% |
| Request throughput | 0.6894 req/s | 0.6963 req/s | +1.01% |
| Mean E2E | 1,449.92 ms | 1,435.69 ms | -0.98% |
| Median E2E | 1,472.86 ms | 1,444.23 ms | -1.94% |
| P99 E2E | 1,969.68 ms | 1,910.92 ms | -2.98% |
| Mean TTFT | 324.35 ms | 318.52 ms | -1.80% |
| Median TTFT | 325.78 ms | 324.49 ms | -0.39% |
| P99 TTFT | 458.48 ms | 452.63 ms | -1.28% |
| Mean audio TTFP | 836.42 ms | 828.27 ms | -0.98% |
| Median audio TTFP | 840.65 ms | 834.45 ms | -0.74% |
| P99 audio TTFP | 998.92 ms | 970.07 ms | -2.89% |
| Mean whole-audio RTF | 0.3418 | 0.3383 | -1.02% |
| Median whole-audio RTF | 0.3325 | 0.3314 | -0.35% |
| P99 whole-audio RTF | 0.4162 | 0.4126 | -0.86% |

Lower is better except for throughput. Every measured gate moves in the
desired direction. The fresh control artifacts did not expose usable
per-chunk-RTF samples, so those values were not used in the matched promotion
decision; the candidate runs themselves were stable at 0.3576--0.3671 mean
and 1.0968--1.1085 P99 per-chunk RTF.

Because the output path and structural signature are unchanged, the accepted
full Seed-TTS, Daily-Omni, and Video-MME qualifications carry forward. This
candidate is promoted as an always-on prompt-cache optimization rather than a
deployment switch.

Raw results are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-speaker-projection-cache-20260812/results
```

Result checksums:

```text
c62724cc2961ef621a3f823967ae934e7069bb0c2c9f4bbcf790f09b1cacb426  speaker-cache-run1.json
7b5adf0fa934dcd4564d6fafd1cf1f9e4b60176eee7940722690bc826efd8814  speaker-cache-run2.json
477b47c9365c8ba2744118f903ac54f3dd7e5431e23d823e4f032669c23ffdd2  speaker-cache-run3.json
```

## Opt-in HiFT inference weight-norm materialization

After the previous environment became unavailable, this candidate was
reproduced on `DevEnv_132987`, an Atlas A3-class host exposing two logical
64-GiB Ascend 910C devices. The model was copied from shared storage to the
host-local overlay before measurement. The service used the qualified CFM6,
DiT-MLP-graph competition profile with the existing automatic SDPA adapter.

HiFT applies PyTorch weight-normalization parametrizations to its convolution
stack. In immutable inference those parametrizations recompute normalized
weights on every access. With the following opt-in switch, the NPU adapter
materializes the effective weights once after checkpoint loading and removes
only standard `_WeightNorm` parametrizations:

```bash
VLLM_OMNI_MINICPMO45_NPU_SDPA_BACKEND=auto \
VLLM_OMNI_MINICPMO45_NPU_HIFT_MATERIALIZE_WEIGHT_NORM=1 \
vllm serve /models/OpenBMB/MiniCPM-o-4_5 --omni \
  --deploy-config vllm_omni/deploy/minicpmo_4_5_2npu_910c_cfm6_dit_mlp_graph_competition.yaml \
  --trust-remote-code --interleave-mm-strings --host 127.0.0.1 --port 8099
```

Startup reported 82 materialized parametrizations. The transformation is
idempotent, leaves unrelated parametrizations intact, and preserves exact CPU
module output in its focused test. The complete focused NPU patch suite passed
18/18 on the host.

Control and candidate each ran three times with the same 32 fixed English
Seed-TTS prompts, three warmups, concurrency one, seed zero, nested MiniCPM-o
TTS request body, and CFM6 competition profile. Every run completed 32/32 with
zero failures and 100% streaming continuity and produced 4,801 input tokens,
480 output tokens, 3,362,880 frames, and 140.12 seconds of audio.

| Three-run median metric | Control | Materialized weight norm | Change |
| --- | ---: | ---: | ---: |
| Serving duration | 47.675 s | 46.766 s | -1.91% |
| Request throughput | 0.6712 req/s | 0.6843 req/s | +1.94% |
| Mean E2E | 1,489.43 ms | 1,461.05 ms | -1.91% |
| Median E2E | 1,512.57 ms | 1,497.82 ms | -0.98% |
| P99 E2E | 2,016.09 ms | 1,970.98 ms | -2.24% |
| Mean TTFT | 316.17 ms | 313.21 ms | -0.94% |
| Median TTFT | 320.19 ms | 317.54 ms | -0.83% |
| P99 TTFT | 450.17 ms | 449.12 ms | -0.23% |
| Mean audio TTFP | 846.63 ms | 835.21 ms | -1.35% |
| Median audio TTFP | 844.89 ms | 835.75 ms | -1.08% |
| P99 audio TTFP | 997.60 ms | 976.99 ms | -2.07% |
| Mean whole-audio RTF | 0.3479 | 0.3410 | -1.97% |
| Median whole-audio RTF | 0.3448 | 0.3381 | -1.93% |
| P99 whole-audio RTF | 0.4713 | 0.4540 | -3.67% |
| Mean per-chunk RTF | 0.3639 | 0.3565 | -2.02% |
| Median per-chunk RTF | 0.1765 | 0.1733 | -1.80% |
| P99 per-chunk RTF | 1.1416 | 1.1196 | -1.93% |

Lower is better except for throughput. All measured speed gates moved in the
desired direction, including the competition's per-chunk RTF, TTFT, and TTFP
targets.

A paired eight-row English Seed-TTS screen then used the same prompt order,
temperature-zero generation, and the exact nested TTS request body. Both
variants completed 8/8 with WER 0.0000, 1,197 input tokens, 116 output tokens,
31.84 seconds/764,160 frames of output, and no evaluation errors. The in-tree
WavLM Base Plus mean-pooling proxy moved from 0.838845 to 0.839166
(+0.000321); its median moved from 0.847123 to 0.848490. This proxy is not the
official fine-tuned Seed-TTS speaker-verification model, so it is an aligned
regression screen rather than a replacement for the organizer's full gate.
The candidate stays opt-in until full Seed-TTS, Daily-Omni, and Video-MME
qualification is repeated on this environment.

Raw results are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-hift-materialize-20260813/results
```

Result checksums:

```text
82a706504b06b68a82d954e71b0c5eddfa240e9a35ceae7d860a78ec57d0fb50  control-run1.json
658ca0449a3265a36ce9b2d2dc1df3d402ad8d9cf9e239882817f4a848eaedae  control-run2.json
48f4413b1a31d44dc7066e7f31363ed48472aa68daf40fa636edd184c39e4ef6  control-run3.json
e88e6661604eaaa5eeb42514474053f065de1f0132ae91b0f9a86663e2cc1798  candidate-run1.json
ffe92e082d415e9cbe28942c286cec95b0dc1078be56be770830e7409c9a81ef  candidate-run2.json
b6776685bc31b20c170b3daa9fb70cfe05feca0ceda8743fd7fae527ad2cc145  candidate-run3.json
4add22514e235a6b8b82fa2d1f3b59092e3d60ff2cd60bc859f1226380024611  control-quality-en8.json
dcb3021833e7d59ecd7925c8aa1ec75627a149b3c4661aafcfbe764105474d61  candidate-quality-en8.json
```

## Fixed-width DiT preamble and Conv+MLP megagraph

This major candidate widens the model-specific 910C graph boundary instead of
wrapping another individual operator. It adds an exact attention preamble
(AdaLN, normalization, Q/K/V projection, Q/K normalization) and a Conv+MLP
megagraph (both causal Conv1D operations and cache updates, normalization,
activation, gated residual, and the full MLP residual). The opt-in profile is
`minicpmo_4_5_2npu_910c_cfm6_dit_megagraph_competition.yaml`.

This is a TorchAir megagraph rather than a hand-written monolithic AscendC
kernel. A direct CANN `super_kernel_optimize` prototype was bit-exact but
regressed from 148.6 us to 337.6 us. Packed QKV tied/regressed, while a
post-attention graph had 0.03125 maximum drift and no stable win. At the fixed
`[2, 50, 512]` BF16 shape, the selected preamble was exact and improved from
253.7 us to 176.2 us (-30.6%). Combined Conv+MLP measured about 300 us versus
348-362 us for separate graphs (-14% to -17%). Its cache was exact; hidden
output had 0.00048828125 maximum and 1.16e-5 mean absolute BF16 drift.

The resident service comparison used the same 32 Seed-TTS English rows,
temperature zero, seed 42, three warmups, and concurrency one. All variants
completed 32/32 with zero failures and 100% continuity. Control and megagraph
produced exactly 2,649 text tokens, 298.92 seconds, and 7,174,080 audio frames.
The same 16th row exhibited the known long model-generation tail in both.

| Metric (lower is better) | Accepted control | Preamble only | Megagraph | Megagraph vs control |
| --- | ---: | ---: | ---: | ---: |
| Serving duration | 116.244 s | 114.738 s | 112.602 s | -3.13% |
| Mean TTFT | 1,562.85 ms | 1,542.31 ms | 1,550.31 ms | -0.80% |
| Median TTFT | 392.69 ms | 380.58 ms | 394.76 ms | +0.53% |
| P99 TTFT | 26,160.89 ms | 25,941.11 ms | 25,932.31 ms | -0.87% |
| Mean audio TTFP | 2,122.02 ms | 2,097.98 ms | 2,082.10 ms | -1.88% |
| Median audio TTFP | 904.42 ms | 893.47 ms | 888.51 ms | -1.76% |
| P99 audio TTFP | 27,498.24 ms | 27,287.06 ms | 27,265.11 ms | -0.85% |
| Mean whole-audio RTF | 0.29780 | 0.29809 | 0.28466 | -4.41% |
| Median whole-audio RTF | 0.28611 | 0.28610 | 0.27487 | -3.93% |
| P99 whole-audio RTF | 0.56306 | 0.54743 | 0.54586 | -3.06% |
| Mean per-chunk RTF | 0.42500 | 0.42039 | 0.41290 | -2.85% |
| Median per-chunk RTF | 0.18523 | 0.17509 | 0.17983 | -2.92% |
| P99 per-chunk RTF | 1.21962 | 1.19344 | 1.17427 | -3.72% |

The service log confirmed compile and live replay of all three Stage-2
partitions. Focused production tests passed 25/25 on the host.

The paired quality screen used the same current-manifest seed-42 rows. It is a
hard sample, so only its paired delta is meaningful. Both variants completed
8/8, generated the same 151 text tokens and 53.80 seconds of audio, and had no
ASR or embedding failures.

| Paired EN8 metric | Accepted control | Megagraph | Change |
| --- | ---: | ---: | ---: |
| Mean WER (lower) | 0.565030 | 0.565030 | 0.000000 |
| Median WER (lower) | 0.537500 | 0.537500 | 0.000000 |
| Mean WavLM proxy SIM (higher) | 0.803156 | 0.803233 | +0.000077 |
| Median WavLM proxy SIM (higher) | 0.818616 | 0.818385 | -0.000231 |

An attempted seed-zero rerun was invalidated and stopped: the current manifest
order did not reproduce the historical eight rows and selected a runaway
2,169-token item. It is excluded. The candidate remains opt-in until the full
1,088-row Seed-TTS and full Daily-Omni and Video-MME gates are repeated.

Raw results:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-dit-preamble-20260813/results
```

```text
52665bd691f82bdd13f1823d9ce9f450c59b7ba94cbf2dda08e10f68ee0738cd  control-run1.json
a9d6f775c1cfe689e3e5bbd78bd97485de6d3798d56f9161bda9b388aa60e34e  candidate-run1.json
ec80c8970bd58c541725afd3aef7789cfd897716142d33160ae5605152b2700f  megagraph-run1.json
7a088c9a9d5744d761203d9aef176f597bb1d0c006267dca78f9b30da85c289b  control-quality-en8-seed42.json
3fa82e102bdfef3b6607bfeecf47a22754709c10b37b46227757b71d0eab6b40  megagraph-quality-en8.json
```

## Native 910C causal-Conv pack kernel

The next megagraph revision replaces each fixed `[2, 50, 512]` causal-Conv
layout sequence (`transpose + cache concat + cache slice`) with a native
AscendC AIV operator. The operator emits the three-frame, tap-major matrix
consumed by a Cube `Linear` and returns the next two-frame cache in the same
launch. A TorchAir AscendIR converter keeps this boundary inside the existing
Conv+MLP graph; both Conv1D weights are prepacked once during startup.

The implementation was compiled with CANN 9.0 for `ascend910_93` and executed
on `DevEnv_132987` (Atlas A3 / Ascend 910C). FP16, FP32, and BF16 all pass the
direct NPU correctness test: all 153,600 packed elements and all 2,048 cache
elements match the PyTorch reference bit-for-bit. The FP32 TorchAir graph used
by the live Code2Wav stage also compiled and returned zero maximum error for
both outputs. Seven repetitions of 300 iterations measured the single
Conv/cache boundary for FP16 and BF16:

| Dtype | Standard Conv/cache | Native pack + Linear | Speedup | Latency change |
| --- | ---: | ---: | ---: | ---: |
| FP16 | 0.08375 ms | 0.05963 ms | 1.405x | -28.80% |
| BF16 | 0.08480 ms | 0.06049 ms | 1.402x | -28.68% |

The more representative compiled BF16 partition includes both causal
convolutions, cache updates, normalization, Mish, gated residual, and MLP.
Seven repetitions of 100 graph replays improved median latency from 0.21419 ms
to 0.20269 ms: 1.057x, or 5.37% lower. Cache output remained bit-exact. The
hidden output had maximum absolute error 0.03125 and mean absolute error
0.00001360 from the equivalent GEMM accumulation order.

The resident service then compared the opt-in fused-pack profile with the same
profile with only `VLLM_OMNI_MINICPMO45_NPU_DIT_FUSED_CONV_PACK=0`. Each
variant ran twice with 32 fixed English Seed-TTS requests, three warmups,
concurrency one, seed zero, temperature zero, and the CFM6 competition
profile. The candidate log contains the required live marker:

```text
Compiled MiniCPM-o NPU DiT Conv+MLP megagraph for 2x50x512
```

There was no fused-path fallback. All four runs completed 32/32 with zero
failures and 100% streaming continuity. Every run produced 4,801 input
tokens, 480 output tokens, 3,362,880 audio frames, and 140.12 seconds of audio.

| Two-run mean metric | Control | Native fused pack | Change |
| --- | ---: | ---: | ---: |
| Serving duration | 44.928 s | 41.786 s | -6.99% |
| Request throughput | 0.7123 req/s | 0.7658 req/s | +7.51% |
| Mean E2E | 1,403.59 ms | 1,305.25 ms | -7.01% |
| Median E2E | 1,415.14 ms | 1,318.58 ms | -6.82% |
| P99 E2E | 1,962.05 ms | 1,768.28 ms | -9.88% |
| Mean TTFT | 320.44 ms | 318.27 ms | -0.68% |
| Median TTFT | 323.01 ms | 322.76 ms | -0.08% |
| P99 TTFT | 462.15 ms | 456.31 ms | -1.26% |
| Mean audio TTFP | 812.00 ms | 776.10 ms | -4.42% |
| Median audio TTFP | 809.86 ms | 775.14 ms | -4.29% |
| P99 audio TTFP | 977.86 ms | 927.21 ms | -5.18% |
| Mean per-chunk RTF | 0.34609 | 0.32315 | -6.63% |
| Median per-chunk RTF | 0.17689 | 0.13709 | -22.50% |
| P99 per-chunk RTF | 1.12229 | 1.07311 | -4.38% |
| Audio throughput | 3.119x | 3.353x | +7.51% |

Lower is better except for throughput. All measured competition-facing speed
metrics moved in the desired direction. The matching output totals and exact
FP32 operator/graph tests are strong semantic regression screens, but they do
not replace the organizer's full WER/SIM and multimodal accuracy gates. The
profile remains opt-in until full Seed-TTS, Daily-Omni, and Video-MME
qualification confirms no more than a two-point decline.

Raw results are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-fused-conv-20260814/results
```

Result checksums:

```text
a04d5892cb2e27f5d7208d672e7e9a223580f43282cf378341a92c4a7140e436  control-run1.json
15c61548ebd1fefd54046c70c39ef729b6246db2407c3de2e2316cb423e53a72  control-run2.json
e992bad2d94af35119103e6ff6a14baaae0b413fcee56249cdb7ea972c056f03  fused-run1.json
8a748f910534e6985cf5ecdcee4c9977b6b74c6531e2aef7ab9ee298c3d47627  fused-run2.json
```

## Mixed AIC/AIV two-Conv block experiment

A more aggressive `MIX_AIC_1_2` kernel was then implemented for the fixed
FP32 `[2, 50, 512]` Code2Wav block. One launch performs both causal packing
operations, both Cube matrix multiplications, LayerNorm, Mish, the gated
residual, and both cache updates. C220 AIV sub-block folding, explicit
MTE2/MTE3 event ordering, and a 16-AIC channel split are used on
`ascend910_93`.

The direct 910C operator suite passed for the mixed block and FP16, FP32, and
BF16 pack paths. Against the eager operator boundary, 300-iteration timing
gave:

| Path | Median latency | P95 latency | Change vs native pack |
| --- | ---: | ---: | ---: |
| Standard eager block | 387.796 us | 389.459 us | +55.08% |
| Native pack path | 250.058 us | 257.685 us | baseline |
| Mixed two-Conv block | 145.516 us | 146.022 us | -41.81% |

The mixed output stayed within the kernel's explicit FP32 approximation
bounds: hidden maximum/mean absolute error `0.003918/0.000198`, and cache
maximum/mean absolute error `0.011787/0.000313`. The error comes from the raw
AscendC vector transcendental approximation used by Mish rather than indexing
or cache corruption.

That microbenchmark win did not survive the resident graph. After adding the
required tiling parse metadata, TorchAir compiled the full mixed Conv+MLP
megagraph and logged live replay with no fallback. Two matched CFM6 runs used
the same 32 English Seed-TTS rows, seed zero, temperature zero, three warmups,
and concurrency one as the native-pack experiment:

| Two-run mean metric | Native fused pack | Mixed block megagraph | Change |
| --- | ---: | ---: | ---: |
| Serving duration | 41.786 s | 55.109 s | +31.88% |
| Request throughput | 0.7658 req/s | 0.5808 req/s | -24.16% |
| Audio throughput | 3.353x | 2.556x | -23.77% |
| Mean TTFT | 318.27 ms | 326.19 ms | +2.49% |
| Median TTFT | 322.76 ms | 328.90 ms | +1.90% |
| P99 TTFT | 456.31 ms | 464.79 ms | +1.86% |

A second deployment kept the mixed block outside GE and preserved the
separate MLP graph. Its single confirmation run was also slower: `54.440 s`,
`0.5878 req/s`, and `2.587x` audio throughput. The opaque block removes graph
optimizer freedom worth more than its eager launch savings. It also changed
the aggregate output from 480 tokens / 140.12 seconds to 481 tokens / 140.84
seconds, so the official WER/SIM gates would be required before any use.

Decision: keep the mixed kernel and profile as opt-in experimental research,
but retain the native causal-pack megagraph as the 910C competition default.
The next kernel work should fuse within a graph-profitable boundary or use a
GE-visible decomposition; eager microbenchmark wins alone are not promotion
criteria.

Mixed-result checksums:

```text
2247cb0f9d85700349fd4f168fd6163f8b70eb21ceb4946b0336ed2fe3cb3002  mix-block-megagraph-run1.json
c950e6bf4c0e0d822b3f7866fe16c5c6c8c3af8b9b8c37fd0ebde05ce3f33332  mix-block-megagraph-run2.json
0abb9ae8e0403d30f8c02672f61cdc3f98f606fee631fe9f0a60fd32f1ee5377  mix-block-split-run1.json
```

## GE-visible lowering of the aggressive Conv profile

The opaque mixed-block boundary has now been removed from resident graph
replay. Two integration layers enforce that boundary:

1. The vLLM Ascend converter for
   `npu_minicpmo_causal_conv_block` decomposes the block into two small
   `MinicpmoCausalConvPack` layout nodes and native GE `MatMulV2`, `Reshape`,
   `LayerNormV4`, `Mish`, `Mul`, `Add`, `SplitV`, and `ConcatV2` nodes.
2. The vLLM-Omni serving path goes further and selects the already-proven
   causal-pack Conv+MLP callable directly. This lets Dynamo and TorchAir use
   their normal ATen-to-GE lowering and gives the aggressive and competition
   profiles one canonical graph and cache key.

The mixed `MIX_AIC_1_2` kernel remains available for eager operator research;
it is not inserted into the resident graph. A kernel cannot simultaneously be
one opaque custom launch and expose its internal GEMMs and vector operations
to GE. The graph-visible path therefore keeps custom code only where it is
profitable: the causal history packing and two-frame cache extraction.

On-device validation on the same `DevEnv_132987` Ascend 910C host produced the
following sequence. The hand-authored GE converter compiled successfully but
measured `57.64 s` and `2.44x` audio throughput. The direct canonical lowering
also compiled and logged live replay:

```text
Compiled MiniCPM-o NPU DiT Conv+MLP megagraph for 2x50x512
MiniCPM-o NPU DiT GE-visible Conv-block + MLP graph replay active
```

Because the historical `41.786 s` native-pack result was no longer
reproducible on the current machine state, a fresh block-off control was run
immediately after the graph-visible trial with the same source, model, 32
Seed-TTS rows, CFM6 schedule, three warmups, concurrency one, and temperature
zero:

| Current-state metric | Native-pack control | GE-visible aggressive profile | Change |
| --- | ---: | ---: | ---: |
| Serving duration (lower) | 55.09 s | 56.49 s | +2.54% |
| Request throughput (higher) | 0.58 req/s | 0.57 req/s | -1.72% |
| Mean TTFT (lower) | 326.27 ms | 318.36 ms | -2.42% |
| Median TTFT (lower) | 332.39 ms | 319.06 ms | -4.01% |
| P99 TTFT (lower) | 461.37 ms | 455.17 ms | -1.34% |
| Audio throughput (higher) | 2.56x | 2.49x | -2.73% |

Both paths completed 32/32 requests with zero failures, 100% continuity,
4,801 input tokens, 481 output tokens, 3,380,160 audio frames, and 140.84
seconds of audio. Since both profiles now select the exact same graph callable,
the small duration/throughput spread is run-to-run service variance, not a
different Conv graph. The change fixes the optimizer boundary but does not
claim a new speedup over the causal-pack graph; that path was already the
graph-profitable implementation.

Focused validation passed 55/55 vLLM-Omni Code2Wav tests and the vLLM Ascend
converter structure test. Full Seed-TTS WER/SIM, Daily-Omni, and Video-MME
qualification remains required before changing the competition accuracy
status.

New raw results are in the existing experiment directory:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-fused-conv-20260814/results
```

Result checksums:

```text
c644fdefc4571eace58c7f604a37d0b36ae3e5ae258f5fe85f7b93d2c72220b7  ge-visible-run1.json
645a85a36274e6a7864f47400af4d51611e0fac2f54942e17bb10944a8cfdf12  ge-visible-direct-run1.json
24f49fce2bbc349768544dd4df28f31e5e56168fb42c85a1a0aa1dd900f5aa02  ge-visible-canonical-run1.json
b9f9e88e26ed5724f4558855461a713bea35c3c0cd4efba3e38c4bf2fe664083  current-native-pack-control-run1.json
```

## Causal-pack plus Cube-projection fusion experiment

The next kernel experiment narrows the opaque boundary that hurt the mixed
two-Conv block. Each custom node now contains exactly one fixed-shape causal
history pack, its immediately consuming FP32 `512 x 1536` Cube projection,
the projection bias, and the two-frame cache update. LayerNorm, Mish, the
gated residual, cache assembly, and the complete MLP remain ordinary nodes in
the surrounding TorchAir graph so GE retains visibility across the expensive
rest of the block.

The `KERNEL_TYPE_MIX_AIC_1_2` implementation uses AIV cores to materialize the
tap-major history while 16 AIC cores split the projection's output channels.
The vLLM Ascend package adds the ACLNN binding, meta function, TorchAir
converter, fixed-shape tiling, and a paired microbenchmark. Its build wrapper
now also removes generated `csrc/build` metadata when the selected operator
set changes. This was required after a stale `binary_info_config.json`
packaged a compiled kernel without registering it.

The clean CANN 9.0 `ascend910_93` package registered all three MiniCPM-o
operators. Five direct NPU tests passed: three pack dtypes, the mixed block,
and the new fused projection. The fused FP32 output matched pack plus
`F.linear` within `rtol=1e-4, atol=1e-3`; the cache was bit-exact. The complete
vLLM-Omni Code2Wav routing suite passed 57/57.

Fifteen paired microbenchmark trials alternated execution order; each trial
contained 100 calls after 50 warmups:

| 910C operator path | Median latency | Minimum latency | Change |
| --- | ---: | ---: | ---: |
| Native pack + projection | 78.320 us | 77.226 us | baseline |
| Fused pack + Cube projection | 69.537 us | 68.813 us | -11.21% (1.126x) |

The resident service then used the same 32 English Seed-TTS rows, seed zero,
temperature zero, three warmups, CFM6, and concurrency one. The candidate log
confirmed live use rather than fallback:

```text
MiniCPM-o NPU DiT fused Conv+Linear + MLP graph replay active
```

It was followed immediately by a clean service restart with fused linear off
and the native pack graph on. Both runs completed 32/32 with zero failures,
100% continuity, 4,801 input tokens, 481 output tokens, 3,380,160 audio frames,
and 140.84 seconds of audio.

| Paired metric | Native-pack control | Fused projection | Change |
| --- | ---: | ---: | ---: |
| Serving duration | 53.954 s | 54.661 s | +1.31% |
| Request throughput | 0.5931 req/s | 0.5854 req/s | -1.30% |
| Audio throughput | 2.610x | 2.577x | -1.30% |
| Mean E2E | 1,685.52 ms | 1,707.81 ms | +1.32% |
| Median E2E | 1,727.27 ms | 1,768.22 ms | +2.37% |
| Mean TTFT | 318.77 ms | 324.25 ms | +1.72% |
| Median TTFT | 320.18 ms | 324.38 ms | +1.31% |
| Mean audio TTFP | 865.62 ms | 880.60 ms | +1.73% |
| Median audio TTFP | 871.47 ms | 881.26 ms | +1.12% |
| Mean per-chunk RTF | 0.40310 | 0.40801 | +1.22% |
| Median per-chunk RTF | 0.25755 | 0.26315 | +2.17% |

Lower is better except for throughput. The isolated boundary is faster, but
the saving is too diluted inside the complete Code2Wav stage to clear service
variance or the promotion gate; every paired end-to-end metric moved in the
wrong direction. The feature therefore remains disabled by default and is
available only through
`minicpmo_4_5_2npu_910c_cfm6_dit_conv_linear_experimental.yaml`. The native
pack plus GE-visible projection remains the competition path. Future kernel
work must remove a larger amount of Stage-2 work without hiding GE-profitable
operations; a sub-10-us boundary saving is not large enough by itself.

Raw results remain in:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-fused-conv-20260814/results
```

```text
604970b0820f58412b50cd406d11efb579668e608c16eb1ca80ce63b5d586da2  conv-linear-k1-seedtts-32.json
12f204d43a0775c5afdfdd1cc8e08238c2eb03c6ae76f2c92ea55f5550b9b22e  paired-pack-control-seedtts-32.json
```

## HiFT F0 feature-graph upgrade

A fresh native Torch-NPU profile of a warmed Seed-TTS request showed that the
remaining Stage-2 cost is no longer dominated by a single DiT boundary. The
largest operator families were `MatMulV2` (49.250 ms, 12.36%), `TransData`
(48.408 ms, 12.15%), `Transpose` (40.329 ms, 10.12%), the custom causal pack
(37.112 ms, 9.31%), and `LayerNormV3` (33.268 ms, 8.35%). Shape aggregation
also exposed a repeated fixed HiFT F0 stack: 405 weight-layout conversions for
`[512, 512, 1, 3]` convolution weights consumed 24.122 ms in the profiled
request.

The new candidate compiles HiFT's five fixed Conv1d+ELU feature layers as one
static TorchAir graph for the steady streaming shape `[1, 80, 58]`. It keeps
the checkpoint's original per-timestep Linear classifier outside GE. The
first `[1, 80, 50]` chunk and every incompatible shape fall back to the
original predictor. Startup checks materialize the immutable inference-only
weight-normalized convolutions and require bit-exact feature output before the
patch is installed. A 910C screening run measured 407.790 us for the complete
eager predictor, 248.509 us for an experimental complete static graph, and
241.228 us for the promoted five-Conv feature graph. The complete graph was
rejected because replacing Linear with a 1x1 Conv changed its accumulation
order; the promoted boundary retains Linear unchanged.

This experiment also found and fixed an orchestration defect: `runtime.env`
from platform stage overlays was not applied while local LLM workers were
spawned. Stage environment variables are now scoped to the serialized spawn,
inherited by only the intended child, and restored afterward. The candidate
log consequently proves both configuration and live execution:

```text
[stage_init] Stage-2 applied runtime env keys: [...HIFT_F0_GRAPH, ...HIFT_F0_GRAPH_WIDTH, ...HIFT_MATERIALIZE_WEIGHT_NORM]
Compiled HiFT F0 feature graph for Ascend NPU: batch=1 width=58
HiFT F0 graph falling back for runtime shape (1, 80, 50); compiled shape is (1, 80, 58)
HiFT F0 feature graph replay active for runtime shape (1, 80, 58)
```

The paired service test used fresh restarts, the same source and model, the
same 32 English Seed-TTS rows, seed zero, temperature zero, three warmups,
CFM6, and concurrency one. The control retained weight-norm materialization
and every existing DiT optimization but disabled only the new F0 graph.

| Paired metric | Weight-norm control | HiFT F0 graph | Change |
| --- | ---: | ---: | ---: |
| Serving duration | 44.724 s | 42.505 s | -4.96% |
| Request throughput | 0.7155 req/s | 0.7529 req/s | +5.22% |
| Audio throughput | 3.133x | 3.297x | +5.22% |
| Mean E2E | 1,397.18 ms | 1,327.91 ms | -4.96% |
| Median E2E | 1,420.00 ms | 1,335.23 ms | -5.97% |
| Mean TTFT | 324.60 ms | 314.55 ms | -3.10% |
| Median TTFT | 330.48 ms | 313.13 ms | -5.25% |
| P99 TTFT | 465.85 ms | 445.42 ms | -4.39% |
| Mean audio TTFP | 809.87 ms | 787.02 ms | -2.82% |
| Median audio TTFP | 819.56 ms | 790.42 ms | -3.56% |
| Mean per-chunk RTF | 0.34559 | 0.32990 | -4.54% |
| Median per-chunk RTF | 0.17820 | 0.14090 | -20.93% |

Lower is better except for throughput. Both paths completed 32/32 requests
with zero failures and 100% streaming continuity. They produced identical
aggregate structure: 4,801 input tokens, 480 output tokens, 3,362,880 audio
frames, and 140.12 seconds of audio.

An additional paired eight-item export preserved the sample rate and exact
frame count for all eight utterances. Fresh service restarts make HiFT's
random excitation prevent byte-identical WAV files; nevertheless, candidate
versus control waveform correlation averaged 0.999938 (minimum 0.999808) with
40.205 dB mean SNR. The exact NPU feature-partition test and 113/113 affected
CPU unit tests also passed. The host did not contain the official Whisper WER
checkpoint and could not reach Hugging Face, so full Seed-TTS WER/SIM remains
an explicit promotion gate alongside Daily-Omni and Video-MME. The graph
profile therefore remains opt-in despite the positive speed result.

Use these paired profiles:

```text
vllm_omni/deploy/minicpmo_4_5_2npu_910c_cfm6_hift_weight_norm_control.yaml
vllm_omni/deploy/minicpmo_4_5_2npu_910c_cfm6_hift_f0_graph_experimental.yaml
```

Raw results and WAV exports are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-hift-f0-20260815
```

Result checksums:

```text
7581689ff07792a4d1952c4543dbee1d40ad19195e6f545478e828ca2bbc68ab  control-en32.json
e30299874e9f5d02f235740ed8b9e68ea26658282fa806a4685a78a2b03e0a61  candidate-en32.json
```

## Prompt-width DiT graph buckets and widened Conv+MLP boundary

The post-HiFT-F0 Stage-2 trace showed that the fixed 50-frame streaming path
was no longer the only relevant DiT shape. Prompt setup and finalization also
repeated stable 302- and 20-frame shapes, but both fell back to the original
eager block. The first candidate generalized the existing TorchAir MLP and
attention-preamble partitions to an explicit `[20, 50, 302]` width set. It
also allowed the uncached setup pass to use the graph path: upstream
`CausalConv1d.forward_chunk(None)` creates an all-zero causal history, so the
adapter can preserve exact cache semantics without requiring a prior chunk.

All six MLP/preamble shapes compiled and replayed on the same 910C host. The
three-run split-boundary median was effectively neutral versus the immediately
preceding HiFT-F0 result: serving duration improved 0.08% and mean per-chunk
RTF improved 0.91%, while mean TTFP regressed 1.01% and median per-chunk RTF
regressed 9.33%. This confirmed that compiling more small islands alone did
not remove enough eager layout and launch overhead.

The widened candidate therefore adds a regular Conv/cache + gated residual +
MLP megagraph for the 20- and 302-frame buckets. It deliberately does not use
the fixed-width native causal-pack converter: regular Conv1d remains visible
inside GE at these widths, while the qualified 50-frame path continues to use
the native-pack megagraph. Both additional graphs compiled, and a live request
logged independent replay at widths 302 and 20 with no fallback:

```text
Compiled MiniCPM-o NPU prompt Conv+MLP megagraph for 2x20x512
Compiled MiniCPM-o NPU prompt Conv+MLP megagraph for 2x302x512
MiniCPM-o NPU prompt Conv+MLP megagraph replay active at width=302
MiniCPM-o NPU prompt Conv+MLP megagraph replay active at width=20
```

Three resident candidate runs used the same 32 fixed English Seed-TTS rows,
three warmups, concurrency one, seed zero, temperature zero, CFM6, and nested
TTS request body. The comparison below uses the three-run candidate median
and the immediately preceding HiFT-F0 control. Lower is better except for
throughput.

| Metric | HiFT-F0 control | Widened prompt graph | Change |
| --- | ---: | ---: | ---: |
| Serving duration | 42.505 s | 41.281 s | -2.88% |
| Request throughput | 0.7529 req/s | 0.7752 req/s | +2.96% |
| Audio throughput | 3.297x | 3.394x | +2.96% |
| Mean E2E | 1,327.91 ms | 1,289.58 ms | -2.89% |
| Median E2E | 1,335.23 ms | 1,318.25 ms | -1.27% |
| P99 E2E | 1,755.99 ms | 1,705.56 ms | -2.87% |
| Mean TTFT | 314.55 ms | 310.09 ms | -1.42% |
| Median TTFT | 313.13 ms | 315.62 ms | +0.79% |
| P99 TTFT | 445.42 ms | 450.95 ms | +1.24% |
| Mean audio TTFP | 787.02 ms | 749.41 ms | -4.78% |
| Median audio TTFP | 790.42 ms | 752.99 ms | -4.74% |
| P99 audio TTFP | 933.19 ms | 915.84 ms | -1.86% |
| Mean per-chunk RTF | 0.32990 | 0.31848 | -3.46% |
| Median per-chunk RTF | 0.14090 | 0.15142 | +7.47% |
| P99 per-chunk RTF | 1.05869 | 1.01870 | -3.78% |

Every widened run completed 32/32 requests with zero failures and retained
the exact aggregate structure: 4,801 input tokens, 480 output tokens,
3,362,880 audio frames, and 140.12 seconds of audio. The full Code2Wav
regression file passed 66/66, including exact partition math at widths 20, 50,
and 302. The candidate remains opt-in because median/P99 TTFT and median chunk
RTF did not improve, and structural parity is not a substitute for the full
Seed-TTS WER/SIM, Daily-Omni, and Video-MME accuracy gates.

Use the experimental profile:

```text
vllm_omni/deploy/minicpmo_4_5_2npu_910c_cfm6_dit_prompt_graph_buckets_experimental.yaml
```

Raw results are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-prompt-graph-buckets-20260815/results
```

Result checksums:

```text
27ae6871bc2043d6bc553c9825a31f60335af63781ff4826f7bbabc5ff8c7e07  candidate-en32-valid.json
104c1ccf6e1cfc135106f5a975e7efb8d99e8724c42bdc7759153f5202405c0e  candidate-en32-run2.json
a34cf535a89e5331e120113e720be3c4fd950626d14757789e1a8063227900a5  candidate-en32-run3.json
bdc4a6ba38a45f5848429cba24749096fff7bcb0bf9f9b78b441cfc9d0e0dcda  prompt-wide-en32-run1.json
0396c16fdc8cb55aca790f9085cfa80305f963e31f1f6a07f8d78c6e67b8c563  prompt-wide-en32-run2.json
83b355b474c1d4ef094b18f6aaec97847da7fd9a8f4c302054eb1d4e67a3c372  prompt-wide-en32-run3.json
```

## Further tuning: HiFT bucket and complete-DiT graph screening

The next tuning pass tested two wider boundaries rather than assuming that
more graph coverage is automatically faster. Both candidates remain disabled
because same-host measurements rejected them.

### HiFT first-chunk bucket

The HiFT feature compiler now supports an optional static-width list through
`VLLM_OMNI_MINICPMO45_NPU_HIFT_F0_GRAPH_BUCKETS`. The candidate added width 50
beside the accepted width 58. Both shapes compiled, matched the eager feature
stack bit-for-bit, and replayed on NPU. Three 32-item runs completed without a
failure, but their 42.650-second median was 3.31% slower than the preceding
41.281-second prompt-width median. The extra bucket is therefore not enabled
by any promoted profile.

This experiment also exposed a configuration correctness issue: stage `env`
maps were replaced rather than recursively merged while resolving a derived
deploy profile. `env` now follows the same deep-merge rule as engine arguments,
so a child profile can add one stage variable without dropping inherited HiFT
flags.

Result checksums:

```text
5183bebda37cfce1a49a2aefae7b5abf6d30932a6bae02ed72e042b028e1ad4a  f0-buckets-en32-run1.json
0b5e277d41fc99a0a917361d1b88f36e04d8c95b4dcaf1aa858a75339f79a2e2  f0-buckets-en32-run2.json
8b9e954e74a43a6e059f2877a2ad39778d2d962fef113e5fb68a3519f3784fec  f0-buckets-en32-run3.json
```

### Full DiT block and 16-block stack graphs

Torch-NPU 2.10 rewrites SDPA to the six-output
`npu_fusion_attention_v3` overload, while the competition image's TorchAir
contains a converter only for the older seven-output overload. The Ascend fork
now has a lazy inference converter that lowers the v3 BNSD/no-dropout subset to
GE `FlashAttentionScore`. With it, complete width-50 DiT block graphs compiled
at the three observed cache lengths 302, 352, and 402 and replayed at all three
lengths in a real request.

That successful lowering was not a speed win. A warmed same-row request took
6.312 seconds versus 1.166 seconds on the restored split boundary; audio TTFP
regressed from 0.760 to 6.014 seconds. Combining all 16 blocks into one graph
removed per-block replay overhead but still took 6.168 seconds. Replacing
small-shape FlashAttention (`q=50`, `kv<=452`, head dimension 64) with explicit
`BatchMatMulV2 -> softmax -> BatchMatMulV2` also compiled as one stack graph,
but took 6.149 seconds. The near-identical results show that the opaque call
boundary was not the limiting cost: this CANN/TorchAir version produces a
large GE plan whose execution is substantially slower than the qualified
split eager/graph path. Cold first use was also 49-59 seconds for these plans.

| Same first Seed-TTS row, warmed | E2E | Audio TTFP | First chunk RTF | Decision |
| --- | ---: | ---: | ---: | --- |
| Restored prompt-width split profile | 1.166 s | 0.760 s | 0.905 | Keep |
| Complete graph per DiT block | 6.312 s | 6.014 s | 7.160 | Reject |
| One 16-block stack, FlashAttention | 6.168 s | 5.820 s | 6.928 | Reject |
| One 16-block stack, explicit BMM attention | 6.149 s | 5.814 s | 6.921 | Reject |

The graph implementations and profiles remain opt-in diagnostics for newer
CANN/TorchAir releases; neither flag is set by the accepted profile. Runtime
guards fail closed to the existing split path on an unsupported layout, cache
length, or compile failure.

After restoring
`minicpmo_4_5_2npu_910c_cfm6_dit_prompt_graph_buckets_experimental.yaml`, two
fresh resident 32-item checks completed 32/32 with zero failures and exact
aggregate parity (4,801 input tokens, 480 output tokens, 3,362,880 frames, and
140.12 seconds of audio). They measured 44.681 and 44.619 seconds on the current
host state. This is slower than the earlier 41.281-second three-run median but
matches the older 44.724-second HiFT control; because all active graph markers
and request structure are unchanged, it is recorded as host/run variance, not
as a promoted regression or improvement. The directly affected Code2Wav and
NPU-platform suites passed 99/99. Official Seed-TTS WER/SIM, Daily-Omni, and
Video-MME gates remain required before promoting any accuracy-changing
attention replacement.

Raw results are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-hift-f0-buckets-20260817
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-full-dit-v3-20260817
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-full-stack-20260817
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-full-stack-bmm-20260817
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-further-tuning-20260817
```

Selected warmed/restored checksums:

```text
de07894d23922b864cbb4489c0927bf9bcd70961579156dc617e7734cf2ee5c3  full-dit-v3-axis-smoke-2.json
161cbd79fb766f6b591ffe6636bbc4b979f2970b12c9ee248246c33c4a3099bc  full-stack-smoke2.json
c078a44dae35b4c0677e6b6e83d3efaa2dd56a45512354bb49d2bebe7e94dca7  full-stack-bmm-smoke2.json
0de7ea26751f335318d5ac944d85a0258b03b15e51cf2b7dfc4e1a9a5116f35b  restored-prompt-en32.json
901f70df0afbc529445f141ff043464f83fc940d0b13f15673affe40be1df133  restored-prompt-en32-run2.json
```

## Cache-major native causal-convolution path

A fresh Stage-2 NPU trace of the restored prompt-width profile identified
`MinicpmoCausalConvPack` as a repeated small-kernel cost: 576 calls consumed
38.321 ms, or 66.53 us per call. Its channel-major cache layout
`[batch, channels, 2]` requires scalar gathers for the two historical taps and
scalar writes for every cache update. The native AscendC operator now also
accepts a cache-major `[batch, 2, channels]` layout. That layout makes both
historical taps and the returned cache contiguous DMA copies while preserving
the old ABI and path.

The vLLM-Omni adapter retains this cache-major layout across the steady
50-frame CFM stream instead of transposing it at each of the 16 DiT blocks.
Prompt and final non-steady boundaries remain on the existing layout. The new
path is guarded by `VLLM_OMNI_MINICPMO45_NPU_DIT_CACHE_MAJOR`, requires the
qualified native pack plus Conv+MLP graph, and fails closed to the existing
implementation.

Post-install NPU validation passed all six dtype/layout cases with exact
outputs. An alternating-order kernel microbenchmark measured the following;
lower is better:

| Native causal-pack layout | Median | Minimum |
| --- | ---: | ---: |
| Channel-major control | 62.777 us | 62.746 us |
| Cache-major candidate | 27.161 us | 26.741 us |

The cache-major kernel is 2.31x faster by median. The custom-op subset build
also exposed stale CMake `AICPU_CUST_OBJ_TARGETS` state when changing the
selected operator list. The build now clears that derived cache before
collecting targets. A clean package containing both `AddRmsNormBias` and
`MinicpmoCausalConvPack` installed with SHA-256:

```text
999387e27f4547660164719ab43a0147e0899823346cf3f1153befde1ab276a6
```

The end-to-end A/B used fresh service starts, followed by three resident runs
per side. Every run used the same 32 Seed-TTS English rows, three warmups,
concurrency one, seed zero, temperature zero, and CFM6 request body. Each run
completed 32/32 requests with zero failures and generated exactly 4,801 input
tokens, 480 output tokens, 3,362,880 frames, and 140.12 seconds of audio.

| Metric (three-run median) | Prompt-width control | Cache-major candidate | Change |
| --- | ---: | ---: | ---: |
| Serving duration | 44.200 s | 43.754 s | -1.01% |
| Request throughput | 0.7240 req/s | 0.7314 req/s | +1.02% |
| Audio throughput | 3.170x realtime | 3.202x realtime | +1.02% |
| Mean E2E | 1,380.78 ms | 1,366.86 ms | -1.01% |
| Median E2E | 1,421.81 ms | 1,407.19 ms | -1.03% |
| Mean TTFT | 315.74 ms | 315.49 ms | -0.08% |
| Mean audio TTFP | 783.94 ms | 775.40 ms | -1.09% |
| Median audio TTFP | 781.19 ms | 783.18 ms | +0.25% |
| Mean per-chunk RTF | 0.33907 | 0.33443 | -1.37% |
| Median per-chunk RTF | 0.18370 | 0.18478 | +0.59% |
| P99 per-chunk RTF | 1.08678 | 1.06626 | -1.89% |

Lower is better except for throughput. The live service logged
`MiniCPM-o NPU cache-major Conv+MLP megagraph replay active`, with no compile
failure or eager fallback. The result is a small end-to-end win rather than a
2.31x service gain because the causal-pack kernel is only one small component
of the full text, audio-token, CFM, and vocoder pipeline. Median TTFP and
median chunk RTF are effectively neutral and slightly worse, so the new
profile remains opt-in:

```text
vllm_omni/deploy/minicpmo_4_5_2npu_910c_cfm6_dit_cache_major_experimental.yaml
```

The aggregate-output and exact native-kernel checks are structural evidence,
not a replacement for the full official Seed-TTS WER/SIM, Daily-Omni, and
Video-MME accuracy gates. Those gates are still required before promotion.

A separately tested HiFT static-weight conversion was rejected: frozen and
unfrozen paths measured 213.259 us and 212.383 us respectively (0.996x), and
the attempted FRACTAL_Z lowering was not semantically valid for this Conv1d
weight layout. Neither experiment is enabled.

Raw results are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-cache-major-20260817
```

Result checksums:

```text
fd0f3623d1adb1ab25e0de14891d5cb7de0b40f510c11851ff6620fa952a9122  candidate-run1.json
c021eca41c2e99afe8b3b0c5650542e14ae322d7d9f2ba95072e24103843eb07  candidate-run2.json
967163735a4001ab69bf370815846bb8ffd28c676a079231a7b8912d4b6321ce  candidate-run3.json
3ec715f54c52f23cf730feb3a8b5f47495ea39bb9c7d9c4a61498414ed524936  control-run1.json
0cc6dc6f17b104de35ea2139b433519da3c5d68176a0c49732a9ee29d0033f5e  control-run2.json
96251e1a84d9f6bee266f67e48578ed46c9da051e11190b532158323e74b4fae  control-run3.json
```

## Post-attention graph and native QKV layout screening

Two follow-up candidates tested narrower lower-level boundaries against the
cache-major path. Neither is promoted.

The first moved the attention residual, `norm3`, modulation, native causal
pack, convolution, and MLP into one post-attention graph. It compiled and
replayed, but a fail-fast 32-request run took 47.460 seconds, 8.47% longer
than the 43.754-second cache-major median. Mean E2E rose to 1,482.69 ms and
mean chunk RTF to 0.36418. The opaque wider graph therefore prevents more
valuable GE scheduling than it saves in Python/launch overhead. It remains an
explicit diagnostic only:

```text
vllm_omni/deploy/minicpmo_4_5_2npu_910c_cfm6_dit_post_attention_experimental.yaml
```

The second candidate kept the successful preamble boundary but replaced the
three fixed `[2,50,512]` BSH-to-BNSD materializations with one AscendC
`MinicpmoQkvPack` launch. Q/K normalization and SDPA remain visible to GE;
widths 20 and 302 continue to use the ordinary preamble graph. The custom
operator passed exact FP16, FP32, and BF16 checks. Alternating-order NPU
microbenchmarks measured the following (lower is better):

| Width-50 QKV layout path | Median | Minimum |
| --- | ---: | ---: |
| Three transpose/materialize operations | 67.618 us | 62.530 us |
| Native QKV pack | 40.267 us | 40.136 us |

The native kernel is 1.68x faster by median, and TorchAir successfully
compiled and replayed it inside the preamble graph. The complete Seed-TTS
fail-fast run nevertheless took 44.642 seconds: 2.03% slower than the
cache-major candidate median and effectively equal to the recent
44.619-second restored control. It completed 32/32 requests with zero
failures and exact aggregate parity: 4,801 input tokens, 480 output tokens,
3,362,880 frames, and 140.12 seconds of audio.

| QKV candidate metric | Result |
| --- | ---: |
| Request throughput | 0.7168 req/s |
| Audio throughput | 3.1388x realtime |
| Mean / median E2E | 1,394.68 / 1,449.75 ms |
| Mean / median TTFT | 319.63 / 328.86 ms |
| Mean / median audio TTFP | 787.90 / 796.39 ms |
| Mean / median chunk RTF | 0.34324 / 0.18816 |

This is a useful kernel but not a serving optimization on the current
CANN/TorchAir stack, so it also remains opt-in:

```text
vllm_omni/deploy/minicpmo_4_5_2npu_910c_cfm6_dit_qkv_pack_experimental.yaml
```

The selected three-op package (`AddRmsNormBias`, causal pack, and QKV pack)
has SHA-256 `47230e94d72cc8c61070126597c3c095eaf1143fc652c0c7b1056fb983a00ab7`.
The Ascend build now supports an explicit selected-op override and an
extension-only rebuild against an already installed ACLNN package, avoiding
accidental recompilation of the full custom-op matrix during kernel iteration.

Raw results are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-post-attention-20260817
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-qkv-pack-20260817
```

Result checksums:

```text
78739916d99822fb74624ecb6046ba380b212d96f262cba576899bce4dfdffd9  post-attention candidate-run1.json
e5b4729f87abfd99c47fd6c29683cc14fb0be8d8719b173be82e182221628ed5  qkv-pack candidate-run1
```

## HiFT stage-0 residual-block graphs

The next Stage-2 boundary targets the vocoder rather than widening the DiT
graphs. For the steady 58-frame mel chunk, HiFT's first transposed-convolution
stage produces `[1, 256, 464]`. It then evaluates three parallel residual
blocks, each containing three `Snake -> Conv1d -> Snake -> Conv1d -> add`
sequences. The new opt-in path compiles each complete residual block as one
static TorchAir graph. Upsampling, source injection, and ISTFT remain visible
to the existing eager pipeline, and every non-matching shape uses the original
bound method.

Startup derives the graph shape from the checkpoint's transposed-convolution
parameters, materializes immutable inference weight norm, compiles all three
siblings, and requires bit-exact output from every graph before installing any
of them. A runtime graph exception disables only that block and fails closed
to eager execution. The focused patch suite passed 40/40 both locally and in
the 910C environment.

The saved standalone NPU-1 microbenchmark used 20 warmups and 100 iterations
per block. Lower is better:

| Three stage-0 residual blocks | Total latency | Relative |
| --- | ---: | ---: |
| Eager | 3,608.739 us | 1.000x |
| TorchAir graphs | 1,671.924 us | 2.158x faster |

All three block outputs had maximum absolute error `0.0`. The live candidate
service subsequently logged all three replay markers on real Stage-2 inputs,
with no graph failure or eager fallback.

The end-to-end A/B used the existing widened prompt-graph profile as control.
The candidate added only:

```text
vllm_omni/deploy/minicpmo_4_5_2npu_910c_cfm6_hift_resblock_graph_experimental.yaml
```

Both variants ran three times over the same 32 English Seed-TTS rows with
three warmups, concurrency one, seed zero, temperature zero, and the same CFM6
request body. Every run completed 32/32 with zero failures and 100% streaming
continuity. Every run also produced exactly 4,801 input tokens, 480 output
tokens, 3,362,880 frames, and 140.12 seconds of audio.

| Metric (three-run median) | Prompt-graph control | HiFT residual graphs | Change |
| --- | ---: | ---: | ---: |
| Serving duration | 44.136 s | 41.632 s | -5.67% |
| Request throughput | 0.7250 req/s | 0.7686 req/s | +6.02% |
| Mean E2E | 1,378.87 ms | 1,300.67 ms | -5.67% |
| Median E2E | 1,420.05 ms | 1,314.86 ms | -7.41% |
| P99 E2E | 1,886.40 ms | 1,767.13 ms | -6.32% |
| Mean TTFT | 314.96 ms | 310.72 ms | -1.35% |
| Median TTFT | 321.22 ms | 311.90 ms | -2.90% |
| P99 TTFT | 453.78 ms | 451.33 ms | -0.54% |
| Mean audio TTFP | 777.75 ms | 760.16 ms | -2.26% |
| Median audio TTFP | 783.08 ms | 763.25 ms | -2.53% |
| P99 audio TTFP | 913.11 ms | 919.56 ms | +0.71% |
| Mean per-chunk RTF | 0.339188 | 0.319996 | -5.66% |
| Median per-chunk RTF | 0.187670 | 0.160253 | -14.61% |
| P99 per-chunk RTF | 1.043074 | 1.025924 | -1.64% |

Lower is better except for throughput. The candidate improves every median
and every reported speed metric except P99 TTFP, whose 0.71% regression is
small but explicit. Exact residual-block output and aggregate serving parity
provide strong semantic evidence, but they are not substitutes for the full
1,088-row Seed-TTS WER/SIM, Daily-Omni, and Video-MME accuracy gates. The
profile therefore remains opt-in until those gates and a longer tail-latency
run are complete. The accepted prompt-graph control was restored after the
experiment.

Raw artifacts are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-hift-resblock-20260817
```

Artifact checksums:

```text
a7a92998e8b6018e144848451114df0c0eaba0e0eacb0266891c14c0a907fb46  micro-stage0.log
943e0bd3b773adf02664dbeb665487e6921bfc97a0383baab0b0607647d9c57b  candidate-run1.json
35650e00f497ddd7b830265f33f43420b80fc750f01aa495e3b6ce8fe09a2242  candidate-run2.json
670b22a8f1e191e57d10e814a19a58e9d35ecc2640e4f1ade2b9e4e9948b2377  candidate-run3.json
8680484e28ecd65c81d16d819d70409cc0bdc775eb9939243864bb790cc698c5  control-run1.json
d17d90c7ca78ad195a85462157cd06a636884839b72f194cc86b52f6e57abd67  control-run2.json
085bb5a82d07c5227d5654fa22b123b3f34b0036c6512b0b34755e71720ac08f  control-run3.json
```

### Wider-stage screening

Stages 1 and 2 were also screened at their derived steady shapes. In
isolation, their three-block eager/graph totals were 3,542.261/1,704.709 us
(2.078x) and 3,651.084/1,693.272 us (2.156x), respectively. All six graph
outputs again had maximum absolute error `0.0`.

Those microbenchmark wins were not additive in the complete service. A
diagnostic candidate compiled and replayed all nine blocks across stages 0,
1, and 2 without fallback, but its first warmed 32-row run took 45.941 s. That
is 4.09% slower than the 44.136-second control median and 10.35% slower than
the selected 41.632-second stage-0 median. Mean E2E was 1,435.34 ms, mean
audio TTFP 797.35 ms, and mean chunk RTF 0.351995. It retained 32/32 success,
100% continuity, and exact aggregate structure, so the loss is execution
efficiency rather than a correctness failure.

The wider boundary is rejected after this fail-fast run. Additional graph
residency and GE/layout interactions outweigh the isolated launch savings;
the committed profile therefore continues to compile stage 0 only.

Additional artifact checksums:

```text
391cbe2476c75afc5b58490e3a0ea5fb855be4a3db0eab389d67ac3ea6beb1ea  micro-stage1.log
1b6f5710dd93b5b0649fe413a93592ffaf6f7022da74ed0f4f19720c0f877d2e  micro-stage2.log
fa4705e403b0d91473e5761138dd39c6725f244b180b73afdf92ad4fc40330c1  all-stages candidate-run1.json
```

### Native full-block and aggregate-graph screening

Two more aggressive ways of reducing the three stage-0 graph replays were
implemented and measured on the same 910C host. Neither passed the promotion
gate, so both implementations were removed rather than retained behind another
environment flag.

The first candidate was a native AscendC operator covering one complete HiFT
residual block. Its isolated ACLNN package and Torch extension built and ran on
the target NPU, but the hand-packed convolution path was both slower and less
accurate than CANN's native Conv1d sequence:

| Kernel size | Eager | Native fused op | Speed | Max / mean absolute error | Cosine |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3 | 700.381 us | 1,217.207 us | 0.575x | 0.029517 / 0.000091 | 0.999961 |
| 7 | 764.729 us | 2,244.750 us | 0.341x | 0.051999 / 0.000198 | 0.999858 |
| 11 | 777.552 us | 2,388.385 us | 0.326x | 0.106543 / 0.013123 | 0.987252 |

This rejects hand-lowering HiFT Conv1d through im2col plus Matmul. A future
native attempt needs a real CANN/AscendC convolution primitive or a
layout-specialized direct convolution; pointwise fusion alone cannot recover a
2--3x convolution regression or the changed accumulation order.

The second candidate kept CANN's existing convolution kernels but compiled all
three parallel residual blocks and their exact sum into one static TorchAir
graph. The first patched block returned the aggregate and the two siblings
returned neutral tensors, while every mismatch or graph failure used the exact
eager sum. Compilation required bit-exact output before installation.

Its real-checkpoint NPU microbenchmark was compelling but misleading in
isolation:

| Three stage-0 residual blocks | Total latency | Relative |
| --- | ---: | ---: |
| Eager | 3,671.139 us | 1.000x |
| Existing three graphs | 1,671.924 us | 2.196x faster than this eager run |
| Aggregate graph | 1,189.099 us | 3.087x faster than eager; 28.88% below three graphs |

The aggregate output had maximum absolute error `0.0`, and the resident service
logged the aggregate replay marker without fallback. End-to-end behavior still
regressed. The first run took 53.211 seconds; after all lazy compilation was
resident, the second run took 46.523 seconds. The table compares that faster
second run with the accepted stage-0 three-graph median. Lower is better except
for throughput:

| Metric | Accepted three-graph median | Aggregate warm run | Change |
| --- | ---: | ---: | ---: |
| Serving duration | 41.632 s | 46.523 s | +11.75% |
| Request throughput | 0.7686 req/s | 0.6878 req/s | -10.51% |
| Mean / median / P99 E2E | 1,300.67 / 1,314.86 / 1,767.13 ms | 1,453.44 / 1,495.60 / 1,980.69 ms | +11.75% / +13.75% / +12.09% |
| Mean / median / P99 TTFT | 310.72 / 311.90 / 451.33 ms | 340.62 / 338.50 / 553.26 ms | +9.62% / +8.53% / +22.58% |
| Mean / median / P99 audio TTFP | 760.16 / 763.25 / 919.56 ms | 818.34 / 821.03 / 1,023.54 ms | +7.65% / +7.57% / +11.31% |
| Mean / median / P99 chunk RTF | 0.319996 / 0.160253 / 1.025924 | 0.353397 / 0.175568 / 1.112279 | +10.44% / +9.56% / +8.42% |

Both aggregate runs completed 32/32 requests with zero failures, 100%
continuity, 4,801 input tokens, 480 output tokens, 3,362,880 audio frames, and
140.12 seconds of audio. The regression is therefore execution efficiency, not
workload or output-structure drift. The larger opaque graph boundary removes
three replay launches but prevents more valuable scheduling/layout optimization
around the existing HiFT path. The result closes aggregate sibling capture for
this software stack: retain the three independent stage-0 graphs, and make any
next HiFT fusion transparent to GE or lower it inside CANN's convolution
implementation.

Raw artifacts are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-hift-aggregate-20260817
```

Artifact checksums:

```text
c200cee3144ec4312eaf8c9116ccb7e103b86992b0db47debcce08ff1c13f016  candidate-run1.json
6bfb71ade93dfb06bb75d201d7be7d1fd1d2b084d938814d6073825f9d207705  candidate-run2.json
```

### Requalification against the current accepted stack

The stage-0 residual boundary was requalified on 2026-08-18 after the accepted
profile gained single-request cache ownership. This matters because the older
5.67% serving win above used an earlier control. The fresh control, the
original three-block graph implementation, and a second aggregate design each
ran three times over the identical 32 English rows, after three warmups, at
concurrency one. All nine runs completed 32/32 requests with zero failures,
100% streaming continuity, 4,801 input tokens, 480 output tokens, 3,362,880
audio frames, and 140.12 seconds of audio.

The new aggregate design compiled the three parallel residual siblings as one
TorchAir graph returning three exact outputs. A thread-local dispatcher let the
unchanged flashcosyvoice reduction consume those outputs in order, avoiding
the earlier neutral-tensor implementation and leaving the sibling sum outside
the graph. Its isolated stage result was again attractive: 3,407.161 us eager
versus 1,345.148 us graph, a 2.533x speedup, with maximum absolute error `0.0`.
The service logged exactly one real-input replay marker and no fallback.

End-to-end results rejected both graph boundaries on the current stack. Lower
is better except for throughput:

| Metric (three-run median) | Current accepted control | Three block graphs | Change | One sibling graph | Change |
| --- | ---: | ---: | ---: | ---: | ---: |
| Serving duration | 40.214 s | 41.654 s | +3.58% | 43.524 s | +8.23% |
| Request throughput | 0.7957 req/s | 0.7682 req/s | -3.46% | 0.7352 req/s | -7.61% |
| Mean E2E | 1,256.29 ms | 1,301.26 ms | +3.58% | 1,359.70 ms | +8.23% |
| Median E2E | 1,281.78 ms | 1,328.67 ms | +3.66% | 1,388.77 ms | +8.35% |
| P99 E2E | 1,696.85 ms | 1,763.51 ms | +3.93% | 1,858.12 ms | +9.50% |
| Mean TTFT | 314.07 ms | 310.29 ms | -1.20% | 316.94 ms | +0.92% |
| Mean audio TTFP | 745.37 ms | 757.09 ms | +1.57% | 776.67 ms | +4.20% |
| Mean chunk RTF | 0.310126 | 0.319916 | +3.16% | 0.332815 | +7.32% |
| Median chunk RTF | 0.144614 | 0.167456 | +15.80% | 0.181294 | +25.36% |
| P99 chunk RTF | 1.007790 | 1.022899 | +1.50% | 1.035547 | +2.75% |

The isolated kernel savings therefore do not compose with the complete HiFT
pipeline. Even one tuple-output graph creates a synchronization/layout boundary
that costs more than its removed launches. Neither candidate advances to the
1,088-row WER/SIM run: accuracy work cannot rescue a failed speed gate. The
accepted profile remains graph-free at this boundary, and future HiFT work must
fuse transparently inside CANN's convolution/layout path rather than add a
TorchAir boundary around sibling blocks.

Raw artifacts are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-hift-resblock-qualification-20260818
```

Artifact checksums:

```text
abe951f9989445761862f3bca81ebc30996d0ac16ee7aeb4ebfbfe7a7aa2fbda  control/control-run-1.json
7d9dd9e2bc4114e75932142ff1b410fe12898490b8de9cb1153f4172c0b43c8f  control/control-run-2.json
02cff3619ceb333b513ea2e92c0b082ceab57a8a7aebf882c286320a7c5a0fd3  control/control-run-3.json
e35136f57b15fd792577769dec05870900478f0a2580c15b3ec5fc6d57bca627  candidate/candidate-run-1.json
7ed4a92cbbde0d468b7204dd7e4471e298cb08ed85c71b9aea51501af65943ba  candidate/candidate-run-2.json
d827e5e53970ee88c3e8e73baf90eac5b22ea6c16758173a881821962cc81dfb  candidate/candidate-run-3.json
3189e51ed4fd9545e9058d2be276986131dc0726e7736cc492c41dc9a800bf70  sibling/sibling-run-1.json
0e846e98dc229937b357bd69dc9f9b4b75ab893cdff800b103877eb6b4474828  sibling/sibling-run-2.json
ca91c08c3859567fb08a6703f27400a2071cf3c3994be770fabb4ae20d6bf1eb  sibling/sibling-run-3.json
2b71d975ebb39a1a0cbfde9b36e08d293cb2895671c8bd53f223f57ab1ce5038  sibling-service-v2.log
```

### Fixed-size HiFT ISTFT graph and layout screens

The next lower-level screen targeted HiFT's final inverse transform. MiniCPM-o
fixes this operation at `n_fft=16` and `hop_len=4`, while the steady 58-frame
mel chunk reaches `[1,9,6961]` magnitude and phase tensors. The candidate
replaced complex-tensor construction and the general `torch.istft` path with
two real 16-by-9 linear transforms and an exact four-way Hann overlap-add.
The centered edge envelope is precomputed from the checkpoint window rather
than assuming an interior constant.

The graph is guarded by shape, dtype, device, and checkpoint ISTFT parameters.
Its startup gate compares the compiled waveform with upstream, and every
unsupported input or graph exception fails closed to the original bound
method. The focused suite passed 44/44. On NPU 1, 30 warmups and 200 measured
steady-width iterations produced:

| Steady HiFT ISTFT | Latency | Relative |
| --- | ---: | ---: |
| Generic complex `torch.istft` | 429.646 us | 1.000x |
| Specialized eager real path | 319.710 us | 1.344x faster |
| Specialized TorchAir graph | 178.029 us | 2.413x faster |

The specialized output had maximum absolute error `8.38e-9`, mean absolute
error `1.43e-9`, and cosine similarity `1.0`. The live service compiled and
replayed the graph at `[1,9,6961]` without fallback.

That isolated 58.57% graph win did not survive the serving gate. A fresh
same-source control used the accepted three stage-0 residual graphs; the
candidate inherited that profile and added only the fixed ISTFT graph. Both
services ran the same 32 English Seed-TTS rows three times after three
warmups. Every run completed 32/32 with zero failures, 100% streaming
continuity, 4,801 input tokens, 480 output tokens, 3,362,880 frames, and
140.12 seconds of audio. The table uses the median of the three per-run
metrics; lower is better except for throughput.

| Metric | Fresh control | Fixed ISTFT graph | Change |
| --- | ---: | ---: | ---: |
| Serving duration | 44.751 s | 46.070 s | +2.95% |
| Request throughput | 0.7151 req/s | 0.6946 req/s | -2.86% |
| Mean / median / P99 E2E | 1,397.99 / 1,420.43 / 1,874.67 ms | 1,439.20 / 1,472.16 / 1,942.34 ms | +2.95% / +3.64% / +3.61% |
| Mean / median / P99 TTFT | 321.18 / 321.28 / 471.20 ms | 316.36 / 314.83 / 449.43 ms | -1.50% / -2.01% / -4.62% |
| Mean / median / P99 audio TTFP | 799.67 / 796.31 / 962.61 ms | 794.57 / 793.96 / 935.50 ms | -0.64% / -0.30% / -2.82% |
| Mean / median / P99 chunk RTF | 0.342507 / 0.185401 / 1.089012 | 0.350319 / 0.194845 / 1.065672 | +2.28% / +5.09% / -2.14% |

The graph improves TTFT and TTFP, but it regresses the primary serving,
throughput, E2E, and central chunk-RTF metrics. As with the rejected aggregate
residual graph, the extra opaque replay boundary prevents more valuable
whole-pipeline scheduling than its local kernel saving recovers. The profile
therefore remains diagnostic-only, and the full Seed-TTS WER/SIM,
Daily-Omni, and Video-MME gates were not spent on a candidate that already
failed the speed gate:

```text
vllm_omni/deploy/minicpmo_4_5_2npu_910c_cfm6_hift_fixed_istft_graph_experimental.yaml
```

Two adjacent layout screens were also closed before service promotion.
Enabling CANN internal formats after NPU initialization retained bit-exact
residual outputs, but paired long-run graph totals were effectively tied at
about 1,705 us candidate versus 1,707 us control. Re-expressing all 18
stage-0 Conv1d operations as singleton-height Conv2d was bit-exact, but each
kernel was neutral to slightly slower; CANN selected the same effective path.
Direct packed Conv3d was rejected by the installed CANN rewrite because its
`Conv3dv2` fusion accepts static shapes only.

Raw serving artifacts are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-hift-fixed-istft-20260817
```

Artifact checksums:

```text
28fb51a87f9c91ccb805cae19706a894a8dfc0a130474e7fa51741cecb6dfec3  candidate-run1.json
69bc48c0bbef830f3dee185ad1b267b1b886de0f12b0b55258a2b1fe911f1ad0  candidate-run2.json
218fb9dca59c1db0f11ffb41a8410a3eb3364aa134db691a476327badcbf0e90  candidate-run3.json
12558f29afdd2295055c30b08a52b6380188e50ec4b7f00698c3669a6bd1b9a8  control-run1.json
c518d4ad1b6904656c34595f995960b4106ba468fe55a4e12db8ae122ca2ebfc  control-run2.json
c9608777664cd5923391949af8bf17636cb33948528dc20bbc5a8196e2f965fc  control-run3.json
```

### Promoted HiFT STFT-window residency

The fixed-ISTFT rejection exposed a smaller transparent optimization.
`flashcosyvoice.HiFTGenerator` assigns its 16-value Hann window as an ordinary
CPU tensor rather than a registered module buffer. Both `_stft` and `_istft`
therefore evaluate `self.stft_window.to(input.device)` on every streamed
chunk. The promoted patch moves that immutable tensor to the Stage-2 device
once after checkpoint loading. The upstream STFT, ISTFT, complex arithmetic,
window values, and accumulation order remain unchanged; their existing
`.to(npu)` calls become no-ops.

A real-width NPU-1 microbenchmark used 50 warmups and 500 iterations. All
outputs were bit-exact:

| Operation at 6,961 spectral frames | CPU window | Resident NPU window | Change |
| --- | ---: | ---: | ---: |
| HiFT ISTFT | 457.194 us | 397.764 us | -13.00% |
| HiFT STFT | 291.760 us | 159.465 us | -45.35% |

The placement is idempotent and fails closed: a missing or incompatible
window logs a warning and leaves the existing per-call copies in place. The
focused NPU patch suite passed 46/46.

The serving candidate added window residency to the selected three stage-0
residual graphs. It was compared with the fresh three-run residual-graph
control collected immediately before this candidate on the same host and
source stack. Both sides used 32 fixed English Seed-TTS rows, three warmups,
concurrency one, seed zero, temperature zero, and CFM6. Every run completed
32/32 with zero failures, 100% continuity, 4,801 input tokens, 480 output
tokens, 3,362,880 frames, and 140.12 seconds of audio. The table reports the
three-run median; lower is better except for throughput.

| Metric | Fresh control | Resident window | Change |
| --- | ---: | ---: | ---: |
| Serving duration | 44.751 s | 42.720 s | -4.54% |
| Request throughput | 0.7151 req/s | 0.7491 req/s | +4.75% |
| Mean / median / P99 E2E | 1,397.99 / 1,420.43 / 1,874.67 ms | 1,334.69 / 1,358.89 / 1,809.03 ms | -4.53% / -4.33% / -3.50% |
| Mean / median / P99 TTFT | 321.18 / 321.28 / 471.20 ms | 308.79 / 308.87 / 441.33 ms | -3.86% / -3.86% / -6.34% |
| Mean / median / P99 audio TTFP | 799.67 / 796.31 / 962.61 ms | 761.79 / 765.78 / 907.33 ms | -4.74% / -3.83% / -5.74% |
| Mean / median / P99 chunk RTF | 0.342507 / 0.185401 / 1.089012 | 0.326712 / 0.180163 / 1.026645 | -4.61% / -2.83% / -5.73% |

Every measured performance gate improves. Since the patch changes no model
operation or value and the isolated STFT/ISTFT outputs are bit-exact, the
existing Seed-TTS, Daily-Omni, and Video-MME qualifications carry forward.
Window residency is therefore promoted as an always-on Ascend HiFT behavior;
it has no deployment flag or separate production profile.

Raw artifacts are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-hift-window-resident-20260818
```

Artifact checksums:

```text
12558f29afdd2295055c30b08a52b6380188e50ec4b7f00698c3669a6bd1b9a8  control-run1.json
c518d4ad1b6904656c34595f995960b4106ba468fe55a4e12db8ae122ca2ebfc  control-run2.json
c9608777664cd5923391949af8bf17636cb33948528dc20bbc5a8196e2f965fc  control-run3.json
9282e7c6c88b515999eff419db4bdf25e6192aab9c14f522ae11112af5f7dd7d  candidate-run1.json
c7579ea11df3dca4ecfdd0d9ec88f563e191c47061ca24a188bfe923e029e046  candidate-run2.json
00345f32a473ffa628013ac4455160ab5c625d7221685cc26b04a32cd95fdd35  candidate-run3.json
```

### HiFT harmonic-residency screen

The next transparent allocation screen targeted `flashcosyvoice.SineGen2`.
Its upstream `forward` constructs the immutable harmonic multiplier with
`torch.FloatTensor` on CPU and copies it to the input device for every audio
chunk. The candidate preserves that exact constructor, values, shape, and
subsequent operations, but creates the tensor once after checkpoint loading
and keeps it on Stage 2's NPU. Device or dtype mismatches delegate to the
original method.

At the real steady waveform width `[1,27840,1]`, 100 warmups and 500 measured
iterations on NPU 1 reduced complete SineGen2 latency from 719.709 us to
560.056 us, a 22.18% isolated improvement. The cached multiplier and the
resulting `f0 * harmonics` tensor were bit-exact (`max_abs_error=0`). Full
sine tensors differed by about `2.93e-4`, but two unmodified baseline calls
with the same CPU and NPU seeds differed by the same amount; this is the
existing randomized NPU phase behavior, not a changed deterministic input.
UV and noise tensors were exact. The focused patch suite passed 48/48, and
the candidate service logged resident-window and resident-harmonic placement
without fallback.

The serving screen measured the new cache incrementally on top of the
promoted resident Hann window and the selected three Stage-0 residual graphs.
Both sides used the same 32 fixed English Seed-TTS rows, three warmups,
concurrency one, seed zero, temperature zero, and CFM6. Every run completed
32/32 with zero failures, 100% continuity, 4,801 input tokens, 480 output
tokens, 3,362,880 frames, and 140.12 seconds of audio. The table reports the
median of three runs; lower is better except for throughput.

| Metric | Fresh control | Resident harmonics | Change |
| --- | ---: | ---: | ---: |
| Serving duration | 43.860 s | 44.688 s | +1.89% |
| Request throughput | 0.7296 req/s | 0.7161 req/s | -1.85% |
| Mean / median / P99 E2E | 1,370.25 / 1,368.05 / 1,915.25 ms | 1,396.02 / 1,405.37 / 1,808.26 ms | +1.88% / +2.73% / -5.59% |
| Mean / median / P99 TTFT | 321.80 / 317.79 / 490.74 ms | 332.88 / 333.73 / 468.17 ms | +3.44% / +5.02% / -4.60% |
| Mean / median / P99 audio TTFP | 764.01 / 760.94 / 939.70 ms | 774.66 / 778.04 / 919.50 ms | +1.39% / +2.25% / -2.15% |
| Mean / median / P99 chunk RTF | 0.340351 / 0.144086 / 1.101073 | 0.343077 / 0.146450 / 1.082056 | +0.80% / +1.64% / -1.73% |

The candidate improves tail metrics but regresses every primary duration,
throughput, mean, and median gate. The saved host-to-device copy is too small
to dominate full-pipeline scheduling variance, and its local microbenchmark
win does not qualify it for the default path. The implementation therefore
remains opt-in for diagnostic work, and the more expensive Seed-TTS WER/SIM,
Daily-Omni, and Video-MME accuracy gates were not spent on a candidate that
already failed the speed gate:

```text
vllm_omni/deploy/minicpmo_4_5_2npu_910c_cfm6_hift_harmonics_resident_experimental.yaml
```

Raw serving artifacts are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-hift-harmonics-resident-20260818
```

Artifact checksums:

```text
bf204c7e79b3d42b80b957ceb2452b1418a2028613d608b205aa15615e4cf75e  control-1.json
47f6525988b7654611b7ed53c58b971e81670ccef3548cefd4f90e299f2ab8ca  control-2.json
ad200c8809ef1264b97a0818b227aec946a16c1ce815252c505d21366811d4ff  control-3.json
a61765e0d150c6ba976a4cb57dbe22cf2421190dab6988dfd609aa7edeb93ee8  candidate-1.json
f71855129c223c285c63894954f4687c88c68101409437d6b051e829ed50d75b  candidate-2.json
0a5a79b6f888c8c27cda3383573b2d2ef8c7420edb660d104ae07aaa6ba9b8e4  candidate-3.json
```

### Direct DiT attention-cache output screen

The next allocation screen targeted the prompt-width DiT attention path.
Each block already receives a correctly sized final attention-cache buffer,
but the existing implementation allocates full K, full V, and packed KV
temporaries before copying the packed result into that buffer. The candidate
uses Ascend's supported `torch.cat(..., out=view)` form to concatenate K and V
directly into the caller-owned packed-cache views. It retains the original
cache order, SDPA inputs, projection, and fallback path.

At the real steady shape (CFG batch 2, 8 heads, 50 new positions, 352 cached
positions, head dimension 64), a 100-warmup, 1,000-iteration NPU-1 screen
measured complete cache assembly plus SDPA. The normal path took 95.387 us;
direct output took 87.076 us, an 8.71% isolated improvement. Attention and
packed-cache outputs were bit-exact. A four-slice `copy_` workspace variant
was rejected earlier because it took 118.311 us. The focused Code2Wav suite
passed 78/78, the complete inherited profile passed its configuration gate,
and the live service logged direct-output activation without graph fallback.

The real serving screen compared the accepted prompt-width DiT graph profile
with an otherwise identical profile adding only direct cache output. Both
sides used the same 32 fixed English Seed-TTS rows, three warmups, concurrency
one, seed zero, temperature zero, and CFM6. Every run completed 32/32 with
zero failures, 100% continuity, 4,801 input tokens, 480 output tokens,
3,362,880 frames, and 140.12 seconds of audio. The table reports the median of
three runs; lower is better except for throughput.

| Metric | Fresh control | Direct cache output | Change |
| --- | ---: | ---: | ---: |
| Serving duration | 46.515 s | 47.664 s | +2.47% |
| Request throughput | 0.6880 req/s | 0.6714 req/s | -2.41% |
| Mean / median / P99 E2E | 1,453.18 / 1,442.89 / 2,140.12 ms | 1,489.15 / 1,501.44 / 2,264.97 ms | +2.47% / +4.06% / +5.83% |
| Mean / median / P99 TTFT | 330.57 / 327.46 / 554.18 ms | 329.55 / 326.25 / 477.57 ms | -0.31% / -0.37% / -13.82% |
| Mean / median / P99 audio TTFP | 796.32 / 797.72 / 1,013.64 ms | 800.24 / 794.39 / 962.98 ms | +0.49% / -0.42% / -5.00% |
| Mean / median / P99 chunk RTF | 0.366797 / 0.177978 / 1.231375 | 0.385161 / 0.184624 / 1.141904 | +5.01% / +3.73% / -7.27% |

The direct-output views improve TTFT and several tails, but regress primary
duration, throughput, E2E, mean TTFP, and central chunk RTF. The noncontiguous
packed-cache views save local allocations while producing a less favorable
layout/scheduling boundary for the surrounding DiT execution. The candidate
therefore remains diagnostic-only, and the full WER/SIM, Daily-Omni, and
Video-MME gates were not spent after the speed gate failed:

```text
vllm_omni/deploy/minicpmo_4_5_2npu_910c_cfm6_dit_attn_cache_out_experimental.yaml
```

Raw serving artifacts are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-dit-attn-cache-out-20260818
```

Artifact checksums:

```text
f928525c3637048d2db62bfdc3af4b94096068c6d1f833613f43de6e04a1bd49  control-1.json
cce1b87205cc54a16fdbf7c5a69083b1290eefa0b32583a09f6275200e73a6a9  control-2.json
cad990be05ac634ee17ff089036b664503d7f264aa0afcc8c328ee9374441e3e  control-3.json
617c422f43f7b04f19de6a1782b0236398b80203021f5e3867f09a644a0ef02d  candidate-1.json
948355dc40d6469139d3eed3d277d1f7893808745f3de7b5ecdaada704140343  candidate-2.json
f6900409b6b6b0514daae46d9e6eeaddede18b120476647f30e078fcdf635f6e  candidate-3.json
```

### Direct stacked CFM cache-output screen

The CFM loop runs six estimator steps. Each step allocated separate CNN and
attention-cache outputs, retained all twelve tensors, and finally allocated
and copied them again with two `torch.stack` calls. The candidate instead
allocates the two final stacked states once and passes each step a view to
write directly. It preserves the original DiT operations, cache values, and
step order.

At the real six-step cache shapes, `[6,6,2,1024,2]` for CNN state and
`[6,6,2,8,402,128]` for attention state, a 20-warmup, 100-iteration NPU-1
screen measured the old allocation-plus-stack path at 277.230 us and the two
direct stacked allocations at 31.235 us. That is an 88.73% isolated reduction
and removes a terminal stack measured independently at 223.170 us. Stacked
values were bit-exact, the focused Code2Wav and deployment-configuration
suites passed, and the live service logged direct stacked-output activation.

The real serving result went in the opposite direction. Both sides used the
same 32 fixed English Seed-TTS rows, three warmups, concurrency one, seed zero,
temperature zero, and CFM6. Every run completed 32/32 with zero failures,
100% continuity, 4,801 input tokens, 480 output tokens, 3,362,880 frames, and
140.12 seconds of audio. The table reports the median of three runs; lower is
better except for throughput.

| Metric | Fresh control | Direct stacked output | Change |
| --- | ---: | ---: | ---: |
| Serving duration | 41.529 s | 44.809 s | +7.90% |
| Request throughput | 0.7705 req/s | 0.7141 req/s | -7.32% |
| Mean / median / P99 E2E | 1,297.42 / 1,326.18 / 1,745.27 ms | 1,399.92 / 1,436.63 / 1,846.78 ms | +7.90% / +8.33% / +5.82% |
| Mean / median / P99 TTFT | 306.19 / 305.41 / 441.22 ms | 315.03 / 316.11 / 452.67 ms | +2.89% / +3.50% / +2.60% |
| Mean / median / P99 audio TTFP | 743.92 / 745.58 / 887.04 ms | 776.33 / 779.57 / 905.73 ms | +4.36% / +4.56% / +2.11% |
| Mean / median / P99 chunk RTF | 0.318128 / 0.171573 / 1.012530 | 0.343567 / 0.181579 / 1.041428 | +8.00% / +5.83% / +2.85% |

An Ascend format probe explains why the allocation-only screen did not
transfer. A normal per-step CNN output is NCHW, while its view inside the
five-dimensional stacked allocation inherits NCDHW. A normal per-step
attention output is NCDHW, while its view inside the six-dimensional stacked
allocation inherits generic ND. The logical shapes and strides are identical,
but the less favorable physical formats slow the much larger DiT writes and
reads by more than the removed copies save. This candidate is therefore kept
opt-in for layout research, and the WER/SIM, Daily-Omni, and Video-MME gates
were not spent after the speed gate failed:

```text
vllm_omni/deploy/minicpmo_4_5_2npu_910c_cfm6_stacked_cache_out_experimental.yaml
```

Raw artifacts are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-cfm-stacked-cache-out-20260818
```

Artifact checksums:

```text
89417ddf61b7cce1a38acb978a1ebd7c7987092cbc4137f5b4b500fdea07290f  control-run-1.json
109d13a7a9ed04d0fdd5acd38a5f6743d6257aea98f625d5be0d17a9d793b5fc  control-run-2.json
91aebfa15b1559575a4cf2d174d065f450e71cf9c6888768cca534aa61ccd569  control-run-3.json
5c016e3947e647c1d7948f43bc441d61bf4ea83dee4bec904d344f8d9e931138  candidate-run-1.json
209671bf5a2a1ab7df6dd8d1d16400bf9d533799195e8d784dcf8d08642be7b3  candidate-run-2.json
00944ff240f0c0d6a2dde9240e16334323f5ecc278ba77205ac6d6bf1466989f  candidate-run-3.json
```

### Promoted single-request cache ownership

The rejected direct stacked-output screen exposed a larger copy outside the
DiT kernels. Competition latency runs use concurrency one, but after every
chunk `_split_flow_cache` copied the entire six-step CFG estimator state into
the request, and before the next chunk `_stack_flow_cache` copied it back into
an identical one-request batch. The old state is read-only during decoding, so
the promoted path transfers tensor ownership directly when the batch contains
exactly one request. Multi-request batches keep the established CFG reorder
and copy behavior.

At the real cache shapes, a 20-warmup, 100-iteration NPU-1 screen reduced a
complete split-plus-restack round trip from 682.480 us to 15.665 us, a 97.70%
reduction. CNN and attention values were bit-exact, and the original NCDHW and
ND formats were preserved. The 421-test focused Code2Wav/config suite passed
before promotion; an additional three-chunk state-and-audio comparison was
bit-exact, and the promoted deployment gate passed.

The live candidate was compared with a fresh accepted-profile control on the
same host and source stack. Both sides used the same 32 fixed English Seed-TTS
rows, three warmups, concurrency one, seed zero, temperature zero, and CFM6.
Every run completed 32/32 with zero failures, 100% continuity, 4,801 input
tokens, 480 output tokens, 3,362,880 frames, and 140.12 seconds of audio. The
table reports the median of three runs; lower is better except for throughput.

| Metric | Fresh control | Cache passthrough | Change |
| --- | ---: | ---: | ---: |
| Serving duration | 43.167 s | 42.684 s | -1.12% |
| Request throughput | 0.7413 req/s | 0.7497 req/s | +1.13% |
| Mean / median / P99 E2E | 1,348.63 / 1,381.66 / 1,823.12 ms | 1,333.55 / 1,360.32 / 1,806.22 ms | -1.12% / -1.54% / -0.93% |
| Mean / median / P99 TTFT | 310.60 / 311.44 / 445.09 ms | 309.55 / 311.00 / 453.84 ms | -0.34% / -0.14% / +1.97% |
| Mean / median / P99 audio TTFP | 774.48 / 773.15 / 912.98 ms | 763.16 / 765.01 / 909.76 ms | -1.46% / -1.05% / -0.35% |
| Mean / median / P99 chunk RTF | 0.331659 / 0.179186 / 1.042957 | 0.328043 / 0.177109 / 1.030444 | -1.09% / -1.16% / -1.20% |

The candidate improves every primary serving, E2E, TTFP, and chunk-RTF gate.
TTFT P99 is the only regression (+1.97%), while its mean and median improve.
Because the fast path changes no model operation or tensor value and the
multi-chunk comparison is bit-exact, the existing Seed-TTS, Daily-Omni, and
Video-MME accuracy qualifications carry forward. Single-request passthrough
is therefore enabled in the accepted prompt-width profile; higher-concurrency
serving continues to use the original state path.

Raw artifacts are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-single-request-cache-passthrough-20260818
```

Artifact checksums:

```text
1f4689d0d7701d26c640b780428fb993ea7e07a4039450a923d69d32c11e3251  control-run-1.json
9cdde0d4883cb94f1043fd80336824dbd3667753b14f0bde52983737d930d729  control-run-2.json
1dbeb38953f199fb01fcad4397a9f74ebeede0716f1bdf07987c39bd07c26ea1  control-run-3.json
446d2f54f6fcedc01747a608425b7ad8ccfd761a6f49a375cd569a0b71197fce  candidate-run-1.json
b8f34a5d07f1f81e3466cad0605faa6d60170f2d52503eb202b40e7d111d5abb  candidate-run-2.json
408c5857170eae2ed619e184cb26e344808543c5a929c99611d2f0dd70e3a8fc  candidate-run-3.json
```

## Rejected HiFT source-noise scratch reuse

`SourceModuleHnNSF2` returns a full-waveform auxiliary noise tensor on every
HiFT invocation, but MiniCPM-o immediately discards that second return value.
The opt-in candidate preserves the exact `randn`, multiply, divide, return
shape, dtype, physical stride, and RNG advancement while reusing one buffer
per waveform shape:

```text
vllm_omni/deploy/minicpmo_4_5_2npu_910c_cfm6_hift_source_noise_scratch_experimental.yaml
```

CPU tests proved bit-exact outputs and next-RNG state, idempotent installation,
shape-isolated storage, and pointer reuse. On NPU 1 at the steady
`[1,27840,1]` source shape, 30 warmups and 200 iterations reduced the complete
source-module invocation from 796.460 us to 793.205 us, only 0.41%. The
discarded noise and next RNG draw had maximum absolute error `0.0`, and the
scratch pointer was reused. The unchanged sine path varied by `7.12e-5` across
seed-reset NPU replays, consistent with the existing phase-kernel
nondeterminism; the candidate does not alter that path.

The final service experiment used fresh services on both sides and the fork's
local benchmark client, avoiding both service-age skew and an older installed
client that omitted Omni timing arrays. Each side ran the same 32 fixed English
Seed-TTS rows three times after three warmups, at concurrency one, seed zero,
temperature zero, and CFM6. Every run completed 32/32 with zero failures, 100%
continuity, 4,801 input tokens, 480 output tokens, 3,362,880 frames, and 140.12
seconds of audio. The table reports the median of three runs; lower is better
except for throughput.

| Metric | Accepted control | Noise scratch | Change |
| --- | ---: | ---: | ---: |
| Serving duration | 41.774 s | 44.729 s | +7.07% |
| Request throughput | 0.7660 req/s | 0.7154 req/s | -6.61% |
| Mean / median / P99 E2E | 1,304.97 / 1,310.02 / 1,889.46 ms | 1,397.34 / 1,420.56 / 1,902.58 ms | +7.08% / +8.44% / +0.69% |
| Mean / median / P99 TTFT | 318.76 / 319.63 / 460.08 ms | 316.52 / 319.56 / 452.54 ms | -0.70% / -0.02% / -1.64% |
| Mean / median / P99 audio TTFP | 780.61 / 764.30 / 947.08 ms | 787.72 / 785.88 / 938.05 ms | +0.91% / +2.82% / -0.95% |
| Mean / median / P99 chunk RTF | 0.321152 / 0.148051 / 1.059848 | 0.341226 / 0.187301 / 1.063686 | +6.25% / +26.51% / +0.36% |

The candidate slightly improves TTFT and tail TTFP, but regresses the primary
duration, throughput, E2E, central TTFP, and every chunk-RTF gate. Runtime logs
also show many waveform widths, so persistent shape-specific buffers perturb
the allocator for a local saving too small to compose with the full pipeline.
The candidate therefore remains diagnostic-only and is not enabled in the
accepted profile. Full Seed-TTS WER/SIM, Daily-Omni, and Video-MME gates were
not spent after the speed gate failed.

Raw artifacts are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-hift-source-noise-scratch-20260818
```

Only the `final-*` files below belong to the clean promotion decision; earlier
files in that directory record harness-validation and warm-state diagnostics.

```text
c95e4f020d624f33ca9c3461ff50e3231a7822de971ddacde10962eb29f896a4  control/final-control-run-1.json
8e79420487fe1bad8c7ca1e855f27993160852b2ebf8f6b63da9815ca386a2f3  control/final-control-run-2.json
25844744dd7954af7b80a7afecc64a7887317a57faffaae2d47275f7ecc18dfa  control/final-control-run-3.json
266597c0e6941f6f0f53db9fa614702e4301ab5094d272d2320cf1a067560918  candidate/final-candidate-run-1.json
1135b27104b593d0c00790d7d55ce9ca3f01ba84833dd6a0e97508a774bbdd85  candidate/final-candidate-run-2.json
ea351aea893d9ae3146999e5e32661acf57dbe5073349547e7db79fd07474fab  candidate/final-candidate-run-3.json
b15f0ef53546918d78b021034b02ecea65286a0d43ae85698b5bb666faa2e565  final-candidate-service.log
f36d65fb21edf6612d90dd6076a7e98f6f9bbabcb9002f07291070170d81b660  final-accepted-service.log
```

## Neutral HiFT F0 classifier-graph experiment

The accepted HiFT F0 graph ends after five Conv1d+ELU layers and runs the
checkpoint's per-timestep Linear classifier and absolute value eagerly. A
larger opt-in boundary now keeps those original operations inside the same
TorchAir graph:

```text
vllm_omni/deploy/minicpmo_4_5_2npu_910c_cfm6_hift_f0_classifier_graph_experimental.yaml
```

This differs from the previously rejected complete graph, which substituted a
1x1 Conv and moved F0 by as much as 0.36 Hz. The new graph uses `F.linear`
with the original classifier weight and bias. TorchAir 8.5 initially inferred
the transposed input as K=58 instead of K=512; materializing that transpose
with `contiguous()` repaired GE shape inference without replacing the model
operation. The real 910C checkpoint compiled at `[1,80,58]` and reported
`max_abs_drift=0` on the nonzero startup gate. Incompatible widths retain the
upstream eager fallback.

Two service restarts and one warmed repetition used the same 32 fixed English
Seed-TTS rows, three warmups, concurrency one, seed zero, temperature zero,
and CFM6. Every measured run completed 32/32 with zero failures and 100%
continuity, while preserving 4,801 input tokens, 480 output tokens, 3,362,880
frames, 140.12 seconds of audio, and identical generated text. The table uses
the median run value from two accepted-control runs and three candidate runs;
lower is better except for throughput.

| Metric | Accepted control | Classifier graph | Change |
| --- | ---: | ---: | ---: |
| Serving duration | 42.041 s | 41.915 s | -0.30% |
| Request throughput | 0.7613 req/s | 0.7634 req/s | +0.28% |
| Mean E2E | 1,313.39 ms | 1,309.30 ms | -0.31% |
| Mean TTFT | 314.84 ms | 321.46 ms | +2.10% |
| Mean audio TTFP | 760.93 ms | 761.65 ms | +0.09% |
| Mean chunk RTF | 0.324751 | 0.322969 | -0.55% |
| Median chunk RTF | 0.154067 | 0.162580 | +5.53% |

The restart-level result changed sign: one fresh comparison improved duration
1.66%, the second regressed it 0.23%, and the warmed comparison improved it
0.29%. The median gain is below normal service variance, while mean TTFT
crosses the 2% guard and median chunk RTF regresses materially. The candidate
therefore remains diagnostic-only and is not inherited by the accepted
prompt-width profile. Full Seed-TTS WER/SIM, Daily-Omni, and Video-MME gates
were not spent after the speed gate failed.

Raw artifacts are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-hift-f0-classifier-20260819
```

Artifact checksums:

```text
d1af2ebaa84a1e44e4a80701060da0bdd6f57ec2d269ab289a5d795fe6d0eee5  control/results/control-seedtts-32.json
e7237229d5d9c3302b6024b0abe27c3fcc6ac3df7ec8b15e5833a9098e1befe0  control/results/control2-seedtts-32.json
1d0d685b90807661532a31d0a401203a4fb05fa67ad1c5bfe83c478f4555169c  candidate/results/candidate-seedtts-32.json
80f340dbcfff3c29770bd7b04351ee10e95ec7c2a92b5d988e90727d61879f6a  candidate/results/candidate2-seedtts-32.json
410e1aa859fc00418bd0cd40011b58195d0079acbda0c1b9034de996080e52f4  candidate/results/candidate3-seedtts-32.json
```

## Rejected HiFT F0 weight-layout experiments

The retained Stage-2 profile identified the largest remaining individual
layout conversion as the HiFT F0 stack's fixed `[512,512,1,3]` Conv1d
weights. NCHW-to-FRACTAL_Z `TransData` ran 405 times and consumed 15.192 ms
across the trace. Two exact-width `[1,80,58]` graph variants screened ways to
remove that repeated packing before spending another end-to-end run.

The first variant marks all ten immutable convolution weight and bias tensors
with guarded static addresses and enables TorchAir's `frozen_parameter`
lowering. It is available only through the diagnostic profile:

```text
vllm_omni/deploy/minicpmo_4_5_2npu_910c_cfm6_hift_f0_frozen_weights_experimental.yaml
```

Two independent 30-warmup, 200-iteration measurements changed sign. The first
measured 212.881 us for the control and 213.122 us frozen (0.999x); the second
measured 213.224 us and 209.324 us respectively (1.019x). Both had zero maximum
absolute output error. This spread is normal microbenchmark noise and does not
support promotion. In particular, accepting static tensor addresses did not
prove that GE eliminated the replay-time Conv1d weight conversions.

The second variant prepacked each kernel-3 weight as a 512-by-1536 matrix and
replaced Conv1d with explicit three-position window packing plus `F.linear`.
It measured 229.973 us versus the same run's 213.224 us control: 7.27% slower.
It also introduced maximum/mean absolute errors of 0.026312/0.000125. The extra
pad, slice, concatenate, and transpose traffic costs more than the conversions
it removes, so this form is rejected and is not wired into serving.

The reproducible focused harness is:

```text
benchmarks/scripts/bench_minicpmo_hift_f0_frozen_weights.py
```

These results close frozen-parameter annotations and framework-level im2col
linearization as F0 optimization directions on the current CANN/TorchAir
stack. A future retry must either change GE's native Conv1d weight-packing
policy or fuse the whole five-layer stack below the framework boundary while
preserving the original accumulation behavior.

## Rejected contiguous-window causal-pack kernel

The same retained Stage-2 profile attributed 38.321 ms to 576 invocations of
the two native `MinicpmoCausalConvPack` nodes. A lower-layer candidate enlarged
the AscendC UB row buffer and replaced three 512-element DMA round trips with
one contiguous 1536-element transfer for the 96 rows that do not cross the
two-frame cache boundary. The first two rows used two-source specialized
copies and narrower MTE2-to-MTE3 event synchronization.

The candidate compiled for Ascend 910C and passed all six exact operator
cases: FP16, FP32, and BF16, each with channel-major and cache-major state.
Fresh 100-warmup, 500-iteration, 15-trial measurements were:

| Layout | Installed kernel | Contiguous-window candidate | Change |
| --- | ---: | ---: | ---: |
| Channel-major | 63.552 us | 63.775 us | +0.35% |
| Cache-major (serving path) | 18.995 us | 18.849 us | -0.77% |

Lower is better. The production-layout gain is below the promotion threshold
and the compatibility layout regressed. Kernel launch and the mandatory
packed-output write dominate after the earlier cache-major optimization, so
reducing internal DMA command count does not produce a material serving win.
The candidate was removed rather than adding a second implementation for a
noise-level result; no end-to-end or accuracy-suite budget was spent.

## Opt-in wide AdaLN projection candidate

The retained Stage-2 profile showed 480 FP32 AdaLN projections with shape
`[2,512] x [4608,512]` in one 32-request trace. MiniCPM-o 4.5 has 16 DiT
blocks, and every block projects the same current CFM timestep independently.
The candidate packs those immutable block weights and biases once, computes
the current timestep's full modulation bank with one
`[2,512] x [73728,512]` Cube GEMM, and passes the corresponding row to each accepted
shape-bucketed attention-preamble graph. It does not retain modulation values
across timesteps or chunks.

The opt-in profile and real-checkpoint screening harness are:

```text
vllm_omni/deploy/minicpmo_4_5_2npu_910c_cfm6_dit_wide_adaln_experimental.yaml
benchmarks/scripts/bench_minicpmo_dit_wide_adaln.py
```

On NPU 1, nine alternating trials with 20 warmups and 100 iterations reduced
the 16-projection group from 1,853.374 us to 104.907 us, a 17.667x
microbenchmark speedup. The real `flow.pt` weights and a nonzero FP32 timestep
produced maximum and mean absolute errors of `0.0` in the isolated harness.
The live service's loaded/lowered tensors instead produced a maximum absolute
drift of `9.53674316e-07`. Serving startup therefore uses a fail-closed,
nonzero-input `1e-6` maximum-absolute-drift gate. Non-finite output, larger
drift, or incompatible block counts, shapes, or devices retain the per-block
path. Four focused model tests and four deploy configuration tests passed in
the server environment.

Fresh candidate and accepted-profile services then ran the same 32 fixed
English Seed-TTS rows three times, with three warmups, concurrency one, seed
zero, temperature zero, and CFM6. Every run completed 32/32 with zero failures
and 100% continuity while preserving 4,801 input tokens, 480 output tokens,
3,362,880 audio frames, and 140.12 seconds of audio. The table reports the
median run value from each side; lower is better except for throughput.

| Metric | Accepted control | Wide AdaLN | Change |
| --- | ---: | ---: | ---: |
| Serving duration | 45.296 s | 43.400 s | -4.19% |
| Request throughput | 0.7065 req/s | 0.7373 req/s | +4.37% |
| Mean / median / P99 E2E | 1,415.07 / 1,453.16 / 1,921.83 ms | 1,355.80 / 1,378.16 / 1,802.60 ms | -4.19% / -5.16% / -6.20% |
| Mean / median / P99 TTFT | 313.98 / 314.00 / 446.27 ms | 319.54 / 326.50 / 464.45 ms | +1.77% / +3.98% / +4.07% |
| Mean / median / P99 audio TTFP | 789.17 / 790.29 / 929.78 ms | 787.94 / 785.92 / 942.90 ms | -0.16% / -0.55% / +1.41% |
| Mean / median / P99 chunk RTF | 0.344823 / 0.195124 / 1.061778 | 0.335521 / 0.151692 / 1.055121 | -2.70% / -22.26% / -0.63% |

This first sample materially improved serving duration, throughput, every E2E
gate, and central chunk RTF. It also slightly improved central audio TTFP, but
text TTFT median and P99 appeared to regress by about 4%, beyond the accepted
profile's 2% guard. A stage-instrumented follow-up showed that result was not a
causal Stage-2 regression: client TTFT is the first Stage-0 text SSE delta, and
the wide AdaLN path runs only in Stage 2.

Fresh stage-instrumented runs measured accepted-control TTFT at
321.60/325.23/455.51 ms mean/median/P99 and active-candidate TTFT at
313.14/316.48/454.80 ms, changes of -2.63%/-2.69%/-0.16%. Stage-0 serving TTFT
and model TTFT also improved in the candidate run. Serving duration was
43.350 s for control and 43.786 s for candidate (+1.01%), while Stage-2
generation time was statistically flat. This resolves the apparent TTFT
regression and demonstrates that TTFT must not be attributed to a downstream
Code2Wav-only change.

A later fresh 32-row pair remained mixed:

| Metric | Accepted control | Wide AdaLN | Change |
| --- | ---: | ---: | ---: |
| Serving duration | 48.961 s | 51.100 s | +4.37% |
| Request throughput | 0.6536 req/s | 0.6262 req/s | -4.19% |
| Mean / median / P99 TTFT | 339.94 / 333.07 / 660.99 ms | 318.45 / 313.84 / 506.91 ms | -6.32% / -5.77% / -23.31% |
| Mean / median / P99 audio TTFP | 837.69 / 796.37 / 1,469.97 ms | 786.80 / 765.33 / 1,062.76 ms | -6.08% / -3.90% / -27.70% |
| Mean / median / P99 chunk RTF | 0.372492 / 0.189244 / 1.409467 | 0.383805 / 0.169480 / 1.265191 | +3.04% / -10.44% / -10.24% |

Both sides completed 32/32 and produced byte-identical audio content hashes
for all 32 requests, in addition to identical token, frame, and duration
counts. The active candidate separately passed the cached Seed-TTS evaluator
on eight rows with WER `0.0`, mean SIM `0.8391076`, zero request/ASR/SIM
failures, and 100% streaming continuity. Because Stage 2 is downstream of the
text answers scored by Daily-Omni and Video-MME, this candidate cannot alter
those two suites' answers; this does not replace their release-level full-suite
execution.

The TTFT blocker is fixed, the accuracy evidence is stronger, and the startup
gate now reflects live Ascend numerics. The end-to-end speed result is not yet
reproducible, however: the original three-run median improved 4.19%, while the
two fresh pairs measured +1.01% and +4.37% duration. The wide graph therefore
remains an implemented opt-in candidate and is not inherited by the accepted
profile until repeated interleaved trials show a stable end-to-end win.

Raw artifacts are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-wide-adaln-20260819
```

Artifact checksums:

```text
79db65d94a68f62806d7d763f43c78e357b2521818c12ad3a4f31dcc8c12774c  control/results/control-1.json
f114c2712108f7278ff55dbca95d08254731eafeaadffa55e9e3306b5d23ab19  control/results/control-2.json
a9f45e910500ab5e0b11c0dab74598c72b30931a00146981b0b73448e1b8f80c  control/results/control-3.json
9a40a5e22295bd89e757413aa4435f49ee1692c74db778dd8ce2c6cf979959f6  candidate/results/candidate-1.json
5bded166874bee05af9f111c036a781ec6fbbc25fc1cf29f509138e788658f0c  candidate/results/candidate-2.json
7744f74dd7924832a208dab8908d2d1e3b97620c4b65d792944a43674de48a2d  candidate/results/candidate-3.json
7a34d5e35fe9eacb0b46360742991cb0024f7127c2cefd4c491bd2cef9f73991  candidate-service.log
a11ecefb9df800dc08f92dd4eb5e9a4b9f844483fd971ffe001e669677777bf0  control/control-service.log
a94a26cc30a63e6e5757afae3ef88dc61d0a052f9db07648934445a249cd07c3  candidate/candidate-service-2.log
```

TTFT-fix and bounded-drift follow-up artifacts are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-wide-adaln-ttft-fix-20260819
```

```text
7698ec3219bcb074fc55dd9ef4f2157c46a9441d9d644520936b8b2fd4d4aa02  control/perf-control-final.json
d52f6bad9f67c72a9373689d69a01f602c551471c38d0925abe91545778b058c  candidate/perf-candidate-2.json
d6e1181aaef52ec70d8719b736ebaa8b7d065e25bd08b37fe22b343b20f8e914  control/stage-control-1.json
245ec7a2c7170a7b1ac5852006fe7e835177c1ab9ccd5959d3a528d9c3b0e97d  candidate/stage-candidate-1.json
5592a3f7fb806a8229e077a8b4bddcbb055583e1572eb63ab907284373382cd7  candidate/quality-candidate-en8.json
51208a4b65896a3e33c088843062a42d97606471db89133b05b23749c4a2dfbe  candidate/service-bounded.log
c84bb02ae8df0a7ef1046a3c71c3c5184109216added37aa4124511c14f10a0a  control/service-control-final.log
```

## Promoted all-step wide AdaLN projection

The next iteration removes the remaining per-timestep launch boundary from
the wide AdaLN candidate. MiniCPM-o 4.5 uses six fixed CFM timesteps in the
accepted profile. Their time embeddings are available before the ODE loop, so
the implementation now projects all six timesteps and all 16 DiT blocks with
one `[6,2,512] x [73728,512]` Cube GEMM per audio chunk. The ODE loop consumes
one `[2,1,16,4608]` view per step without retaining values across chunks.

The real-checkpoint NPU 1 harness is:

```text
benchmarks/scripts/bench_minicpmo_dit_wide_adaln_steps.py
```

With the loaded `flow.pt` tensors, six current-step wide projections took
719.428 us median while the single all-step projection took 97.258 us median,
a 7.397x reduction at this boundary. Maximum and mean absolute drift were both
`0.0`, including distinct embeddings for every timestep. The live startup
gate also passed with `steps_max_abs_drift=0`; the existing single-step gate
remained bounded at `9.53674316e-07`. Any compile, replay, shape, non-finite, or
drift failure disables the optimization and restores the original per-block
projections.

Fresh candidate and accepted-control processes each ran the same 32 fixed
English Seed-TTS rows three times with three warmups, concurrency one, seed
zero, temperature zero, and CFM6. Every run completed 32/32 with zero failures
and 100% continuity, preserving the identical structural signature: 4,801
input tokens, 480 output tokens, 3,362,880 audio frames, and 140.12 seconds of
audio. The table reports the median run value from each side. Lower is better
except for throughput.

| Metric | Accepted control | All-step AdaLN | Change |
| --- | ---: | ---: | ---: |
| Serving duration | 42.361 s | 40.613 s | -4.13% |
| Request throughput | 0.7554 req/s | 0.7879 req/s | +4.30% |
| Mean / median / P99 E2E | 1,323.45 / 1,360.62 / 1,798.47 ms | 1,268.77 / 1,294.40 / 1,681.84 ms | -4.13% / -4.87% / -6.49% |
| Mean / median / P99 TTFT | 309.21 / 317.84 / 454.15 ms | 315.27 / 314.85 / 456.45 ms | +1.96% / -0.94% / +0.51% |
| Mean / median / P99 audio TTFP | 762.18 / 767.61 / 909.11 ms | 744.94 / 746.11 / 886.44 ms | -2.26% / -2.80% / -2.49% |
| Mean / median / P99 chunk RTF | 0.324539 / 0.171741 / 1.021916 | 0.312637 / 0.145323 / 1.008885 | -3.67% / -15.38% / -1.28% |

All primary serving, E2E, TTFP, and RTF gates improve. TTFT is generated by
Stage 0 before this Stage-2-only path executes; its median improves and the
mean/P99 variation remains within the 2% guard. Unlike the earlier current-step
candidate, the three-run all-step result is stable: candidate duration ranged
40.443--41.690 seconds versus 42.275--44.107 seconds for control. The
optimization is therefore enabled in the accepted prompt-width profile.

The full cached Seed-TTS WER/SIM result, Daily-Omni result, and Video-MME
result remain valid because the fused projection is mathematically identical,
passes an exact real-checkpoint parity gate, and changes neither Stage-0 text
generation nor output structure. This is not a substitute for rerunning all
three suites at the final competition release gate.

Focused validation on the server passed all 84 Code2Wav tests and all 26
relevant 910C deploy-configuration tests. The live service logged active
all-step replay and completed 96 measured requests without falling back.

Raw artifacts are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-all-step-adaln-20260819
```

### Profile-guided screens rejected before serving

Three lower-level ideas were closed with real-checkpoint NPU harnesses before
spending an end-to-end service cycle:

- Freezing the DiT Conv+MLP graph weights was exact but 0.59% slower than the
  accepted explicit-weight graph. The compiler already retains these inputs
  efficiently, so making them opaque removes optimization freedom without
  eliminating useful work.
- Replacing the two causal-convolution taps with one batched tap matmul was
  numerically bounded (`3.87e-07` maximum absolute drift) but improved only
  107.9 us to 106.4 us, or 1.4%, before integration. That is below the launch
  and maintenance threshold.
- Casting the FP32 DiT matrices to actual Ascend `FRACTAL_NZ` format 29 changed
  `F.linear` semantics: maximum hidden/cache errors were 1.96/3.78. The earlier
  ND-format neutral result was a false negative; the real NZ path is rejected
  on correctness before timing can qualify it.

The reproducible screens are
`bench_minicpmo_dit_frozen_weights.py`, `bench_minicpmo_dit_tap_matmul.py`,
and `bench_minicpmo_dit_nz_weights.py` under `benchmarks/scripts/`.

## Width-64 causal-pack kernel and rejected 32-frame schedule

The next systems experiment preserved the accepted 25-frame first audio
packet, then increased steady packets to 32 codec frames. This changes the
steady DiT width from 50 to 64 and the HiFT F0 width from 58 to 72. The first
unfused screen was decisively slower because the native causal Conv packing
operator and Omni graph compatibility gate accepted only width 50: serving
duration was 66.611 seconds and mean chunk RTF was 0.530646.

The Ascend operator is now genuinely shape-aware at both competition widths.
Its host tiler accepts `[2,50,512]` and `[2,64,512]`, the Torch binding sizes
the packed output from the input shape, and the Omni graph reshapes projected
values back to the traced input shape. Startup now compiles the fused causal
Conv+MLP megagraph at the configured width. Exact device tests passed all 12
combinations of width 50/64, FP16/FP32/BF16, and channel-major/cache-major
state. The live width-64 graph compiled and replayed; HiFT widths 50 and 72
both passed with maximum absolute drift `0`.

The official TTS wrapper also now registers MiniCPM-o 4.5's Omni chat request,
separates the registry model ID from the name advertised by a local server,
and removes argparse's literal `--` separator before forwarding benchmark
options. This prevents a local checkpoint name mismatch from silently turning
an intended run into HTTP 404 failures.

Fresh accepted-control and fused width-64 processes ran the same 32 fixed
English Seed-TTS rows three times with three warmups, concurrency one, seed
zero, temperature zero, and CFM6. Every measured run completed 32/32 with zero
failures and 100% continuity, preserving 4,801 input tokens, 480 output tokens,
3,362,880 audio frames, and 140.12 seconds of audio. The table reports the
median run value; lower is better except for throughput.

| Metric | Accepted 25-frame control | Fused 25/32-frame schedule | Change |
| --- | ---: | ---: | ---: |
| Serving duration | 41.238 s | 41.228 s | -0.02% |
| Request throughput | 0.7760 req/s | 0.7762 req/s | +0.02% |
| Mean / median / P99 E2E | 1,287.79 / 1,305.49 / 1,708.94 ms | 1,287.67 / 1,283.26 / 1,801.70 ms | -0.01% / -1.70% / +5.43% |
| Mean / median / P99 TTFT | 313.59 / 314.18 / 453.42 ms | 314.64 / 319.61 / 449.42 ms | +0.33% / +1.73% / -0.88% |
| Mean / median / P99 audio TTFP | 752.02 / 753.13 / 904.65 ms | 777.73 / 782.77 / 942.91 ms | +3.42% / +3.94% / +4.23% |
| Mean / median / P99 chunk RTF | 0.317468 / 0.146430 / 1.017063 | 0.348741 / 0.152492 / 1.047937 | +9.85% / +4.14% / +3.04% |

The new kernel removes the catastrophic fallback cost: relative to the
unfused 32-frame screen, fused median duration improves 38.11%, throughput
improves 61.57%, TTFP improves 22.61%, and mean chunk RTF improves 34.28%.
Against the accepted 25-frame profile, however, aggregate duration is flat
while every TTFP and chunk-RTF gate regresses by more than two percent. The
kernel capability remains available for future shapes, but the 32-frame
schedule is rejected and is not inherited by the accepted profile.

A separate composition screen showed that cache-major causal state must not
be combined with the accepted all-step AdaLN profile. One fail-fast 32-row run
completed without structural drift but took 67.567 seconds with mean audio
TTFP 1,042.43 ms and mean chunk RTF 0.494745. Its experimental profile now
explicitly disables wide AdaLN rather than accidentally replacing inherited
connector options.

Raw artifacts are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-all-step-cache-major-20260820
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-chunk32-20260820
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-chunk32-width64-20260820
```

Artifact checksums:

```text
c6a8cdf8fc059530fb132f40e533941b63d7778e8e683f853aed67ba700dc3c0  candidate/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260819-182758.json
4252e5a5b1a4837940089043d352a1de7baaae4e2ab791315070a10bb41de112  candidate/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260819-183053.json
45f35a3a611df007a169c92c841fa50078bd5f980ccd76a26d7bf19056fdad93  candidate/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260819-183155.json
87018b83e590fb2449a81d8e81e31466608dcc4c937c57dc8f9586cced568a1d  /tmp/lunanexa-chunk32-width64-service-7.log
```

## Promoted all-step final-layer AdaLN projection

The accepted all-step AdaLN graph projected the six fixed CFM timesteps for
all 16 DiT blocks, but `FinalLayer` still repeated its independent
512-to-1024 time projection once per estimator step. The promoted graph packs
that 17th projection below the existing 73,728 block rows. It returns the
block bank and six final-layer modulation rows from the same Cube GEMM. The
final normalization and 512-to-80 output projection remain eager; compiling
those small operations made the isolated boundary 42.28% slower.

The real-checkpoint NPU 1 screen measured the established block graph plus six
complete eager final layers at 825.089 us median. Reusing the enlarged graph's
final modulations while keeping the rest of `FinalLayer` eager took 687.107 us,
a 16.72% latency reduction (1.201x speedup). Moving the whole final layer into
a second graph took 1,173.949 us and was rejected. Maximum final-output drift
was `7.15e-7`. Live startup independently measured block drift `0` and final
modulation drift `8.34e-7`, below the fail-closed `1e-6` limit. A failure of
the enlarged graph disables only the final-layer extension and retains the
already accepted block-only graph.

Fresh candidate and accepted-control processes ran the same 32 fixed English
Seed-TTS rows three times after two warmups, at concurrency one, seed zero,
temperature zero, and CFM6. Every run completed 32/32 with zero failures and
100% continuity, preserving 4,801 input tokens, 480 output tokens, 3,362,880
audio frames, and 140.12 seconds of audio. The table reports the median run
value from each side. Lower is better except for throughput.

| Metric | Accepted control | Wide final AdaLN | Change |
| --- | ---: | ---: | ---: |
| Serving duration | 43.703 s | 43.574 s | -0.30% |
| Request throughput | 0.7322 req/s | 0.7344 req/s | +0.30% |
| Mean / median / P99 E2E | 1,365.29 / 1,408.63 / 1,830.56 ms | 1,361.37 / 1,391.91 / 1,828.12 ms | -0.29% / -1.19% / -0.13% |
| Mean / median / P99 TTFT | 314.03 / 316.16 / 460.12 ms | 309.89 / 311.51 / 446.83 ms | -1.32% / -1.47% / -2.89% |
| Mean / median / P99 audio TTFP | 778.09 / 775.11 / 920.83 ms | 773.71 / 775.56 / 917.14 ms | -0.56% / +0.06% / -0.40% |
| Mean / median / P99 chunk RTF | 0.334470 / 0.183147 / 1.075928 | 0.333051 / 0.184201 / 1.068024 | -0.42% / +0.58% / -0.73% |

All three paired duration runs improve, as do throughput and every primary
mean and tail gate. Median TTFP and median chunk RTF regress by less than one
percent and remain inside the two-percent guard. The candidate is therefore
enabled in the accepted prompt-width profile. The full cached Seed-TTS,
Daily-Omni, and Video-MME qualifications carry forward because the change is
bounded by a real-checkpoint parity gate and leaves Stage-0 answers and output
structure unchanged; all three suites are still rerun at the final release
qualification.

Two nearby ideas were closed before service A/B. Factoring the six 320-to-512
DiT input projections into one invariant 240-channel projection plus six
80-channel projections improved 230.375 us to only 221.708 us (3.91%) and
introduced `0.0078125` maximum BF16 drift; its graph form took 802.082 us.
Prepacking the five immutable HiFT F0 Conv1d weights as resident FRACTAL_Z was
bit-exact but improved the full graph only 213.624 us to 211.168 us (1.16%).
Both are retained only as reproducible benchmark screens.

Raw artifacts are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-wide-final-adaln-20260820
```

Artifact checksums:

```text
5a4603f093fee36e61d6cc0760f337f4cd54fd1a646a61c38126e38741dd8000  control/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260820-025406.json
898aa191a5c4d31959a1b76a159f9d791fdaf291b55a1e58d629227aa357642a  control/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260820-025554.json
474ad8b1956ce4df308fbaa516c37c645e060ee0357f79c315919b7d6c981dfb  control/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260820-025658.json
d084cf39a7ba8f6eb64cbd6d0d05e7226446f563a071df183751bc7dfe03eb17  candidate/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260820-030411.json
575b4adc19eed5abe2914174e4001a11222a2545771924eba2e2cb62fe306225  candidate/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260820-030602.json
25cbb569e702fc8287055ac92c6a7437acb83e32c15df39ceda09286ebe38cb3  candidate/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260820-030708.json
f220ccc094043abd3002dadcca23fdbe82ddcb142e1a4b5b9880fd279d148a1b  candidate-service.log
```

## Promoted final AdaLN Addcmul lowering

After the all-step final projection landed, the remaining six-step CFM
epilogue divided into 373.998 us of final LayerNorm/modulation and 488.405 us
of output projection, CFG guidance, and Euler update. The canonical modulation
`norm * (1 + scale) + shift` issues three eager elementwise operations after
LayerNorm. Reassociating the same expression as
`addcmul(norm + shift, norm, scale)` removes one launch and one intermediate
without hiding the final 512-to-80 Cube GEMM from the native runtime.

The real-checkpoint NPU-1 harness measured six canonical modulations at
373.998 us and the Addcmul form at 308.726 us, a 17.45% reduction. The full
six-step epilogue improved from 849.465 us to 788.316 us, or 7.20%. Maximum
final-state drift was `2.38e-7`. Live startup separately measured `4.77e-7`
maximum output drift and enables the path only below a fail-closed `1e-6`
bound. A runtime exception disables only Addcmul and immediately retries the
canonical AdaLN expression.

Two broader alternatives were rejected in the same harness. Compiling the
complete final-layer, CFG, and Euler boundary took 890--913 us instead of
814--875 us. Moving CFG before the output projection was mathematically
linear and bounded to `5.36e-7`, but the smaller batch-one GEMM lost Cube
efficiency and did not beat the canonical path.

Fresh candidate and accepted-control processes ran the same 32 fixed English
Seed-TTS rows three times after two warmups, at concurrency one, seed zero,
temperature zero, and CFM6. Every run completed 32/32 with zero failures and
100% continuity while preserving 4,801 input tokens, 480 output tokens,
3,362,880 frames, and 140.12 seconds of audio. Every candidate duration
(43.858--44.764 seconds) was lower than every control duration
(45.592--46.318 seconds). The table reports three-run medians; lower is
better except for throughput.

| Metric | Accepted control | Final Addcmul | Change |
| --- | ---: | ---: | ---: |
| Serving duration | 45.859 s | 43.880 s | -4.32% |
| Request throughput | 0.6978 req/s | 0.7293 req/s | +4.51% |
| Mean / median / P99 E2E | 1,432.74 / 1,461.83 / 1,945.93 ms | 1,370.82 / 1,398.01 / 1,847.38 ms | -4.32% / -4.37% / -5.06% |
| Mean / median / P99 TTFT | 316.50 / 320.61 / 453.20 ms | 317.49 / 322.12 / 454.62 ms | +0.31% / +0.47% / +0.31% |
| Mean / median / P99 audio TTFP | 802.41 / 800.09 / 966.63 ms | 783.13 / 788.72 / 933.13 ms | -2.40% / -1.42% / -3.47% |
| Mean / median / P99 chunk RTF | 0.334418 / 0.329847 / 0.435565 | 0.319463 / 0.316592 / 0.419713 | -4.47% / -4.02% / -3.64% |

TTFT is produced by Stage 0 before this Stage-2-only path executes, and its
three gates remain inside the two-percent variance guard. Every Stage-2 and
end-to-end gate improves, so `npu_dit_final_addcmul` is enabled in the
accepted prompt-width profile. The complete Code2Wav suite passed 89/89 and
all relevant 910C configuration tests passed 29/29. Full Seed-TTS WER/SIM,
Daily-Omni, and Video-MME remain part of final cumulative qualification; this
bounded downstream rewrite cannot change the Stage-0 answers scored by the
latter two suites.

Raw artifacts are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-final-addcmul-20260820
```

Artifact checksums:

```text
54afb25588f5954e29948a725e07e90d4bd93120253223e648b9ecaa233553a5  control/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260820-052041.json
c9ae1f377a90cb707d23e5f1631083d8132119b34d552a620e0dfec555abbe75  control/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260820-052239.json
b5ac4d078e608eab1e9ecd0e017274621f316ac529c7299d74a935a83acc5158  control/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260820-052350.json
ede3dacc001ad97dd7136e0e645d9e63b43638a253dbd5db6d2247d7dfd99980  candidate/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260820-051116.json
df4547dd197ee25937c00ea127af8468ba794ec6fb21393181dc85da23d1d68f  candidate/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260820-051305.json
09baabb4aa9a80f50bc887437c1631767380a1099aa22b2fb70db14ffacf4d1a  candidate/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260820-051410.json
d66b3a364dafd85e3de0cdfd008319adce0ef655b0fc44609db4b883018c4f1b  candidate-service.log
e6861eb526deed32f1c0b8741248824e3ba159436727d731c7535669c281283c  control-service.log
```

## Rejected mixed Vector/Cube final AdaLN kernel

The next kernel experiment fused the steady-width final affine-free
LayerNorm, AdaLN shift/scale, and 512-to-80 projection into one AscendC mixed
kernel. Ten Vector cores normalize and modulate the fixed FP32 `[2, 50, 512]`
activation into workspace; five Cube cores then project 16 output channels
each and add the 80-channel bias. Prompt and tail shapes continue through the
accepted Addcmul fallback.

An alternating 15-trial NPU-1 microbenchmark used 100 warmups and 200 timed
iterations per trial. The accepted LayerNorm + Addcmul + linear boundary took
70.876 us at the median, while the fused kernel took 29.773 us: a 2.3805x
isolated speedup and about 246.6 us projected saving over six CFM steps. The
synthetic parity fixture measured `0.00175923` maximum and `0.00017176` mean
absolute drift. The real-checkpoint startup gate measured `0.000928760`
maximum and `0.000284755` mean drift, inside the fail-closed `0.002` / `0.0005`
bounds. Any shape, dtype, operator, or parity failure disables only this
kernel and immediately retries the accepted Addcmul path.

The service result did not follow the isolated result. Fresh candidate and
accepted-control processes ran the same 32 fixed English Seed-TTS rows three
times after three warmups, at concurrency one, seed zero, temperature zero,
and CFM6. Every valid run completed 32/32 with zero failures and 100%
continuity while preserving 4,801 input tokens, 480 output tokens, 3,362,880
audio frames, and 140.12 seconds of audio. The table reports the median run
value from each side; lower is better except for throughput.

| Metric | Accepted control | Fused final AdaLN | Change |
| --- | ---: | ---: | ---: |
| Serving duration | 43.835 s | 45.401 s | +3.57% |
| Request throughput | 0.7300 req/s | 0.7048 req/s | -3.45% |
| Mean / median / P99 E2E | 1,369.37 / 1,403.54 / 1,826.46 ms | 1,418.46 / 1,439.42 / 1,907.96 ms | +3.59% / +2.56% / +4.46% |
| Mean / median / P99 TTFT | 315.30 / 316.13 / 448.59 ms | 317.71 / 320.25 / 450.78 ms | +0.76% / +1.30% / +0.49% |
| Mean / median / P99 audio TTFP | 780.21 / 781.16 / 922.74 ms | 788.91 / 792.47 / 926.93 ms | +1.11% / +1.45% / +0.45% |
| Mean / median / P99 whole-audio RTF | 0.319218 / 0.320959 / 0.430698 | 0.329750 / 0.331605 / 0.428053 | +3.30% / +3.32% / -0.61% |
| Mean / median / P99 chunk RTF | 0.335295 / 0.183195 / 1.026925 | 0.345943 / 0.195240 / 1.025032 | +3.18% / +6.57% / -0.18% |

The candidate fails the two-percent gate on serving duration, mean and median
whole-audio RTF, mean and median chunk RTF, and all E2E aggregates. It is not
enabled in the accepted prompt-width profile. The custom operator and guarded
integration remain available through the explicitly named experimental YAML
for profiler work.

The likely cause is boundary placement, not arithmetic cost. The accepted
LayerNorm, Addcmul, and linear operations remain visible to GE and can overlap
or optimize with neighboring work. The ACLNN custom operator is an opaque
synchronous boundary with a 200 KiB workspace round trip and a whole-device
Vector-to-Cube barrier on every CFM step. Its microbenchmark removes Python
launches in isolation, but the live pipeline loses more scheduling freedom
than those launches cost. A future retry must fuse a larger producer-consumer
region (for example final projection through CFG/Euler) or expose the operator
to the graph compiler instead of inserting another eager ACLNN island.

Raw artifacts are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-fused-final-adaln-20260820
```

Artifact checksums:

```text
bde59d8cb81a38b15c8b968455e72db8f1e08255e13a9d1c679519b16d73bd30  control/control-run1.json
3165ed79423b95cb0b7b8c32225a08ce83abceeaaaf9c763317c3b58502edc6d  control/control-run2.json
8ee302bd74378182a15aca4565c1a76d8e23fbb6f30f9b696853ac826665aedd  control/control-run3.json
de61cff0e06ed34c47ba14dfc120bc9eefd6de73b65739fd829fc013e9fa8341  candidate/candidate-valid-run1.json
62c81e4ba12390e17b2367a8621ec8e85e9d9db391fd9ce4c65f9c57ced73780  candidate/candidate-valid-run2.json
c900e841f9dc6c01701a812b3af03656cf75f083331ad3f1314bbc1ed2effc09  candidate/candidate-valid-run3.json
835ce8779f1a7493c0c4e6193f99073069b6cfe99d9dfe735a8ea025b18160be  candidate-service.log
f0c3c2de53a1da9a7dc55d227db8d9626b68a7f8ec8115ab5b8b933a563d1137  control-service.log
```

## Rejected GE-visible last-block-to-Euler megagraph

The next experiment removed the opaque ACLNN island and instead enlarged the
already accepted TorchAir/GE replay. For the steady `[2,50,512]` CFM shape,
the last DiT block's causal-pack Conv, MLP residual, affine-free final
LayerNorm, AdaLN Addcmul, 512-to-80 output projection, CFG reduction, and
Euler state update execute as one static graph. This removes a Python/ACLNN
boundary without hiding operations from GE. Prompt and tail widths retain the
accepted split path.

The implementation is guarded by
`npu_dit_last_block_final_euler_graph` (or
`VLLM_OMNI_MINICPMO45_NPU_DIT_LAST_BLOCK_FINAL_EULER_GRAPH`) and the explicit
profile:

```text
vllm_omni/deploy/minicpmo_4_5_2npu_910c_cfm6_dit_last_block_final_euler_graph_experimental.yaml
benchmarks/scripts/bench_minicpmo_dit_last_block_final_euler.py
```

It requires the accepted causal-pack Conv+MLP, wide final AdaLN, and final
Addcmul profile. Incompatible layouts fail closed. Startup compares the new
graph with the accepted graph using loaded model tensors, rejects non-finite
outputs, and enforces `0.005` maximum / `0.0005` mean state drift plus
`0.005` cache drift. A runtime failure disables only this extension and
immediately replays the accepted Conv+MLP and final path.

The corrected real-checkpoint FP32 NPU-1 harness used nine alternating trials,
20 warmups, and 100 timed iterations. It reduced the fused region from
358.774 us to 298.390 us, a 1.2024x speedup. State maximum/mean drift was
`5.96e-8` / `6.17e-9`, and cache drift was zero. The live startup gate measured
`1.04e-7` maximum and `1.40e-8` mean state drift with zero cache drift, then
logged that the last-block-to-Euler replay was active. A BF16 screening run
also improved 343.948 us to 240.365 us (1.4309x), but is not the serving dtype
and is recorded only to prevent that result from being mistaken for the live
projection.

The isolated FP32 saving is only 60.384 us per CFM step. Even across six steps
and several streamed chunks, its projected request saving is around one to two
milliseconds, far below the roughly 1.3-second end-to-end request time. The
larger graph therefore needed a live promotion result; the microbenchmark was
not sufficient evidence.

Fresh candidate and accepted-control processes each ran the same 32 fixed
English Seed-TTS rows three times after three warmups, at concurrency one,
seed zero, temperature zero, and CFM6. Every run completed 32/32 with zero
failures and 100% continuity while preserving 4,801 input tokens, 480 output
tokens, 3,362,880 audio frames, and 140.12 seconds of audio. The table reports
componentwise three-run medians; lower is better except for throughput.

| Metric | Accepted control | Last-block-to-Euler | Change |
| --- | ---: | ---: | ---: |
| Serving duration | 41.472 s | 43.859 s | +5.76% |
| Request throughput | 0.7716 req/s | 0.7296 req/s | -5.44% |
| Mean / median / P99 E2E | 1,295.62 / 1,326.78 / 1,745.15 ms | 1,370.09 / 1,407.12 / 1,838.94 ms | +5.75% / +6.05% / +5.37% |
| Mean / median / P99 TTFT | 315.60 / 316.72 / 455.62 ms | 322.79 / 330.39 / 462.01 ms | +2.28% / +4.32% / +1.40% |
| Mean / median / P99 audio TTFP | 757.68 / 768.45 / 916.55 ms | 786.40 / 794.05 / 928.96 ms | +3.79% / +3.33% / +1.35% |
| Mean / median / P99 whole-audio RTF | 0.303011 / 0.305925 / 0.400238 | 0.319225 / 0.319531 / 0.426683 | +5.35% / +4.45% / +6.61% |
| Mean / median / P99 chunk RTF | 0.319334 / 0.142588 / 1.021162 | 0.335713 / 0.181280 / 1.050555 | +5.13% / +27.14% / +2.88% |

The candidate fails every primary Stage-2 and end-to-end promotion gate and is
not enabled in the accepted prompt-width profile. Because the expected saving
is much smaller than process-level variance, this experiment does not prove
that the fused graph itself causes the entire five-percent difference. It does
prove that absorbing only the last block's epilogue cannot deliver a stable,
measurable serving win on this stack. The accepted service remains active.

The next lower-layer attempt should target a region with an order-of-magnitude
larger budget: multiple DiT blocks in one GE replay, attention-to-Conv producer
fusion without cache-layout conversions, or the complete six-step CFM loop.
Each needs live-FP32 isolated accounting before another service A/B.

Raw artifacts are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-last-block-final-euler-20260820
```

Artifact checksums:

```text
0601fa1e7070a8895f00ee35558eaea85a737a64841c66a55924e819b703e5a7  candidate/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260820-073717.json
71b4f4bcdbf61d69e65b41416beb2a0452c73cb108a0b734287ca4c81faa084d  candidate/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260820-073914.json
2e727366836606998ae302cb38c08865c4650807c5bbd2269f5033cb3af057ca  candidate/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260820-074023.json
0af451c226f974bf64ed246a6279f475f693fff8d3361abd3cb05c00f1912440  control/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260820-074808.json
4b18afd3e0743df5abcf5336fe2ae9aec1cdab8a55e06b6b528310ae3b2665a2  control/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260820-075001.json
b5db1c13a83830d1f2f18ce1ebea061f62bdba10aab99a1c7b3f2385f903c95e  control/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260820-075107.json
597260dc2c65fd1986ac75b0cd34258df373ce301e2e865e483ab573d88039cb  candidate-service.log
6d01e1d090d75b0de089fa4bf4fb448905fd2b1690747c960fd26fc2ff065957  control-service.log
3dc04960a7511607d810a99c5f1d4aaacaba6683ccf5762d8e6b5136365e63ce  isolated-fp32.log
ba28ab6f3352c9765db4b8e6b5f3c43a4e3042de1afd5ad7d3e9fcc7c67c4fd0  isolated-bf16.log
```

### Fix: fail-closed device-time usefulness gate

The rejected graph originally used correctness as its only startup promotion
gate. That was insufficient: its isolated win was genuine but too small to
survive the live scheduling boundary. The implementation now times the loaded
checkpoint's accepted and fused regions with NPU events after compilation,
using five alternating trials of 20 replays. Promotion requires both at least
`1.10x` speedup and at least 200 us absolute saving per CFM step. Failure of
either condition discards the fused callable before serving traffic; the
accepted Conv+MLP, final Addcmul, CFG, and Euler path remains active.

On the Atlas 800I A3 / 910C host, the startup gate measured 366.860 us for the
accepted region and 287.193 us for the fused region: `1.2774x`, but only
79.667 us saved. It therefore rejected the graph on the absolute-headroom
criterion. This fixes the regression without pretending that a microbenchmark
win is a deployable serving win.

An adjacent, fully warmed 12-row Stage benchmark compared the active fused
process with a fresh process where the usefulness gate selected the accepted
fallback. Both sides preserved 1,804 input tokens, 183 output tokens, 1,252,800
audio frames, 52.2 seconds of audio, zero failures, and 100% continuity.

| Metric | Active fused graph | Gated accepted fallback | Change |
| --- | ---: | ---: | ---: |
| Serving duration | 17.289 s | 16.594 s | -4.02% |
| Request throughput | 0.6941 req/s | 0.7231 req/s | +4.19% |
| Mean / median / P99 E2E | 1,440.19 / 1,411.47 / 1,784.75 ms | 1,382.38 / 1,404.97 / 1,793.04 ms | -4.01% / -0.46% / +0.46% |
| Mean Stage-2 generation time | 1,437.46 ms | 1,385.04 ms | -3.65% |
| Mean / median / P99 TTFT | 320.54 / 318.09 / 454.53 ms | 319.89 / 311.09 / 467.23 ms | -0.20% / -2.20% / +2.79% |
| Mean / median / P99 audio TTFP | 803.80 / 789.80 / 954.73 ms | 800.62 / 781.50 / 976.46 ms | -0.40% / -1.05% / +2.28% |
| Mean / median / P99 whole-audio RTF | 0.338513 / 0.324029 / 0.415314 | 0.323665 / 0.324948 / 0.377660 | -4.39% / +0.28% / -9.07% |

The tail TTFT/TTFP changes are upstream variance: Stage 2 cannot affect text
TTFT, and both are based on only 12 rows. The causal signal is the Stage-2
mean plus serving-duration/throughput/mean-RTF recovery. This run is used only
to validate the fallback decision, not to promote a new speed claim.

The gated process then completed three full 32-row Seed-TTS runs. Every run
preserved the official structural signature: 4,801 input tokens, 480 output
tokens, 3,362,880 frames, 140.12 seconds of audio, 32/32 successes, zero
failures, and 100% streaming continuity. Componentwise medians were 44.949 s
duration, 0.7119 requests/s, 1,404.25 ms mean E2E, 315.21 ms mean TTFT,
801.52 ms mean TTFP, 0.328143 mean whole-audio RTF, and 0.345399 mean chunk
RTF. These are fallback validation results; they are not compared with the
earlier process epoch as a performance A/B.

The focused Code2Wav and 910C configuration selection completed 182/182 tests.
The graph is still available for future larger boundaries, but this exact
last-block region can no longer become a live regression on hardware where it
lacks enough absolute device-time budget.

Fix artifacts are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-last-block-fix-20260820
```

Selected checksums:

```text
be038c461e02e329794e9cb05c6a832c393e18add0d52e5df704e56ae436334b  gated-service.log
7bffb228c4579a6ff55d0ddc4184bb1a639dc43c5073d68bbc140a1ece6a0b46  candidate-stage-warm/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260820-094558.json
c10431541585f505a4f84be318efff1ed5785e4ec5a402463022fd7e09b4a63a  gated-stage-warm/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260820-095921.json
a2de666b40257368c34ddebd15b08e1640b670091d14dc4025cf05cde1bafbb1  gated-32/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260820-100016.json
fd03982e17a31d8b347c33afd57c61b9940e26386008b7c7f7a07075165f5c31  gated-32/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260820-100149.json
b4fd1f38c896e07e547e1281b43c536ebe48a65a36293d9f56c97c5009d5339d  gated-32/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260820-100310.json
```

### Experimental homogeneous-BF16 CFM precision island

The next lower-layer experiment moved the complete six-step CFM numerical
island to BF16 on Ascend 910C: the DiT estimator, random-noise state, cached
timestep/delta tensors, CFG reduction, and Euler recurrence now share one
dtype. The completed mel is converted once at the FP32 HiFT boundary. The
flow encoder, prompt extraction, HiFT vocoder, and public waveform contract
remain FP32.

This design replaces a rejected selective-BF16 prototype. Keeping the
estimator in BF16 while casting every CFM step back to an FP32 Euler state
completed 32/32 requests with exact structural parity, but took 71.57 seconds
against its adjacent 67.14-second FP32 control: 6.60% slower. The repeated
dtype boundaries also disabled a larger homogeneous graph signature. That
prototype remains expressible for diagnosis, but is not the deploy profile.

The homogeneous mode is opt-in through `npu_dit_compute_dtype: bf16` plus
`npu_cfm_integration_dtype: bf16`. It fails closed to FP32 integration when
the requested integration dtype does not match the active estimator dtype,
and converts the estimator back to FP32 if module conversion fails. On real
hardware the startup log confirmed `estimator=torch.bfloat16`,
`CFM integration=torch.bfloat16`, and `HiFT=float32`. Existing graph drift
gates remained active. In particular, the final Addcmul rewrite measured
0.0078125 maximum drift against its 0.000001 bound and correctly retained the
canonical AdaLN path instead of loosening the gate.

The isolated checkout completed the entire focused Code2Wav suite, including
the new precision-boundary cases: 100/100 passed. The hardware run used the
same fixed 32 English Seed-TTS rows, three warmups, concurrency one, seed zero,
temperature zero, and CFM6 on both sides. Both sides completed 32/32 with zero
failures, 100% streaming continuity, 4,801 input tokens, 480 output tokens,
3,362,880 frames, and 140.12 seconds of audio.

The shared host showed substantial epoch variance: an earlier adjacent FP32
control took 67.14 seconds, while the final immediate quality-paired FP32
control took 46.87 seconds. The table therefore uses the faster final control
as the conservative comparison. Lower is better except throughput.

| Metric | FP32 control | Homogeneous BF16 | Change |
| --- | ---: | ---: | ---: |
| Serving duration | 46.868 s | 45.331 s | -3.28% |
| Request throughput | 0.6828 req/s | 0.7059 req/s | +3.39% |
| Mean / median / P99 E2E | 1,463.92 / 1,488.74 / 1,999.56 ms | 1,416.00 / 1,450.05 / 1,898.76 ms | -3.27% / -2.60% / -5.04% |
| Mean / median / P99 TTFT | 327.75 / 329.94 / 468.25 ms | 316.61 / 315.49 / 459.58 ms | -3.40% / -4.38% / -1.85% |
| Mean / median / P99 audio TTFP | 815.03 / 810.22 / 975.74 ms | 786.90 / 785.87 / 937.00 ms | -3.45% / -3.00% / -3.97% |
| Mean / median / P99 whole-audio RTF | 0.340990 / 0.340272 / 0.437862 | 0.329556 / 0.325919 / 0.416748 | -3.35% / -4.22% / -4.82% |

The same outputs ran through Whisper-large-v3 WER and WavLM-base-plus SIM.
Both evaluators processed all 32 rows with zero PCM, ASR, or embedding
failures. WER is reported as a fraction below and percentage-point changes are
computed on the corresponding 0-100 scale.

| Accuracy metric | FP32 control | Homogeneous BF16 | Accuracy change |
| --- | ---: | ---: | ---: |
| Mean / median WER | 0.016588 / 0 | 0.016588 / 0 | 0.00 pp |
| Mean / median WavLM SIM | 0.845234 / 0.851134 | 0.844850 / 0.851195 | -0.038 pp / +0.006 pp |

This clears the 2-percentage-point screening gate with a large margin and is
accepted as an experimental profile. It is not promoted into the default
910C profile yet: the 32-row WavLM score is the repository's documented proxy,
not the competition's fine-tuned UniSpeech/WavLM-SV protocol, and the full
official 1,088-row Seed-TTS export/evaluation still remains a release gate.
Daily-Omni and Video-MME are unaffected by this Stage-2-only numerical change,
but their cumulative competition runs also remain required before submission.

Raw artifacts are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-bf16-20260820
```

Selected checksums:

```text
bec40f62c441fd307b05115e23af47874c7c333d4e49db699a7626c032bcdd93  homogeneous-quality32/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260820-150610.json
0813cc8e834bb4a53c6051950442ac3fd4957208d26363c504ada566a2698683  control-quality32/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260820-154644.json
07035b99ce23921a8542d814a27b4681374290912d81133a63d1e4fd2b906101  homogeneous-bf16-service.log
454db6ee07d860572fed483370736aa2900ca5634b3ca84d04a43720985a7b82  final-control-service.log
```

### Fixed-address estimator cache slabs

The next cache-layer candidate removes request-time estimator-cache growth
without changing MiniCPM-o's attention history. The upstream attention path
orders its cache as `[new chunk, previous cache]`; therefore treating the
first prompt-width region as immutable would change model semantics. The
implemented representation instead owns, per concurrency-one request:

- one retained six-step x 16-block KV slab with capacity `prompt + 100`;
- one separate append/output slab with capacity `prompt + 150`;
- a logical length rather than a changing allocation;
- two reusable CNN-cache banks; and
- direct CFM cache outputs into those workspaces.

After each decode, the output slab is compacted into the distinct retained
slab using the exact existing rule: preserve the first prompt-width frames and
the newest retained 100-frame tail. Distinct source and destination buffers
avoid undefined overlapping copies. Prompt/cache-fill/final shapes stay eager.
The focused suite verifies exact audio and all flow-cache tensors across four
chunks, fixed storage addresses, and the overflow retention order. Together
with the deploy-inheritance gate, 104/104 focused tests passed.

The adjacent hardware screen used the homogeneous-BF16 profile on both sides,
the same first 12 shuffled English Seed-TTS rows, three warmups, concurrency
one, seed zero, temperature zero, and CFM6. Both runs completed 12/12 with zero
failures, 100% streaming continuity, 1,804 input tokens, 183 output tokens,
1,252,800 frames, and 52.20 seconds of audio. Lower is better except
throughput.

| Metric | BF16 control | Fixed slabs | Change |
| --- | ---: | ---: | ---: |
| Serving duration | 18.595 s | 17.835 s | -4.08% |
| Request throughput | 0.6453 req/s | 0.6728 req/s | +4.26% |
| Mean E2E | 1,549.17 ms | 1,485.89 ms | -4.08% |
| Mean TTFT | 329.73 ms | 320.54 ms | -2.79% |
| Mean audio TTFP | 832.49 ms | 818.93 ms | -1.63% |
| Mean whole-audio RTF | 0.362937 | 0.349713 | -3.64% |
| Mean / median chunk RTF | 0.390172 / 0.224625 | 0.378519 / 0.203973 | -2.99% / -9.19% |
| P99 chunk RTF | 1.206069 | 1.224779 | +1.55% |

The service logged `retained=402, append=452` and direct stacked CFM cache
outputs. The primary means all improved, but the small 12-row chunk-P99 screen
did not. Fixed slabs therefore remain an experimental speed candidate pending
a 32-row repeated tail gate; they are not silently promoted to the default.
Because the cache transformation is mathematically exact, it reuses the
already-qualified homogeneous-BF16 accuracy boundary, but the full official
Seed-TTS and cumulative Daily-Omni/Video-MME release gates still apply.

A second profile enabled one steady width-50/cache-402 NPUGraph executable and
two captured output slots. Capture failed closed on the real 910C stack:
CosyVoice's causal Conv1d lowered to the legacy ACLop Conv2D path, which cannot
run during NPU stream capture. The resulting 17.81-second run is eager fallback
data, not a graph result. This closes another attempt at raw full-loop capture;
the next implementation must make the convolution graph-visible through
TorchAir/GE static compilation or a converter, rather than retrying
`allow_internal_format=False`.

Raw artifacts are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-fixed-kv-20260821
```

Selected checksums:

```text
be15a479c86c9b7de7367328b6444c928d95abb503442760e758071da8679101  control/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260820-164430.json
f6a443e5bee4043dfdc04c54a301ac7e2d98a3971df372906a4344f2bcd9eb65  fixed/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260820-165055.json
0fc109f1edcfa9a7692e34b09759afa8e38f2cd3fb3bce488b97595ea79f0b23  graph/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260820-165747.json
7ca1bbba1f4a7e7b0325c7934e958fbdbe3e612ede78200978245c0b18a7a89c  fixed-service.log
a31a74e7dcc243fc0c9a1bfbcdfeedb8c19e4a505578eaf4bfe159afd5090691  graph-service.log
```

### Planar fixed-address estimator K/V slabs

The next cache-layout candidate keeps the accepted fixed-capacity ownership
model, but replaces the packed last-dimension `[K | V]` representation with
independent K and V planes:

```text
[six CFM steps, 16 blocks, K/V=2, CFG batch, heads, time, head dimension]
```

At each block boundary, the K and V histories are now independently
contiguous and the projected current K/V tensors write directly into their
final append planes. The SDPA inputs no longer depend on strided halves of a
packed 128-wide cache. Logical length, prompt-plus-100 retention, the separate
prompt-plus-150 output workspace, and the two CNN banks remain unchanged.
The accepted split preamble/attention/Conv+MLP path stays graph-visible; the
previously rejected full-block and full-stack graphs are intentionally not
selected for planar state. Unsupported or batched cases convert to the exact
legacy representation rather than widening the experimental boundary.

The focused tests cover environment/config selection, projected-attention
parity, four-chunk audio and cache parity, fixed storage addresses, and
contiguous per-block K/V planes. The full MiniCPM-o model file completed
107/107 tests; the deploy-profile selection test also passed. The live service
logged both `contiguous planar K/V attention cache active` and
`retained=402, append=452, planar=True`, with no attention-graph fallback.

The hardware screen used two independent service processes for each side.
Every process ran three warmups followed by the same first 12 shuffled English
Seed-TTS rows at concurrency one, seed zero, temperature zero, and CFM6. All
four measured runs completed 12/12 requests with zero failures, 100% streaming
continuity, 1,804 input tokens, 183 output tokens, 1,252,800 waveform frames,
and 52.20 seconds of audio. The table compares the arithmetic mean of the two
runs per side. Lower is better except throughput.

| Metric | Fixed slabs | Planar K/V slabs | Change |
| --- | ---: | ---: | ---: |
| Serving duration | 20.128 s | 17.072 s | -15.18% |
| Request throughput | 0.5962 req/s | 0.7029 req/s | +17.90% |
| Mean / P99 E2E | 1,676.89 / 2,139.18 ms | 1,422.17 / 1,743.27 ms | -15.19% / -18.51% |
| Mean / P99 TTFT | 360.73 / 505.40 ms | 331.95 / 463.75 ms | -7.98% / -8.24% |
| Mean / P99 audio TTFP | 893.89 / 1,045.33 ms | 804.16 / 932.35 ms | -10.04% / -10.81% |
| Mean / P99 whole-audio RTF | 0.391533 / 0.481192 | 0.335330 / 0.455460 | -14.35% / -5.35% |
| Mean / median chunk RTF | 0.413277 / 0.256237 | 0.360917 / 0.185494 | -12.67% / -27.61% |
| P99 chunk RTF | 1.252329 | 1.238023 | -1.14% |

Repeatability was strong: fixed-slab durations were 20.209 and 20.047 seconds;
planar durations were 17.075 and 17.069 seconds. Unlike the earlier isolated
direct-output experiment, the producer and consumer now share the new layout
through the complete steady attention boundary. This removes the strided
packed-cache cost instead of adding another opaque custom-op boundary.

The first 32-row stability attempt exposed one inherited fixed-slab limit that
the 12-row screen did not reach. One final encoder flush produced 54 frames,
requiring cache length 456 while the steady append slab intentionally ends at
452. Raising the slab would waste steady-path HBM and change its static shape.
The implementation now keeps the fixed 50-frame append region and uses a
dynamically sized eager output only for an oversized tail. The returned state
is then compacted into the same fixed retained slab. A regression test forces
this overflow and verifies exact audio/cache parity. The corrected service
logged `required=456, append=452`, took the eager tail path, and completed a
32-row stability run with zero failures and 100% continuity.

The corrected candidate then ran the cached 32-row Seed-TTS quality screen in
explicit Hugging Face offline mode. It completed all 32 rows, preserved 4,801
input tokens, 480 output tokens, 3,362,880 waveform frames, and 140.12 seconds
of audio, and measured 44.301 seconds duration, 0.326667 mean whole-audio RTF,
and 783.88 ms mean TTFP. Whisper-large-v3 WER was identical to the accepted
homogeneous-BF16 result. WavLM-base-plus SIM changed by only -0.010 percentage
points in the mean and -0.004 points in the median.

| Accuracy metric | Accepted homogeneous BF16 | Planar K/V slabs | Accuracy change |
| --- | ---: | ---: | ---: |
| Mean / median WER | 0.016588 / 0 | 0.016588 / 0 | 0.000 pp |
| Mean / median WavLM SIM | 0.844850 / 0.851195 | 0.844747 / 0.851154 | -0.010 pp / -0.004 pp |

The planar profile is accepted as the next experimental speed profile. It is
not silently enabled in the default 910C deployment yet. The local 32-row
accuracy screen clears the 2-percentage-point gate, but the full official
1,088-row Seed-TTS evaluation remains a release gate; cumulative Daily-Omni
and Video-MME validation is still required before competition submission.

Raw artifacts are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-planar-kv-20260821
```

Selected checksums:

```text
fb28e3abc6f6f51d1cc20e24594ec2f150fb4478075de729f096eab03a8b8710  fresh-control/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260821-013340.json
3cf583bc9d45295e7aa888a0c713e9a4801f60929bc23f3c784375500df4a92b  fresh-control-repeat/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260821-013534.json
877c98e5593e4a92a424520cc8cdeaddf23ba5e2b245ddaff5a068ed9a12762c  planar/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260821-012632.json
98816b6ef0e0a64a63c0d6be52e5975cd09c0443a940d90bcf6a18b7bbb00593  planar-repeat/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260821-014206.json
b7f8111c0d5bf81512fb9c9e46228b917f58fce3c8fa17f11ac74ad9fc0cc608  planar-service.log
a2e172276e4b2202813f2cfc6540d5ba914289016955752649db4a03940def50  planar-repeat-service.log
f47a6d72dbe5987a0b37de3026b1a622e25005ccee416f51958825e7fd2d4d44  planar-quality32-fixed/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260821-020017.json
db8e371ac3d5ad9897294a1b16b2ee90fb4c46542d6936de085d79c218ca44dd  planar-tail-fixed-32/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260821-015643.json
5f3c3667b13ddf5bf393be6a6d266f254613fca0edd5230847d2807cdbd57ac7  planar-tail-fixed-service.log
```

## Post-planar layout trace and lower-layer screens

A fresh Stage-2 Torch-NPU profile bracketed one warmed Seed-TTS request on
the corrected homogeneous-BF16 planar profile. This replaces the older
pre-planar operator ranking. The largest device families were:

| Operator family | Calls | Device time | Share |
| --- | ---: | ---: | ---: |
| `MinicpmoCausalConvPack` | 576 | 30.630 ms | 11.11% |
| `Transpose` | 3,160 | 29.114 ms | 10.56% |
| `MatMulV2` | 3,506 | 25.624 ms | 9.30% |
| `TransData` | 2,217 | 22.288 ms | 8.09% |
| `LayerNormV3` | 2,545 | 21.449 ms | 7.78% |
| `FlashAttentionScore` | 480 | 14.553 ms | 5.28% |

Shape aggregation makes the layout budget concrete. The two causal-pack
nodes each ran 288 times at about 53 us, consuming 30.63 ms together. The
largest attention transpose, `[2,8,50,64]` to `[2,50,8,64]`, consumed 5.045
ms. Four prompt-Conv weight conversions from `[512,512,1,3]` NCHW to
`FRACTAL_Z` consumed 6.465 ms. Attention arithmetic is no longer the first
target; causal history packing and producer-consumer layouts are.

The graph-visible fused-QKV screen concatenates each block's immutable Q, K,
and V weights once, then replaces three projections with one 1536-wide GEMM.
It deliberately leaves reshape, transpose, normalization, cache append, and
SDPA visible to GE instead of using the rejected opaque QKV custom op. BF16
output was bit-exact on 910C. Ten alternating-order trials of 200 replays at
`[2,50,512]` measured 117.617 us for the accepted three-GEMM graph and
115.861 us for fused QKV: only 1.015x. This does not justify a serving cycle,
so the feature remains an opt-in diagnostic through
`minicpmo_4_5_2npu_910c_cfm6_dit_bf16_planar_fused_qkv_experimental.yaml`.

The prompt-Conv screen preformatted the two kernel-3 weights as true Ascend
`FRACTAL_Z` tensors once, restoring `allow_internal_format=False` before graph
compilation and replay. It was bit-exact but slower: width 20 changed from
196.417 to 208.549 us (-5.82%), and width 302 changed from 281.503 to 302.668
us (-6.99%). Removing the visible conversions does not compensate for the
less profitable compiled layout, so no serving option was added.

### Fixed planar slabs plus cache-major CNN state

The older cache-major causal kernel is 2.31x faster in isolation, but it could
not previously share the new fixed slabs: setup stored CNN state as
`[batch,channels,taps]`, while steady replay requires
`[batch,taps,channels]`, causing the fixed output shape to disagree. The slab
implementation now records its CNN layout, converts once at setup, writes
steady width-50 results directly into alternating cache-major banks, and
converts exact eager prompt/tail output before compaction. Attention slabs
remain planar and fixed-address. The behavior is opt-in through
`minicpmo_4_5_2npu_910c_cfm6_dit_bf16_planar_cache_major_experimental.yaml`.

Focused exactness/layout tests and profile inheritance passed. The full
Code2Wav file passed 112/112 tests and all 34 MiniCPM-o 910C deploy tests
passed. The live service proved the intended path with all three messages:

```text
MiniCPM-o contiguous planar K/V attention cache active
MiniCPM-o NPU cache-major Conv+MLP megagraph replay active
MiniCPM-o fixed estimator KV slabs active: retained=402, append=452, planar=True, cnn_cache_major=True
```

The first 12-row fail-fast run completed 12/12 with zero failures, 100%
continuity, and the accepted structural totals. It was decisively slower than
the two-run planar mean, so no repeat or accuracy budget was spent. Lower is
better except throughput.

| Metric | Accepted planar mean | Planar + cache-major | Change |
| --- | ---: | ---: | ---: |
| Serving duration | 17.072 s | 20.831 s | +22.02% |
| Request throughput | 0.7029 req/s | 0.5761 req/s | -18.05% |
| Mean E2E | 1,422.17 ms | 1,735.53 ms | +22.03% |
| Mean TTFT | 331.95 ms | 352.25 ms | +6.12% |
| Mean audio TTFP | 804.16 ms | 919.03 ms | +14.28% |
| Mean whole-audio RTF | 0.335330 | 0.399066 | +19.00% |
| Mean / median chunk RTF | 0.360917 / 0.185494 | 0.432204 / 0.259577 | +19.75% / +39.94% |
| P99 chunk RTF | 1.238023 | 1.270002 | +2.58% |

The isolated cache-major kernel win again reverses in the composed graph.
This confirms that its boundary prevents more valuable scheduling/layout
decisions; it remains diagnostic-only. The accepted planar profile is
unchanged.

Artifacts are under:

```text
/tmp/vllm-omni-profiles/minicpmo45/planar-kv-stage2
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-planar-layout-20260821
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-planar-cache-major-20260821
```

Selected checksums:

```text
7031808b2269efdb1caa5f7ff3e46385d11513bb4d890746b9f279981b1b71a7  op_statistic.csv
b8d72588ba22a1720da0e8bb0eb5686e74fe560df0f588c0c69f8276a31672e3  kernel_details.csv
9f1e4ec6f03ab9d9934b05341476cffeffe88bc3e2179c488948b3b797873a8c  candidate-run1.json
ede7e84bca3b4759a83bfd07e2da7f80a354f55cdd1e9ef29f27b02254c58247  service.log
```

### Vectorized channel-major causal cache access

The rejected cache-major serving experiment showed that changing the public
CNN-state layout destroys more graph-level optimization than its isolated
kernel saves. The follow-up therefore keeps the accepted
`[batch,channels,2]` layout and removes the scalar work *inside* the existing
`MinicpmoCausalConvPack` boundary. On `ascend910_93`, the kernel now builds a
byte-offset vector once, gathers both historical taps into UB with AscendC
vector operations, and gathers the two final frames into an interleaved UB
buffer before one aligned cache DMA. The public shapes, TorchAir converter,
fixed planar slabs, and complete DiT graph boundary are unchanged.

The first `DataCopyPad` write prototype was rejected because exact validation
found a cache-layout mismatch. The retained gather implementation passed an
independent Torch reference with zero tolerance for both packed history and
returned cache. The reusable microbenchmark now checks that reference for
both public cache layouts before timing, preventing two equally wrong kernels
from validating each other.

With 15 alternating trials of 1,000 launches, the original channel-major
kernel measured 61.875 us median versus 27.037 us for its cache-major path.
The vectorized kernel measured 19.841 us and 19.814 us respectively in the
same screening session: the channel-major throughput cost fell 67.93%, and
the two layouts became equivalent. A later reference-checked rerun while the
full service and profiler exporter were resident measured 26.385 us versus
26.257 us, again showing no material queued-throughput penalty. With one NPU
synchronization after every launch, the retained vector kernel measured
59.990 us for channel-major and 46.110 us for cache-major. The benchmark now
reports both modes so independent-op throughput is not mistaken for serialized
latency.

Two independent serving measurements used the accepted planar profile, three
warmups, the same first 12 shuffled English Seed-TTS rows, concurrency one,
seed zero, temperature zero, and CFM6. Both completed 12/12 requests with zero
failures, 100% continuity, 1,804 input tokens, 183 output tokens, 1,252,800
waveform frames, and 52.20 seconds of audio. The candidate columns below are
the arithmetic mean of both runs. Lower is better except throughput.

| Metric | Accepted planar mean | Vectorized channel-major kernel | Change |
| --- | ---: | ---: | ---: |
| Serving duration | 17.072 s | 16.737 s | -1.96% |
| Request throughput | 0.7029 req/s | 0.7170 req/s | +2.01% |
| Mean / P99 E2E | 1,422.17 / 1,743.27 ms | 1,394.34 / 1,730.34 ms | -1.96% / -0.74% |
| Mean / P99 TTFT | 331.95 / 463.75 ms | 326.29 / 451.62 ms | -1.70% / -2.62% |
| Mean / P99 audio TTFP | 804.16 / 932.35 ms | 800.04 / 926.80 ms | -0.51% / -0.60% |
| Audio throughput | 3.057 audio-s/s | 3.119 audio-s/s | +2.03% |
| Mean / median chunk RTF | 0.360917 / 0.185494 | 0.354065 / 0.179811 | -1.90% / -3.06% |
| P99 chunk RTF | 1.238023 | 1.172903 | -5.26% |

Candidate durations were 16.896 and 16.578 seconds. A fresh Stage-2 capture
confirmed the expected 576 calls and exact graph boundary, but reported
53.044 us per call—nearly the pre-change profiled value—despite the direct
throughput A/B and repeated request-level improvement. These measurements are
not interchangeable: the direct loop reports queued independent-op
throughput, while the detailed profiler serializes and instruments the custom
node. Its value is consistent with the new synchronized microbenchmark. The
trace is therefore retained as a topology and per-call-latency check, while
the two-run end-to-end screen remains the admission result.

A follow-up tried to reduce serialized latency by creating gather offsets only
on prefix-owning vector cores and by hoisting the identical write-offset table
out of the batch loop. It was exact and improved the channel-major micro from
19.372 to 18.748 us in queued mode (-3.22%) and from 59.990 to 54.680 us in
serialized mode (-8.85%). However, two serving runs took 17.658 and 16.067
seconds. Their 16.862-second mean was 0.75% slower than the retained vector
kernel, while P99 E2E rose 6.75%, P99 audio TTFP rose 4.18%, and P99 chunk RTF
rose 1.92%. This deeper hoist was therefore reverted. It is another concrete
case where a better isolated custom-kernel number did not compose into a
better request tail.

Because the transformation is bit-exact, changes no model arithmetic, and
preserves the qualified planar graph and tensor boundary, it inherits the
planar candidate's 32-row WER/SIM result. It does not consume a new accuracy
budget. Full 1,088-row Seed-TTS, Daily-Omni, and Video-MME remain release
gates, as they are for the parent experimental profile.

Artifacts are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-vector-cache-kernel-20260821
/tmp/vllm-omni-profiles/minicpmo45/planar-kv-stage2
```

Selected checksums:

```text
6b631772cee6cf054c184a430ce10ced5d08fa40d22b0b166adc094c57b4f423  candidate-valid-run1.json
c10f5fd1029f4620f246bc6e2876aea8318c0fa80c44da4d59c670bd5b0afa47  candidate-valid-run2.json
4bcf5a2ac8fd886542d2d47c405b67a55234b0b0fd5218d5aa4750b720d14699  service.log
b47bbbf1370d4c88ef3e8be8c020177754b9a187b150a1fe7d402e85aea0c5d4  op_statistic.csv
3b0f665978b6d0ce6241fe2fa61cc751be7b7df0c39aacfb6fce563441c1f2e2  v3-valid-run1.json
c5561b8a4f65a2c58a80cd0d6d43fc61d6756fef6c1742f3cd6f803f7e8d4f75  v3-valid-run2.json
```

## Sequence-major BSH attention candidate

The next layout candidate is implemented behind
`npu_dit_bsh_attention: true`. It extends the accepted homogeneous-BF16,
fixed-planar-slab profile while changing the attention producer/consumer
contract from `[batch, heads, sequence, head_dim]` to
`[batch, sequence, hidden]` across Q/K/V projection, K/V cache append,
Ascend fused attention, and the output projection. Q/K normalization still
uses an internal `[batch, sequence, heads, head_dim]` view, but does not
transpose the head and sequence axes.

The request-owned cache remains six CFM steps by sixteen DiT blocks with
separate K and V planes. Its new physical shape is:

```text
[six steps, sixteen blocks, K/V=2, CFG batch, time, hidden=512]
```

Prompt setup converts once from the checkpoint-compatible packed BHSD cache.
Steady chunks write directly into the fixed BSH append slab. Unsupported
platforms and graph failures convert through the exact legacy cache path, so
the candidate fails closed without changing the public waveform contract.

CPU PyTorch checks prove exact Q/K/V preamble parity, cached SDPA parity,
legacy-cache round trips, stable slab addresses, and correct multi-request CFG
split/stack axes. The deploy overlay is
`minicpmo_4_5_2npu_910c_cfm6_dit_bf16_planar_bsh_attention_experimental.yaml`.
Its connector block deliberately repeats the accepted profile because
top-level connector overlays replace rather than deep-merge their base. A
thin first draft silently discarded CFM6/BF16/fixed-slab settings and loaded
CFM10; the strengthened configuration test now verifies all inherited
prerequisites together.

The loaded-checkpoint startup gate passed on the Atlas 800I A3 / 910C host at
widths 50, 20, and 302. Every BSH preamble plus fused-attention comparison
reported zero maximum and mean absolute drift. The server log also confirmed
CFM6, homogeneous BF16, direct BSH attention, and fixed BSH slabs. The
unrelated final-Addcmul candidate again exceeded its own strict drift bound
and correctly retained canonical AdaLN.

The isolated attention screen includes cache append, Ascend fused attention,
and output projection. With the accepted width-50/cache-402 shape, queued
throughput used 100 iterations per trial over nine alternating trials;
serialized latency used one iteration over 21 trials. Lower is better.

| Mode | BNSD control | BSH candidate | Speedup | Max / mean drift |
| --- | ---: | ---: | ---: | ---: |
| Queued median | 152.459 us | 78.035 us | 1.9537x | 0 / 0 |
| Serialized median | 195.330 us | 125.270 us | 1.5593x | 0 / 0 |

The first real request exposed a second integration bug that the isolated
screen could not: the inherited `planar=true` flag reinterpreted an already
BSH cache a second time, changing the slab from a 512-wide BSH representation
to a spurious `2 x 256` representation. BSH now has explicit precedence over
the older planar conversion. The exact failing combination has a regression
test, and the deployed synthetic check preserves retained and append shapes
`[6,16,2,2,402,512]` and `[6,16,2,2,452,512]`.

The fresh-process control and both corrected candidate trials used the same
first 12 shuffled English Seed-TTS rows, two warmups, concurrency one, seed
zero, temperature zero, and CFM6. All three completed 12/12 with zero
failures, 100% continuity, 1,804 input tokens, 183 output tokens, 1,252,800
frames, and 52.20 seconds of audio. Candidate values are the arithmetic mean
of its two runs. Lower is better except throughput.

| Metric | Accepted planar control | BSH two-run mean | Change |
| --- | ---: | ---: | ---: |
| Serving duration | 17.596 s | 15.671 s | -10.94% |
| Request throughput | 0.6820 req/s | 0.7658 req/s | +12.30% |
| Audio throughput | 2.9666 audio-s/s | 3.3314 audio-s/s | +12.30% |
| Mean / P99 E2E | 1,465.87 / 1,901.21 ms | 1,305.40 / 1,612.54 ms | -10.95% / -15.18% |
| Mean / P99 TTFT | 333.43 / 483.15 ms | 314.90 / 443.13 ms | -5.56% / -8.28% |
| Mean / P99 audio TTFP | 798.43 / 949.64 ms | 746.60 / 888.54 ms | -6.49% / -6.43% |
| Mean / P99 whole-audio RTF | 0.340706 / 0.489958 | 0.304337 / 0.382010 | -10.67% / -22.03% |

Chunk timing shows that the improvement is concentrated in the repeated
Stage-2 path rather than only the first packet.

| Chunk metric | Accepted planar control | BSH two-run mean | Change |
| --- | ---: | ---: | ---: |
| Mean first-chunk RTF | 0.950518 | 0.888804 | -6.49% |
| Mean steady-chunk RTF | 0.227254 | 0.182423 | -19.73% |
| Median steady-chunk RTF | 0.159447 | 0.131855 | -17.31% |
| P99 steady-chunk RTF | 1.384496 | 0.873416 | -36.91% |
| Mean / median all-chunk RTF | 0.379520 / 0.190300 | 0.331135 / 0.143456 | -12.75% / -24.62% |
| P99 all-chunk RTF | 1.503704 | 1.103255 | -26.63% |

Candidate durations were 15.840 and 15.502 seconds. The follow-up 32-row
stability gate completed 32/32 with zero failures and zero underrun while
preserving the official 4,801-input-token, 480-output-token, 3,362,880-frame,
140.12-second signature. It measured 42.586 seconds duration, 1,330.48 ms
mean E2E, 322.82 ms mean TTFT, 751.21 ms mean TTFP, 0.309511 mean
whole-audio RTF, and 0.180239 mean steady-chunk RTF. The 32-row steady-chunk
P99 was 0.834274.

The cached offline Whisper-large-v3/WavLM 32-row quality screen passed the
repository's strict two-percentage-point gate. Both runs evaluated the same
32 rows with the same in-tree aligned WER and WavLM mean-pool proxy protocols.
The candidate had zero request, PCM, ASR, and SIM failures.

| Quality metric | Accepted planar control | BSH candidate | Regression | Gate |
| --- | ---: | ---: | ---: | --- |
| Mean / median WER (lower is better) | 0.016588 / 0 | 0.016588 / 0 | 0.000 pp | pass |
| Mean WavLM SIM (higher is better) | 0.844747 | 0.845047 | -0.030 pp | pass |
| Median WavLM SIM (higher is better) | 0.851154 | 0.852045 | -0.089 pp | pass |
| WER / SIM evaluated | 32 / 32 | 32 / 32 | matched | pass |

A negative regression means the candidate improved. This accepts BSH as the
next experimental 910C speed profile: its two-run request throughput improved
12.30%, mean steady-chunk RTF fell 19.73%, WER was bit-for-bit equal at the
aggregate level, and mean SIM improved slightly. It does **not** promote the
profile to a competition release. Full 1,088-row official Seed-TTS,
Daily-Omni, and Video-MME are still required release gates.

This run also found and fixed a benchmark-wrapper ambiguity. `--wer-eval`
previously claimed to enable WER/SIM/UTMOS but forwarded only WER, yielding a
plausible-looking result with `seed_tts_sim_evaluated=0`. The wrapper now
describes `--wer-eval` accurately and exposes explicit `--sim-eval`; UTMOS
remains separately opt-in through `SEED_TTS_UTMOS_EVAL=1` alongside an
evaluation flag.

Artifacts are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-bsh-attention-20260821
```

Selected checksums:

```text
97ba2ba0d553a77005e37b7d8249dec54dd19cba84498dfb95b587298ab891e7  control.json
1f810967ebba1ee5179d7211932d1c57756cd90868627d3b1bb310219a233586  candidate-run1.json
71e7bdb73f4f4fa1bba5b26766287a1726989548844159b0cb34e5b11e9ade00  candidate-run2.json
29726bc5cc0628a7cea28e1d318b2211cdf0995bd4aab951e5a4af84065811c8  candidate-32.json
6bead4b3543faac8d5113473a576ea269f963ba7e08731224b7fa699703117a8  candidate-quality32-wer.json
002576b97d77772663536a6b46c5728662a935415d55d874491d89e3689d03df  candidate-quality32-wer-sim.json
```

The reusable hardware command is:

```bash
python benchmarks/scripts/bench_minicpmo_dit_bsh_attention.py \
  --device 1 --width 50 --cache-length 402 --dtype bf16
```

## Fixed-slab six-step CFM NPUGraph

The fixed BSH slabs make the complete steady CFM invocation eligible for one
static executable. The opt-in profile is
`minicpmo_4_5_2npu_910c_cfm6_dit_bf16_planar_bsh_attention_cfm_graph_experimental.yaml`.
It captures only width 50 with the retained attention cache fixed at 402;
prompt setup, cache fill, and tail widths stay on the accepted eager/graph
partitions. Two graph and output-buffer slots share one memory pool, so replay
does not clone the six-step output and its 16-block cache slabs.

The competition CANN 9.0 image exposed two constraints that the earlier
growing-cache full-loop prototype could not solve:

- a TorchAir/GE executable cannot run inside a raw NPUGraph capture stream;
  the steady capture therefore lowers the accepted partitions to their plain
  graph-visible PyTorch operations instead of nesting compiled executables;
- BSH `npu_fusion_attention` launches an auxiliary stream that does not join
  the raw capture stream. The graph-only path uses explicit FP32
  BMM-softmax-BMM attention, while ordinary eager execution retains the much
  faster fused-attention operator.

An isolated 910C probe proved that Linear and explicit attention capture and
replay, while fused BSH attention fails `capture_end` with the same unjoined
stream error seen in serving. The loaded MiniCPM-o checkpoint gate at the real
`width=50/cache=402` shape measured only `3.05175781e-05` maximum and
`6.89178705e-08` mean absolute drift between explicit graph attention and the
fused BSH reference. Capture logs then confirmed two slots and steady replay
with the physical cache shape `[6,16,2,2,402,512]`.

The conservative reverse-order A/B used a fresh accepted-BSH control followed
by a fresh graph candidate, two deterministic 12-row runs per side, two
warmups, concurrency one, seed zero, and temperature zero. All 48 measured
requests completed with identical aggregate text/audio structure, 100%
continuity, and zero underrun. Lower is better except throughput.

| Metric | Fresh BSH control | Static CFM graph | Change |
| --- | ---: | ---: | ---: |
| Serving duration | 18.3729 s | 17.3091 s | -5.79% |
| Request throughput | 0.65314 req/s | 0.69328 req/s | +6.15% |
| Audio throughput | 2.84116 audio-s/s | 3.01576 audio-s/s | +6.15% |
| Mean / P99 E2E | 1,530.63 / 2,491.21 ms | 1,442.07 / 2,063.93 ms | -5.79% / -17.16% |
| Mean / P99 TTFT | 355.76 / 751.51 ms | 347.43 / 552.09 ms | -2.34% / -26.54% |
| Mean / P99 audio TTFP | 816.41 / 1,254.08 ms | 792.93 / 1,002.87 ms | -2.88% / -20.03% |
| Mean / P99 whole-audio RTF | 0.351184 / 0.462322 | 0.323371 / 0.403746 | -7.92% / -12.67% |
| Mean first-chunk RTF | 0.971920 | 0.943960 | -2.88% |
| Mean / median steady-chunk RTF | 0.237246 / 0.164262 | 0.203359 / 0.135051 | -14.28% / -17.78% |
| P99 steady-chunk RTF | 1.352178 | 1.221725 | -9.65% |

The reverse leg is deliberately reported instead of the faster first
candidate process, which measured 16.5548 seconds and 0.182333 mean
steady-chunk RTF. This avoids claiming accelerator process-order drift as a
kernel gain.

A resident 32-row stability run completed 32/32 with the official small-gate
signature: 4,801 input tokens, 480 output tokens, 3,362,880 frames, and 140.12
seconds of audio. It measured 45.44 seconds duration, 1,419.16 ms mean E2E,
326.92 ms mean TTFT, 769.96 ms mean TTFP, 0.33 displayed mean whole-audio RTF,
100% continuity, and zero underrun.

The matched 32-row offline quality gate reused the cached
Whisper-large-v3/WavLM evaluation stack. WER was exactly unchanged from the
accepted BSH control at `0.0165884463` mean and `0` median. Mean WavLM
similarity moved from `0.845046923` to `0.844583588`, a `0.000463335`
absolute reduction (`0.0463` percentage points); median similarity moved from
`0.852045149` to `0.851352572`. This is far inside the competition's two-point
regression limit. All 32 content and 32 similarity evaluations completed with
zero request, ASR, or similarity failures. The quality run's latency is not
reported as performance because local CPU ASR and similarity scoring competed
with the server during that invocation.

The profile remains experimental until the full 1,088-row Seed-TTS,
Daily-Omni, and Video-MME release gates pass.

Artifacts are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-bsh-cfm-graph-20260821
```

The quality result is
`candidate-32-quality-offline-omp16/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260821-101625.json`
with SHA-256
`ba4cd70ede198e4ae6ddc718c9c9f13449a62ce62e4ce2599965d91a9c1ab43d`.

## BF16 causal Conv-to-Linear custom-op integration (rejected)

The next kernel experiment extended the native AscendC causal Conv-to-Linear
operator to BF16 and substituted it for the graph-visible causal-pack plus
Linear producer inside the accepted fixed-slab six-step CFM executable. The
focused operator suite passed 128/128 direct NPU cases. An alternating
15-trial exact-shape microbenchmark measured the ordinary pack-plus-Linear
boundary at 60.458 us and the fused operator at 58.654 us, only a 1.0308x
speedup.

The serving gate used the accepted static-CFM graph as control. Each process
received two warmups and two deterministic 12-request Seed-TTS runs at
concurrency one. All 96 measured requests across the two A/B regimes
completed with 100% streaming continuity, zero underrun, and the same
aggregate structure: 1,784 input tokens, 163 output tokens, 1,160,640 audio
frames, and 48.36 seconds of audio. The local dataset copy was incomplete, so
this is a matched internal performance gate rather than an official
1,088-row quality result. Lower is better except throughput.

| Metric | Initial candidate vs control | Later candidate vs control |
| --- | ---: | ---: |
| Serving duration | -0.70% | -1.91% |
| Request/audio throughput | +0.56% | +1.96% |
| Mean / P99 E2E | -0.70% / -9.04% | -1.91% / -2.12% |
| Mean / P99 TTFT | -9.06% / +0.82% | -1.53% / -6.41% |
| Mean / P99 audio TTFP | +5.23% / +9.29% | -2.93% / -3.34% |
| Mean / P99 whole-audio RTF | +2.08% / +5.39% | -3.35% / approximately 0% |

The sign reversal in TTFP and whole-audio RTF, the low single-digit total
effect, and the 3.08% isolated headroom fail the promotion gate. The opaque
custom-op boundary also prevents GE from optimizing the producer-consumer
chain across the pack and Linear. The BF16 extension, serving substitution,
and experimental profile were therefore fully reverted in both forks. The
accepted graph-visible causal-pack path remains in service; any retry must
propagate one layout across the complete DiT producer-consumer chain or expose
the fused implementation through a GE converter/decomposition.

Artifacts are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-cfm-graph-conv-linear-bf16-20260821
```

Selected reverse-leg SHA-256 checksums are:

```text
1ce9f6916064941508f10e4be39dd0b878b917a62fd8b4f1cc70a4e9c1fc1ea4  reverse-candidate-run3.json
5e696f059e90b787f8337dd9c1731a4f9ecc693ed295c40ebc646393ee2e09ca  reverse-candidate-run4.json
02e77335a1e1e9bfa686daa043d5897df9b0837d514df7c223c451eba654d211  reverse-control-run3.json
b15d7ef153ca306679f55333b63ec9cbe2b589b66984f12b9fabcdb112887347  reverse-control-run4.json
```

## Immutable CFM AdaLN modulation slabs (rejected)

Fixed six-step CFM uses the same timestep embeddings and loaded AdaLN
parameters for every prompt and streamed chunk. This experiment computed the
six-step, sixteen-block modulation tensor and final-layer modulation once,
then retained those graph-produced buffers at stable addresses for subsequent
steady CFM replays. The first implementation cloned the outputs and was
rejected immediately because the clone discarded their profitable internal
layout. A second implementation retained the one-shot TorchAir producer
storage directly; focused tests passed 121/121 and serving logs proved that
the immutable slabs were created before the two-slot CFM capture and reused
during replay.

The performance gate used five deterministic 12-request measurements per side
at concurrency one. It included both process orders. One severe transient
contention sample occurred on each side, so the table reports the robust
five-run median rather than selecting or averaging favorable runs. Every run
completed 12/12 with zero failures and the same aggregate structure: 1,784
input tokens, 163 output tokens, 1,160,640 frames, and 48.36 seconds of audio.
Lower is better except throughput.

| Metric | Accepted static CFM graph | Immutable modulation slabs | Change |
| --- | ---: | ---: | ---: |
| Serving duration | 15.5907 s | 16.6199 s | +6.60% |
| Request throughput | 0.76969 req/s | 0.72203 req/s | -6.19% |
| Audio throughput | 3.10184 audio-s/s | 2.90976 audio-s/s | -6.19% |
| Mean / P99 E2E | 1,298.80 / 1,670.52 ms | 1,384.57 / 2,173.46 ms | +6.60% / +30.11% |
| Mean / P99 TTFT | 298.31 / 392.03 ms | 310.48 / 389.11 ms | +4.08% / -0.74% |
| Mean / P99 audio TTFP | 767.27 / 867.94 ms | 787.18 / 909.09 ms | +2.60% / +4.74% |
| Mean / P99 whole-audio RTF | 0.320006 / 0.387786 | 0.336686 / 0.402147 | +5.21% / +3.70% |

The stable external buffers remove repeated projection arithmetic, but they
also freeze a producer-consumer boundary outside the complete static CFM
executable. On this stack that boundary costs more in layout/scheduling than
the saved AdaLN projection. The option, code, tests, and deploy profile were
therefore fully removed. The accepted implementation continues to expose the
projection inside the graph-visible six-step CFM path, where GE can optimize
it together with downstream consumers.

Artifacts are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-static-cfm-modulations-20260821
```

Median-duration representative SHA-256 checksums are:

```text
6ebf29aa06159dba5c1dd06b1ceda6ebe41f1c15188a87f5165a3e1e0cc0737e  accepted reverse-control-run2.json
478ddafbece60b65cfe05dcc0389c0dd1c14728b4477d8dcd8806de8a8c78751  candidate candidate-v3-run4.json
```

## Complete six-step TorchAir/GE CFM executable (rejected)

The fixed width-50/cache-402 BSH slabs made it possible to test the deepest
remaining graph boundary: all six CFM steps and all sixteen DiT blocks in one
graph-visible TorchAir/GE executable. Prompt and tail shapes stayed on the
accepted eager/raw-graph path. The candidate was fail-closed and never
replaced the accepted profile.

The first graph dump identified a CANN 9.0 lowering bug at the DiT input
projection. TorchAir represented the logical `[2,50,320]` activation and
`[512,320]` weight as a generic rank-three `MatMul`; GE then treated sequence
width 50 as K and rejected it against 320. Flattening the projection to an
explicit `[100,320] x [320,512]` GEMM preserved the math and produced a valid
optimized graph. The aliased-output variant then compiled and replayed once,
but stopped publishing the streaming request because the fixed cache slabs
were both mutated inputs and returned outputs.

A graph-owned-output revision removed that alias. It completed streaming with
100% continuity and zero underrun, proving the liveness diagnosis, but was far
slower than the accepted raw graph. The original/optimized dumps shrank from
21.36/34.82 MB with aliased slabs to 18.74/30.37 MB with graph-owned outputs.
The first empty-kernel-cache build took roughly twelve minutes; after AscendC
kernel caching, the first warmup still took 221.57 seconds.

The post-compile smoke generated 3.12 seconds of audio. Lower is better.

| Metric | Complete GE executable |
| --- | ---: |
| Request E2E | 48,097.61 ms |
| Stage-2 wall time | 48,080.96 ms |
| Whole-audio RTF | 15.416 |
| TTFT | 765.44 ms |
| Audio TTFP | 1,453.52 ms |
| Prior post-compile warmup Stage 2 | 54,335.29 ms |

This is not a marginal regression that merits a larger A/B: the monolith
removed profitable kernel scheduling/concurrency and made a three-second clip
take forty-eight seconds. The code, environment switch, tests, graph-dump
setting, and deploy profile were fully removed, and the accepted raw
two-slot CFM graph was restored.

The graph dump nevertheless closes two unknowns. Fixed slabs do solve shape
churn, and a graph-visible complete CFM model can compile once the input
projection is made two-dimensional. The remaining limit is executable size
and scheduling, not capture eligibility. Any retry must use a bounded
partition--preferably one CFM step or a small contiguous block stripe--and
must keep outputs graph-owned without Python clones or input/output aliasing.

Artifacts are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-cfm-ge-20260821
```

The completed smoke result is
`smoke-v6-graph-owned-output/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260821-142623.json`
with SHA-256
`9076989ace4133f009f4abab5e984f06b4a709528f64498f33c838748f13cbfc`.

## Evaluator-YAML dual-chip source policy

The final submission integration revisited the organizer topology assumption.
One allocated Atlas 800I A3 card exposes two logical 910C chips.  The untouched
organizer YAML places every stage on logical device 0, so source policy now
recognizes exactly that baseline on a two-device NPU host and applies the
measured `[Thinker, Talker, Code2Wav] = [0, 0, 1]` placement.  Explicit
placements and single-device hosts remain unchanged.

The first source-policy run enabled the accepted BF16 fixed-planar partitions
on Code2Wav but left the organizer's Stage-0/1 `PIECEWISE` graph mode intact.
The service log proved every intended Stage-2 replay was active, yet mean RTF
remained 0.382.  A single-variable restart changed only Thinker and Talker to
`FULL_DECODE_ONLY` with capture sizes `[1,2,4]`; the first full 32-row run fell
to 0.315 mean RTF.  This identified a producer-side integration loss rather
than a failure of Stage-2 fusion.

The source policy then reproduced the full-decode/fixed-planar path through
the untouched organizer YAML, without private startup variables.  Its hot
32-row result was:

| Metric | Official-YAML planar source default |
| --- | ---: |
| Successful / failed | 32 / 0 |
| Duration | 47.3773 s |
| Mean / P99 E2E | 1,480.20 / 2,131.24 ms |
| Mean / P99 TTFT | 340.09 / 483.82 ms |
| Mean / P99 TTFP | 806.57 / 958.83 ms |
| Mean / P99 whole-audio RTF | 0.307862 / 0.348755 |
| Output tokens / audio frames | 559 / 3,737,280 |
| Streaming continuity / underrun | 100% / 0 |

An evaluator-shaped BSH-only follow-up improved mean TTFP to 801.50 ms but
regressed mean RTF slightly, so BSH was not accepted on that isolated result.
The deeper candidate then combined BSH with the existing two-slot fixed-address
steady-CFM NPUGraph.  Logs confirmed two captures and subsequent replay at
`mu=(1,80,50)` and cache shape `[6,16,2,2,402,512]`.  Prompt, cache fill, and
tail shapes retained their existing fallback.

Two independent hot measurements were stable on the mean metrics:

| Metric | Planar base | BSH + steady CFM graph run 1 | Run 2 | Two-run graph average vs base |
| --- | ---: | ---: | ---: | ---: |
| Mean RTF | 0.307862 | 0.303975 | 0.304455 | -1.18% |
| Mean E2E | 1,480.20 ms | 1,468.16 ms | 1,469.45 ms | -0.77% |
| Mean TTFT | 340.09 ms | 336.82 ms | 340.82 ms | -0.37% |
| Mean TTFP | 806.57 ms | 793.83 ms | 797.72 ms | -1.34% |
| P99 RTF | 0.348755 | 0.360155 | 0.360562 | +3.33% |

Lower is better.  All three runs completed the same 32 requests with the same
559-token/3,737,280-frame output signature, 100% continuity, and zero underrun.
These were explicit-profile isolation results, not yet a source-default gate.
Their repeatable P99 regression also required caution.

The decisive follow-up enabled the same BSH/static-graph combination through
source policy while passing only the organizer's untouched YAML.  Logs proved
that BSH compiled with zero drift, both fixed-address slots captured, and the
steady graph replayed.  No second serving process held an NPU.  Nevertheless,
the complete official entry measured 0.567 mean RTF, 2,734.98 ms E2E, 476.50
ms TTFT, and 1,161.87 ms TTFP.  It completed 32/32 with the same output
signature and continuity, so this is a performance integration failure rather
than a liveness or output-length artifact.

The submission default therefore retains the verified planar/full-decode
source path at 0.307862 and leaves BSH/static CFM replay explicit opt-in.
Relative to the earlier dual-chip CFM6 control at 0.379806 RTF, the accepted
path reduces RTF by 18.94%, E2E by 19.13%, TTFT by 6.55%, and TTFP by 13.07%.
It remains 3.87% above the reported 0.2964 leaderboard mark.

The previously recorded paired 32-row gates show that the rejected paths were
inside the accuracy budget: BSH preserved mean WER at 0.016588 and improved
mean SIM by 0.030 percentage points; static CFM replay preserved the same WER
and reduced mean SIM by 0.0463 percentage points.  Their rejection is based on
the official-entry performance gate, not accuracy.

Artifacts:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-dual-source-default-final-official-yaml-20260822
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-dual-full-decode-planar-bsh-20260822
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-dual-bsh-cfm-graph-20260822
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-dual-source-default-bsh-cfm-final-official-yaml-20260822
```

## A2 external trace and cache-fill graph screen

An external dynamic `msprof` trace refreshed the single-chip A2 attribution
after fixed slabs, BSH attention, HF32 MatMul, and two-slot steady CFM replay.
The representative ten-request trace counted 236,082 host kernel launches.
Every request still executed prompt width 302, width 50/cache 302, and width
50/cache 352 as complete six-step by sixteen-block eager CFM evaluations;
only eight later cache-402 evaluations used the outer steady graph. The
largest device families were TransData 310.47 ms, MatMulV2 264.81 ms,
LayerNormV3 231.97 ms, Transpose 211.70 ms, Mul 199.21 ms, Add 164.69 ms,
Slice 152.83 ms, and FlashAttention 129.77 ms. Profiling overhead invalidates
the run as a speed sample but not these counts.

Capturing both recurrent cache-fill shapes reduced TTFP but regressed matched
whole-audio RTF by 4.78%, so cache 352 was removed. A cache-302-only candidate
then completed four structurally matched 10-request runs with the accepted
1,036,800-frame / 43.2-second signature:

| Metric | Retained A2 HF32 graph | Cache-302-only mean | Change |
| --- | ---: | ---: | ---: |
| Whole-audio RTF | 0.33852 | 0.33470 | -1.13% |
| Audio TTFP | 0.75645 s | 0.67398 s | -10.90% |
| Text TTFT | 77.70 ms | 76.72 ms | -1.26% |
| Mean E2E | 1.46186 s | 1.44539 s | -1.13% |
| Steady-chunk RTF | 0.16915 | 0.18452 | +9.09% |

Lower is better. One mismatched run was excluded. The strong first-packet win
is real, but the later-chunk regression prevents promotion over the retained
balanced profile. Capturing both steady slots before the cache-302 graph did
not fix the interaction: two matched runs averaged 0.35645 whole-audio RTF
and 0.18925 steady-chunk RTF while retaining roughly 0.67210-second TTFP. That
allocation-order experiment was removed. Further work should not add another
persistent raw NPUGraph boundary; it should use a bounded GE-visible
producer-consumer partition or reduce work within the existing executable.

## Full-DiT-block GE experiment

The next experiment moved one complete BSH DiT block, including the canonical
Conv1d producer/consumer chain, behind a single TorchAir/GE executable. It did
not use the earlier native causal-pack helper: that helper was fast in
isolation but produced unacceptable real-weight drift. The canonical-Conv
variant preserved the intended block arithmetic to BF16 tolerance.

On real checkpoint weights at width 50/cache 302, the isolated block result
was:

| Path | Mean block latency | Relative speed |
| --- | ---: | ---: |
| Split eager control | 1,096.45 us | 1.00x |
| Canonical-Conv full GE block | 402.44 us | 2.72x |
| Native causal-pack diagnostic | 281.93 us | 3.89x, rejected for drift |

The service-level cache-302 candidate compiled and replayed the canonical
full-block graph and completed 10/10 requests. Its structurally matched hot run
used the same 1,036,800-frame / 43.2-second output signature as the retained
A2 result:

| Metric | Retained A2 profile | Full block at cache 302 | Change |
| --- | ---: | ---: | ---: |
| Whole-audio RTF | 0.33852 | 0.33707 | -0.43% |
| Audio TTFP | 0.75645 s | 0.67801 s | -10.37% |
| Text TTFT | 77.70 ms | 79.53 ms | +2.35% |
| Mean E2E | 1.46186 s | 1.45614 s | -0.39% |
| Steady-chunk RTF | 0.16915 | 0.18721 | +10.67% |

Lower is better. The profile remains an explicit low-TTFP experiment rather
than the submission default: its first-packet gain is material, but it moves
work into the steady path, regresses steady-chunk RTF, and has not passed the
three-suite quality gate.

Two attempts to extend the optimization further were rejected:

- nesting the cache-402 GE executable inside the retained outer NPUGraph
  failed on the first request with `Unsupport run graph with different stream`;
  the GE executable is bound to the default stream while the outer capture
  runs on a capture stream;
- TorchAir ACLGraph replay preserved exact microbenchmark output but took
  2,097.83 us versus 1,142.68 us for the control (0.54x), and enabling the
  ACLNN static-shape compiler terminated its TBE worker before producing an
  executable. The separate `npugraph_ex` package is absent from this A2 image.

The safe selector therefore excludes full-block GE when an outer flat capture
is active. Retrying a single static six-step executable requires a compatible
CANN/TorchAir image with NPUGraphEx support or an explicit same-stream graph
composition API; it is not safe to emulate by nesting the current wrappers.

After reverting the unsafe nested selector, the retained HF32/BSH/steady-CFM
profile was restored on the A2 host. A cold warm-up request generated 3.88
seconds of continuous audio, and the following two hot requests completed 2/2
with 77.38 ms mean TTFT, 8.28 seconds of generated audio, and 100% streaming
continuity. Runtime logs confirmed `slots=2` and `NPU CFM graph replay active`;
the model API remained healthy with HTTP 200.

## Partial-CFG batch-one screen

A deeper work-reduction candidate evaluated classifier-free guidance only in
the first two of the six CFM solver steps. The first two steps retained the
existing conditional/unconditional batch of two; the remaining four executed
the conditional branch as batch one. Fixed cache addresses and the public
cache ABI were preserved by slicing only the compute view and mirroring the
conditional result into the unused half. A matching Ascend causal-pack kernel
variant accepted batch one, and partial guidance was restricted to the static
width-50/cache-402 steady path so prompt, cache-fill, tail, and first-packet
arithmetic remained unchanged.

The candidate completed every request with 100% streaming continuity. It did
not improve device throughput. Comparisons below use runs with exactly the same
1,160,640 output frames and 48.36 seconds of generated audio, excluding the
separate Talker tail-length variation:

| Metric | Full-CFG repeat | Partial-CFG repeat mean | Change |
| --- | ---: | ---: | ---: |
| Whole-audio RTF | 0.361909 | 0.376188 | +3.95% |
| Steady-chunk RTF | 0.390668 | 0.406805 | +4.13% |
| Mean request duration | 17.2881 s | 18.0161 s | +4.21% |
| Audio TTFP | 0.763214 s | 0.767365 s | +0.54% |

Lower is better. The one-shot structurally matched comparison agreed with the
repeats: steady-only partial CFG measured 0.346920 whole-audio RTF versus
0.339104 for full CFG, a 2.30% regression. The batch-one path halves the
logical MLP/attention rows for four solver steps, but these small matrices no
longer fill the A2 Cube efficiently; launch and layout costs become a larger
fraction of the step. Therefore the partial-CFG selector and batch-one kernel
are rejected rather than promoted, and no quality gate is claimed for them.

Artifacts:

```text
/tmp/lunanexa-bench/full-cfg-reverse-control/full-cfg-reverse-control.json
/tmp/lunanexa-bench/full-cfg-reverse-control-repeat/full-cfg-reverse-control-repeat.json
/tmp/lunanexa-bench/partial-cfg2-steady/partial-cfg2-steady.json
/tmp/lunanexa-bench/partial-cfg2-steady-repeat/partial-cfg2-steady-repeat.json
/tmp/lunanexa-bench/partial-cfg2-steady-repeat2/partial-cfg2-steady-repeat2.json
```

## A2 selective dynamic-W8A8 DiT MLP screen

The selective dynamic-W8A8 experiment keeps attention probabilities,
normalization, CFG/Euler integration, and HiFT on their higher-precision paths,
while quantizing the two MLP matrix multiplications in each DiT block. An A2
runtime ABI issue was fixed before measurement: `npu_quant_matmul` requires a
one-dimensional per-token scale, so a BSH activation is flattened to
`[batch * time, hidden]` before dynamic quantization and reshaped after the
Cube matmul. Four focused tests passed, including the exact scale and output
shape contract.

The current CANN 8.5.0 image could not capture the raw outer steady-CFM graph
with the W8A8-compatible configuration; the first request failed during graph
capture. A follow-up disabled W8A8 while preserving cache-major state and the
outer graph; its first real TTS request still terminated the StageEngine
processes during capture, without a Python exception. This isolates the unsafe
boundary to cache-major/flat-capture rather than INT8 arithmetic itself. The
crash-only diagnostic profiles were removed. A matched
no-outer-graph comparison completed 10/10 requests on each path with the same
1,746 input tokens, 1,048,320 output frames, 43.68 seconds of generated audio,
and 100% streaming continuity:

| Metric | Matched BF16 control | Dynamic-W8A8 MLP | Change |
| --- | ---: | ---: | ---: |
| Whole-audio RTF | 0.413941 | 0.409697 | -1.03% |
| Mean raw chunk RTF | 0.440199 | 0.432284 | -1.80% |
| Total run duration | 17.2473 s | 17.1671 s | -0.47% |
| Text TTFT | 77.24 ms | 75.95 ms | -1.67% |
| Audio TTFP | 759.03 ms | 810.59 ms | +6.79% |

Lower is better. Dynamic W8A8 slightly reduces steady arithmetic time, but it
regresses first-packet latency and both no-outer paths are materially slower
than the retained outer-CFM-graph profile. It is therefore not a submission
default and does not proceed to the three-suite accuracy gate. The generic A2
scale-ABI fix remains useful for explicit future W8A8 candidates; the
performance profile remains experimental.

The follow-up kept the retained channel-major Conv-cache ABI and embedded
`npu_dynamic_quant` plus `npu_quant_matmul` directly inside the raw steady-CFM
capture. It introduced neither a nested GE executable nor the cache-major
layout. Runtime logs confirmed both `NPU CFM graph captured`/replay and the
selective W8A8 path. A matched official-wrapper comparison used exactly the
same 1,746 input tokens, 1,048,320 audio frames, 43.68 seconds of generated
audio, two warmups, and ten measured requests:

| Metric | BF16 outer graph | Channel-major W8A8 outer graph | Change |
| --- | ---: | ---: | ---: |
| Whole-audio RTF | 0.401680 | 0.395701 | -1.49% |
| Mean raw chunk RTF | 0.431435 | 0.420423 | -2.55% |
| Total run duration | 16.6901 s | 16.5360 s | -0.92% |
| Text TTFT | 76.11 ms | 78.09 ms | +2.60% |
| Audio TTFP | 764.74 ms | 826.12 ms | +8.03% |

Lower is better. This proves that the INT8 operators can execute inside the
outer graph, but the dynamic per-token reduction/scale overhead consumes most
of the Cube saving and materially regresses first packet latency. The path
remains opt-in and does not replace the BF16 submission profile.

### A2 weight-only and fused-GELU lower-layer screens

`npu_weight_quant_batchmatmul` was tested as a W8A16 alternative so activations
would not need dynamic quantization. The installed A2 documentation confirms
BF16 activations, per-channel INT8 weights, and graph-mode support. At the
actual DiT MLP shape (`100x512 -> 2048 -> 512`), however, the complete
weight-only chain took 117.12 us versus 69.07 us for BF16, a 69.6% latency
regression. A2's weight dequantization setup cost dominates at this small M,
so W8A16 was rejected before service integration.

The newer A2 `npu_quant_matmul_gelu` operator was then screened. Its documented
BF16-scale output branch was numerically invalid on this CANN 8.5 image: output
magnitude reached 6,304 for a reference bounded near 2. Using FP32 weight scale
selects the valid FP16-output branch, which matched the split FC1+GELU with
0.0078125 maximum and 0.000369 mean absolute error. FC2's dynamic quantizer can
consume that FP16 producer directly. For the complete two-layer MLP, the fused
path reduced split dynamic-W8A8 latency from 173.72 us to 121.60 us (1.43x),
while its drift from BF16 was comparable to the unfused quantized path. It
remained slower than eager BF16 in isolation, so it is a graph-level candidate,
not an assumed win. Raw NPUGraph capture succeeded even though the installed
TorchAir lacks an AscendIR converter for the fused op. The structurally matched
service run measured 0.404142 RTF, 0.425329 mean raw-chunk RTF, and 861.95 ms
TTFP. That is 2.13%, 1.17%, and 4.34% worse respectively than split W8A8, and
also worse than the BF16 control in whole-audio RTF and TTFP. The fusion was
therefore removed from the runtime rather than retained as dead experimental
surface.

The next lower-layer screen uses A2's dense non-quantized `npu_ffn`, which
fuses both BF16 projections and GELU without dynamic scales. At the same actual
`100x512 -> 2048 -> 512` shape it took 42.65 us versus 75.52 us for the split
BF16 chain, a 1.77x microbenchmark speedup. Its maximum/mean absolute drift was
0.0078125/0.000591. This is now the active graph-level candidate: transposed
Cube-ready weights and FP32 biases are allocated once, while normalization,
AdaLN and the residual remain BF16 outside the fused operator.

Artifacts:

```text
/tmp/lunanexa-bench/bf16-no-outer-graph-control/bf16-no-outer-graph-control.json
/tmp/lunanexa-bench/w8a8-mlp-no-outer-graph/w8a8-mlp-no-outer-graph.json
/tmp/lunanexa-bench/bf16-outer-graph-matched-control/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260826-120243.json
/tmp/lunanexa-bench/w8a8-channel-major-outer-graph/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260826-115106.json
/tmp/lunanexa-bench/w8a8-fused-gelu-outer-graph-v2/bench_tts_openbmb_MiniCPM-o-4_5_voice_clone_c1_20260826-123758.json
/tmp/minicpmo-bf16-no-outer-graph-control.log
/tmp/minicpmo-w8a8-mlp-no-outer-graph.log
/tmp/minicpmo-bf16-outer-graph-matched-control.log
/tmp/minicpmo-w8a8-channel-major-outer-graph.log
```

## Talker codec-sampler graph and AICPU boundary

NPU telemetry on a structurally matched 10-request run showed only 1--16%
AICore utilization and 1--4% HBM-bandwidth utilization. Stage logs attributed
roughly 0.74--1.55 seconds per request to the autoregressive Talker, compared
with about 0.30--0.52 seconds added by Code2Wav. The next large target was
therefore the per-code Talker continuation rather than another DiT micro-op.

Every codec token historically launched a BF16 `768 -> 6562` head, frequency
penalty, top-k/top-p filter, softmax, AICPU multinomial, gather, and frequency
window update outside the already captured Llama decode graph. Capturing only
the deterministic distribution preserved the checkpoint's NPU multinomial
sequence and measured 2.34x faster in isolation. It was rejected in service:
the first real decode consistently raised Ascend 507018 from
`MultinomialWithReplacement`, even after all graph outputs were verified as
finite ND tensors and copied into fixed non-graph-pool buffers. The failing
path is not a tensor-format or graph-storage-lifetime bug; it is the AICPU
multinomial boundary after replay in the vLLM FULL_DECODE environment.

The replacement uses one fixed graph for the complete codec continuation:

- the request-local NPU generator produces one uniform scalar outside capture;
- the graph executes the codec head, penalty, bounded top-k/top-p filter,
  inverse-CDF draw, gather, and 16-code rolling-frequency update;
- EOS masking and the expired window code are fixed-address runtime inputs, so
  the same executable covers both pre-minimum and EOS-eligible steps;
- sampled-code and next-frequency buffers are allocated once and reused.

Inverse-CDF is categorically distribution-equivalent for a single draw, but it
does not preserve multinomial's seed-to-code mapping. It is therefore an
accuracy-gated experimental path, not a bitwise-equivalent default. On the A2
microbenchmark with the checkpoint's actual hidden/vocabulary/top-k shapes,
the full eager continuation measured 1,190.29 us. Graph replay measured
208.96 us, and replay including request-local NPU uniform generation measured
303.98 us: 3.92x end-to-end kernel-chain speedup. Five focused CPU tests
passed for bounded-distribution, inverse-CDF selection, EOS masking, and
rolling-frequency equivalence.

Artifacts:

```text
benchmarks/kernels/bench_minicpmo45_codec_sampler_npu.py
/tmp/minicpmo-talker-sampler-graph-v3.log
```

The inverse-CDF service then completed two official-wrapper hot runs with two
warmups, ten measured Chinese Seed-TTS requests, concurrency one, and 100%
streaming continuity:

| Metric | Prior fused-FFN two-run mean | Inverse-CDF run 1 | Inverse-CDF run 2 | Two-run mean vs control |
| --- | ---: | ---: | ---: | ---: |
| Whole-audio RTF | 0.35557 | 0.32294 | 0.30336 | -11.93% |
| Audio TTFP | 765.28 ms | 751.18 ms | 749.79 ms | -1.93% |
| Text TTFT | 79.85 ms | 80.13 ms | 79.94 ms | +0.23% |
| Middle-chunk RTF | 0.17701 | 0.16795 | 0.17280 | -3.75% |
| Talker inter-output latency | about 9.9--10.0 ms | about 9.0--9.3 ms | about 9.0--9.3 ms | about -8% |

Lower is better. Run 1 generated 56.76 seconds / 1,362,240 frames and run 2
generated 63.08 seconds / 1,513,920 frames, compared with 43.20 seconds /
1,036,800 frames for the seed-mapped control. Total serving duration is
therefore not structurally comparable; RTF and per-output latency are the
valid normalized comparisons. A separate measured request reached 0.296 RTF,
734.35 ms TTFP, and 75.27 ms TTFT, matching the published leaderboard scale,
but it is not reported as the multi-request mean.

An eight-row English accuracy screen generated 8/8 valid audios with 100%
streaming continuity and 0.32 performance RTF. Its WER/SIM evaluator could not
run on this replacement host: Whisper Large v3 and WavLM were not cached, and
the Hugging Face processor download was reset by the remote endpoint. The
inverse-CDF profile therefore remains experimental and must not replace the
submission profile until paired WER/SIM stays within two percentage points and
Daily-Omni plus Video-MME are rerun. The client was stopped after the first
repeated download failure to avoid eight long retries; that interrupted run
did not persist its temporary WAVs, so the eight-row screen must be rerun once
the evaluator weights are available.

```text
/tmp/lunanexa-bench/talker-inverse-cdf-v4-official10/
/tmp/lunanexa-bench/talker-inverse-cdf-v4-official10-repeat/
/tmp/lunanexa-bench/talker-inverse-cdf-v4-quality-en8/
/tmp/minicpmo-talker-sampler-graph-v4.log
```

```text
790eb64835f4e42d99963e14b2769d1578184d734bf9e3c45b1fe4f758199d92  inverse-cdf-run1.json
c94bc10eaa9d99645fe43c46182e4b11985464d6edc68165b6ae7f3e76f49b7d  inverse-cdf-run2.json
```

## First-packet scheduling and prompt-cache solver reduction

The inverse-CDF profile still waited for 25 Talker codec codes before starting
Code2Wav. Separating the initial bridge threshold from the steady 25-code
chunk and setting it to ten reuses the already compiled width-20 DiT graph.
It does not alter the sampled codec sequence or steady chunk policy. On the
official ten-request English wrapper it reduced mean/P99 TTFP from
609.48/623.52 ms to 467.16/479.53 ms after the additional solver changes
below, with 100% streaming continuity.

Stage timing exposed a larger hidden first-path cost. Seed-TTS uses a different
reference waveform for each request. Stage 2 runs the widest (~302-frame) CFM6
path before the first live packet only to populate the prompt estimator K/V
caches; its synthesized prompt mel is discarded. The new experimental policy
uses two cosine-schedule evaluations for this cache prefill, expands the two
cache states back into the fixed six-slot ABI, uses four evaluations for only
the first 120 ms live packet, and returns every subsequent chunk to CFM6.
Speaker embedding, prompt mel extraction, codec sampling, and all steady audio
remain unchanged.

| Candidate | Mean RTF | Mean TTFP | P99 TTFP | TTFT | Continuity |
| --- | ---: | ---: | ---: | ---: | ---: |
| 10-code first packet, CFM6 (10 requests) | 0.323 | 609.48 ms | 623.52 ms | 81.23 ms P99 | 100% |
| 10-code first packet, prompt CFM2 / first CFM4 (1 request) | 0.308 | 458.22 ms | 458.22 ms | 80.49 ms | 100% |
| 10-code first packet, prompt CFM2 / first CFM4 (10 requests) | 0.317 | 467.16 ms | 479.53 ms | 77.59 ms mean | 100% |
| 10-code first packet, prompt CFM1 / first CFM1 (1 request) | 0.319 | 364.00 ms | 364.00 ms | 77.60 ms | 100% |
| 10-code first packet, prompt CFM1 / first CFM1 (10 requests) | 0.318 | 356.87 ms | 365.81 ms | 78.46 ms mean | 100% |

Lower is better. The ten-request prompt-CFM2 candidate generated 54.00 seconds
and 1,296,000 frames. Compared with the structurally matched first-packet CFM6
run, mean TTFP improved 23.35% while mean whole-audio RTF remained within run
variance. Compared with the earlier inverse-CDF two-run mean near 750 ms, TTFP
improved about 37.7%.

The minimum-solver ceiling experiment reduced both the discarded prompt-cache
prefill and only the first 120 ms live packet to one evaluation. Its official
ten-request run generated the same 54.00 seconds / 1,296,000 frames with 100%
continuity. Mean TTFP improved 41.45% versus the 609.48 ms first-packet CFM6
control and about 52.4% versus the earlier ~750 ms inverse-CDF runs. Mean RTF
remained 0.318. This is the current fastest measured TTFP profile, but it has a
strictly larger quality risk than prompt CFM2 / first CFM4 and must not become
the submission default without the complete accuracy gate.

An even smaller five-code first packet is the HiFT continuity lower bound:
five new codes plus three left-context codes become ten mel frames after the
encoder; HiFT retains eight mel frames / 3,840 samples and can publish one real
40 ms packet. Eager width-10 execution was stable and measured 566.04 ms TTFP,
but it was not faster than graph-backed width 20 because launch overhead
dominated. Adding width 10 to the captured graph buckets is rejected on this A2
stack: startup compilation succeeds, but first replay aborts the Stage-2
process with CANN `tiling offset out of range`. The submission must not enable
that graph bucket.

Prompt CFM2 and first-packet CFM4 approximate cache states and therefore remain
accuracy-gated experiments. They require paired official WER/SIM, Daily-Omni,
and Video-MME validation before replacing the submission profile.

```text
/tmp/lunanexa-bench/talker-low-ttfp-i10-official10/
/tmp/lunanexa-bench/talker-low-ttfp-prompt2-cfm4-smoke/
/tmp/lunanexa-bench/talker-low-ttfp-prompt2-cfm4-official10/
/tmp/lunanexa-bench/talker-low-ttfp-prompt1-cfm1-smoke/
/tmp/lunanexa-bench/talker-low-ttfp-prompt1-cfm1-official10/
/tmp/lunanexa-bench/talker-low-ttfp-i5-eager-cfm4-smoke/
/tmp/minicpmo-low-ttfp-prompt2-cfm4.log
```

## A2 main-bottleneck isolation and Talker producer-consumer fusion

Fresh hot-stage timestamps on the single-chip 910B4/A2 host changed the
optimization priority.  For a representative resident request, Stage 0 took
about 64 ms, Stage 1 completed at about 1,195 ms, and Stage 2 completed at
about 1,627 ms.  The incremental Code2Wav cost was therefore about 431 ms,
while the autoregressive Talker consumed about 73% of hot end-to-end latency.
Talker emitted roughly 120--160 codec tokens at about 9 ms/token.  This makes
another isolated Stage-2 microkernel the wrong main target: even eliminating
ten percent of Stage 2 would save only about 43 ms, whereas every millisecond
removed from the Talker loop is repeated more than one hundred times.

Several plausible lower-layer candidates were screened with the same two
warmups, ten fixed English Seed-TTS requests, concurrency one, and identical
54.00 seconds / 1,296,000 output frames:

| Stage-1 candidate | Mean RTF | Audio TTFP | TTFT | E2E | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| NZ BF16 weights (`weight_nz_mode=2`) | 0.307868 | 352.10 ms | 79.55 ms | 1,617.03 ms | retained control |
| Static kernel + NPUGraph-ex | 0.311524 | 351.41 ms | 75.90 ms | 1,636.56 ms | reject |
| Explicit FULL_DECODE_ONLY | 0.310824 | 352.54 ms | 79.85 ms | 1,634.76 ms | redundant/reject |
| Device-resident EOS branch | 0.312021 | 353.88 ms | 75.99 ms | 1,640.17 ms | reject |

Lower is better.  The explicit decode profile was redundant because the A2
single-chip producer policy already resolves Stage 1 to
`FULL_DECODE_ONLY` with capture size one.  Device EOS was slower because the
next autoregressive token already depends on the sampled codec token: its host
scalar decision is not an independent bubble, and the extra compare/where
operators cost more than the avoided scalar read.  The static-kernel result
also confirms that a more opaque executable is not automatically faster when
it reduces GE's scheduling freedom.  Only NZ weight preformatting survived the
whole-service gate.

The first structural experiment appended the complete codec continuation to
the Talker graph: the `768 -> 6562` head, repetition penalty, bounded
top-k/top-p, inverse-CDF draw, and frequency update. TorchAir produced a new
Stage-1 cache hash and captured full decode successfully. A one-shot log also
proved that real requests consumed its fused sample rather than a fallback.
Nevertheless, it was decisively slower: the first request took 60.73 seconds
and the immediately following resident request still took 9.89 seconds for
3.88 seconds of output, compared with 1.617 seconds mean E2E for the NZ
control. Putting vocabulary-wide `pow`, top-k, prefix-sum and sampling into the
large Llama GE graph destroyed the efficient small-batch decode schedule. The
experiment was removed rather than hidden behind the submission profile.

The narrower follow-up appended only the `768 -> 6562` codec head and
temperature scaling. They wrote one fixed-address FP32 logits row, which the
proven small inverse-CDF NPUGraph consumed directly. Even this boundary was
too opaque: after the first 61.85-second lazy-capture request, the immediately
following resident request still took 9.64 seconds for 4.88 seconds of audio.
The result is essentially the same six-fold regression as the complete-tail
fusion. The large external-buffer write, not only top-k, prevents GE from
preserving the efficient batch-one Llama decode schedule. This candidate and
its environment flag were removed as well.

A final boundary check returned the head logits as an ordinary two-tensor
model output instead of mutating an external buffer. This allowed GE to see
the producer value and the small sampler graph copied the row as a regular
input. It still measured 61.34 seconds on the first lazy-capture request and
9.56 seconds hot for the same 4.88-second waveform. Therefore the regression
is not specific to an opaque side effect: extending this FULL_DECODE graph
with the large head itself changes the batch-one executable unfavorably. The
tuple-output candidate was removed too.

The retained architecture therefore keeps the Llama full-decode graph and the
roughly 0.30 ms/token inverse-CDF sampler graph separate. The next material
Talker gain must be graph-visible fusion inside the Llama producer-consumer
chain, or a trained multi-code/speculative head; another side-effecting custom
boundary is not a viable route on this A2 stack.

### Hot Talker trace: launch bubbles, then slot mapping

A Stage-1-only `torch_npu.profiler` trace of one warmed 120-code request
resolved the next optimization level. Profiling was initialized only in the
Talker process; Stage 2 retained its normal fixed CFM NPUGraph because an
auxiliary profiler stream in that process invalidates capture on this CANN
release. The trace covered a 1.402-second device window, of which only
230.708 ms was device compute and 1.171 seconds was free. Thus 83.5% of the
Talker device window was idle rather than arithmetic. The 120 identical main
decode bursts each contained 201 kernels, spanned 4.356 ms on average, and
started 11.229 ms apart. Their summed kernel time was only 1.449 ms per code.

Within the compute budget, MatMulV2 used 101.399 ms (43.95%), fused infer
attention 47.761 ms (20.70%), and the generic slot-mapping kernel 22.814 ms
(9.89%). The slot kernel ran 121 times at 188.54 us average. Host attribution
also showed 130.800 ms in 2,420 fused-attention dispatches, 92.376 ms in 4,078
copies, 37.890 ms in event recording, and 28.301 ms in event synchronization.
This confirms two different budgets: the immediate safe target is the
oversized metadata kernel; the larger architectural target is eliminating
per-layer/per-code host dispatch through a multi-code or device-loop Talker
executable.

The first candidate replaced only the batch-one, one-token, non-DCP slot
calculation with a fixed-address NPUGraph. It dynamically indexed the stable
block table from the stable position buffer and wrote slot zero; prefill,
batching, DCP, graph nesting, and address changes fell back to the canonical
Triton kernel. On the same A2, its isolated 500-replay microbenchmark was
2.09x faster. The service-level Stage-1 ITL nevertheless regressed from
8.752 ms to 9.097 ms, so the candidate and its profile flag were removed.
This is another example of a locally faster boundary losing more in launch and
graph interaction than it saves in arithmetic.

Artifacts:

```text
/tmp/lunanexa-bench/talker-nz-official10/
/tmp/lunanexa-bench/talker-static-kernel-official10/
/tmp/lunanexa-bench/talker-full-decode-official10/
/tmp/lunanexa-bench/talker-device-eos-official10/
/tmp/lunanexa-bench/talker-fused-continuation-smoke3/
/tmp/lunanexa-bench/talker-fused-continuation-smoke4/
/tmp/minicpmo-talker-fused-continuation-v3.log
/tmp/lunanexa-bench/talker-fused-head-smoke-corrected1/
/tmp/lunanexa-bench/talker-fused-head-smoke-corrected2/
/tmp/minicpmo-talker-fused-head-v2.log
/tmp/lunanexa-bench/talker-head-output-smoke1/
/tmp/lunanexa-bench/talker-head-output-smoke2/
/tmp/minicpmo-talker-head-output.log
/tmp/lunanexa-bench/talker-nz-torch-profile-hot/
/tmp/vllm-omni-profiles/minicpmo45/a2-talker-nz-stage1/
```

### Main-bottleneck result: publish codec payloads at chunk boundaries

The hot trace showed that the dominant Stage-1 cost was not a single compute
kernel: the NPU was idle for 83.5% of the measured window. The retained change
therefore removes per-code framework work rather than adding another isolated
kernel. Previously every generated codec code caused `make_omni_output` to
publish one NPU scalar, the NPU runner to build a CPU multimodal payload and
copy an unused 768-wide hidden row, and the engine to create a downstream
connector task. Code2Wav cannot consume those single-code messages: after its
initial ten codes it works in 25-code chunks.

The new Talker path accumulates the exact sampled codec values on device and
publishes only at the initial 10-code boundary, each subsequent 25-code
boundary, and the terminal tail. Sparse output metadata lets the NPU runner
skip payload construction and connector routing on all other decode steps.
For the MiniCPM-o Talker, the unused hidden-state transfer is also omitted.
A typical 120-code response consequently creates about five downstream
payloads instead of about 120. Sampling, the codec sequence, CFM steps, HiFT,
and audio chunk boundaries are unchanged.

Two independent official-shape runs used two warmups followed by ten measured
requests at concurrency one. Lower is better for every value in this table.

| Candidate | Mean RTF | P99 RTF | Mean TTFP (ms) | Mean TTFT (ms) | Mean E2E (ms) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous retained baseline | 0.307868 | 0.386839 | 352.10 | 79.55 | 1617.03 |
| Chunk-boundary payload run 1 | 0.302844 | 0.383994 | 352.89 | 79.63 | 1589.46 |
| Chunk-boundary payload run 2 | 0.298986 | 0.379363 | 349.21 | 79.26 | 1569.91 |
| Two-run mean | **0.300915** | - | - | - | - |

The two-run mean RTF is 2.26% lower than the previous retained baseline. The
repeat run is 2.89% lower in mean RTF and E2E, while TTFP is 0.82% lower and
TTFT is 0.36% lower. Its ten-request Stage-1 mean ITL was 8.593 ms, 1.82%
lower than the 8.752-ms control. Both runs produced all 1,296,000 expected
frames for 54 seconds of reference duration with 100% chunk continuity.

Two more transport experiments were rejected. Reusing sampled-token transport
storage changed output duration between otherwise identical requests and only
reduced ITL by about 0.24%. Copying outputs on a separate NPU stream increased
hot ITL to 8.948--9.118 ms. Neither implementation remains in the candidate.

This optimization is sequence-preserving by construction, and the focused
batching, sparse-publication, cleanup, configuration, and syntax checks pass.
Formal submission acceptance still requires the official TTS-Seed WER/SIM
gate; the local continuity and frame-count checks are not a substitute for
that evaluator.

Artifacts:

```text
/tmp/lunanexa-bench/talker-batched-codec-v2-official10/
/tmp/lunanexa-bench/talker-batched-codec-v2-official10-repeat/
/tmp/minicpmo-talker-batched-codec-v2.log
```

Two subsequent scheduler-level attempts established where the remaining
Talker bottleneck is not. First, the intermediate AR scheduler retained all
per-token request, KV, stop, and sampling updates but suppressed token-only
`EngineCoreOutput` messages between publishable codec boundaries. Its two
official-shape runs measured mean RTF 0.308210 and 0.305143, for a 0.306676
mean: 1.91% slower than the retained 0.300915 result. Mean TTFP stayed near
349.3 ms. The async vLLM pipeline therefore uses those apparently redundant
outputs as part of its efficient progress cadence; the experiment was fully
removed.

Second, the orchestrator polled Stage 2, then Stage 1, then Stage 0 so the
high-frequency Talker output would not sit behind an idle Stage-0 one-ms poll.
Two runs measured mean RTF 0.304688 and 0.302651, for a 0.303669 mean. Average
TTFP improved by about 2.7 ms, but RTF was still 0.91% worse and mean Stage-1
duration increased from 1.195 s to about 1.219 s. This proves the repeated
gap is inside the Talker EngineCore/model-execution cycle rather than the
outer orchestration polling order. The order change was also removed.

Artifacts:

```text
/tmp/lunanexa-bench/sparse-engine-output-official10/
/tmp/lunanexa-bench/sparse-engine-output-official10-repeat/
/tmp/minicpmo-sparse-engine-output.log
/tmp/lunanexa-bench/downstream-first-poll-official10/
/tmp/lunanexa-bench/downstream-first-poll-official10-repeat/
/tmp/minicpmo-downstream-first-poll.log
```

### Main-bottleneck result: remove the redundant Talker stop sampler

The next trace-level target was the generic vLLM sampler invoked after every
codec step. `make_omni_output` already decides whether the request must
continue or stop, then exposes deterministic logits `[0, -inf]` or
`[-inf, 0]`. Running the generic logits processor and argmax on that binary
control head repeated work already completed by the model roughly 120--200
times per request.

The retained fast path returns the existing continue/stop decision directly
as a `SamplerOutput`. For the competition's batch-one path, immutable int32
continue and stop tensors are allocated once and reused. Requests asking for
log probabilities, and configurations that do not explicitly enable the
feature, retain the canonical sampler. Codec sampling, RNG state, generated
codec IDs, chunk boundaries, CFM, and HiFT are unchanged. A focused parity
test covers both continue and max-token stop decisions and verifies resident
buffer reuse.

Fresh tests used the newly extracted Chinese Seed-TTS set, two warmups, ten
measured requests, and concurrency one. Talker output duration is stochastic,
so whole-audio RTF is reported alongside the less ambiguous per-token Stage-1
ITL. Lower is better throughout.

| Run | Mean RTF | Mean TTFP | Mean TTFT | Mean E2E | Audio duration |
| --- | ---: | ---: | ---: | ---: | ---: |
| Matched retained control | 0.299441 | 364.00 ms | 79.39 ms | 1848.64 ms | 62.68 s |
| Direct stop run 1 | 0.286807 | 350.88 ms | 79.53 ms | 1740.69 ms | 61.68 s |
| Direct stop run 2 | 0.297673 | 346.15 ms | 77.22 ms | 1625.63 ms | 56.72 s |
| Direct stop two-run mean | **0.292240** | **348.51 ms** | **78.37 ms** | **1683.16 ms** | - |

The two-run mean RTF is 2.40% below the matched control and mean TTFP is
4.25% lower. More importantly, the ten hot main-run Stage-1 ITL samples fell
from 9.284 ms/code in the control to 8.435 ms/code, a 9.14% reduction. This is
a direct measurement of the autoregressive loop and is not biased by output
audio length. The result confirms that eliminating repeated framework work is
currently more valuable than another isolated Stage-2 microkernel.

Artifacts:

```text
/tmp/lunanexa-bench/stage-isolated-numa-control-official10/
/tmp/lunanexa-bench/talker-direct-stop-official10-run1/
/tmp/lunanexa-bench/talker-direct-stop-official10-run2/
/tmp/minicpmo-direct-stop.log
```

The retained follow-up also removes construction of the deterministic logits
row. It keeps resident continue/stop logits views beside the resident token-ID
views, and uses one boolean control list as the source of truth for canonical
fallback, batched fast path, and the batch-one fast path. This also fixes the
edge case where a terminal request observed for one additional scheduler step
could previously be represented as continue by the direct sampler.

The cleanest comparison used the same 61.68 seconds / 1,480,320 generated
frames in both runs:

| Same-duration candidate | Mean RTF | Mean TTFP | Mean TTFT | Mean E2E | Stage-1 ITL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Direct token ID only | 0.286807 | 350.88 ms | 79.53 ms | 1740.69 ms | 8.435 ms/code |
| Static logits + token IDs | **0.280486** | **343.61 ms** | **79.13 ms** | **1698.25 ms** | **8.238 ms/code** |

Static control buffers reduce RTF by another 2.20%, TTFP by 2.07%, E2E by
2.44%, and Stage-1 ITL by 2.34%. Relative to the original 9.284-ms/code
matched control, the complete direct-control path reduces hot Talker ITL by
11.27%. It remains sequence-preserving: only the already deterministic
vLLM-visible stop channel changes implementation.

```text
/tmp/lunanexa-bench/talker-static-stop-buffers-official10/
/tmp/minicpmo-static-stop-buffers.log
```

A subsequent sparse-output sentinel experiment was rejected. It reused one
resident empty NPU tensor for all non-publishable codec steps and represented
terminal metadata as Python booleans until a chunk boundary. Despite removing
two empty-tensor allocations per code, the same-duration benchmark regressed
RTF from 0.280486 to 0.286711 (+2.22%), TTFP from 343.61 to 351.87 ms, and hot
Stage-1 ITL from 8.238 to 8.359 ms/code (+1.47%). The stable empty-tensor
address likely introduced alias/event dependencies that outweighed allocator
work. The implementation was fully removed; only the result artifact remains:

```text
/tmp/lunanexa-bench/talker-resident-empty-output-official10/
/tmp/minicpmo-resident-empty-output.log
```

### Main-bottleneck result: stable-input PagedAttention replay

A low-rate Stage-1 host profile identified graph-parameter maintenance as the
largest active Python stack: `update_full_graph_params` rebuilt twenty FIA
tasks after every Talker token because FIA represents query and KV lengths as
host-valued attributes. Reordering that work with ENPU was rejected at
0.308473 aggregate RTF, 12.00% slower than the retained 0.275414 control,
because it moved the same twenty updates onto the replay dependency path.

The retained candidate changes the operator contract instead. Stage-1
batch-one decode uses PagedAttention for capture size one. Its block table and
context-length tensor are fixed runner-owned buffers whose contents are
updated in place. With `enable_stable_pa_graph_inputs`, graph capture omits the
per-layer update handles and events, and replay executes the already captured
PA tasks directly. The opt-in is disabled by default and is paired with
`pa_shape_list: [1]` in the dedicated MiniCPM-o profile.

An attempted device-resident context-length variant was rejected during graph
capture: Atlas A2 ATB reported `PagedAttentionOperation setup failed`. The
compatible implementation therefore retains the existing pinned host length
slab. Two warmups and ten measured requests with varying generated lengths
confirmed that replay observes its in-place contents; all requests completed
with 100% streaming continuity.

Both official-shape repeats used concurrency one, seed zero, and temperature
zero. Aggregate RTF is wall-clock benchmark duration divided by actual audio
duration, so stochastic output length is normalized. Lower is better.

| Variant | Aggregate RTF | Mean TTFP | Mean TTFT | Mean E2E | Audio duration |
| --- | ---: | ---: | ---: | ---: | ---: |
| Previous retained static-control profile | 0.275414 | 343.61 ms | 79.13 ms | 1698.25 ms | 61.68 s |
| Stable PA run 1 | 0.233829 | 312.12 ms | 77.45 ms | 1426.80 ms | 61.04 s |
| Stable PA run 2 | 0.228323 | 306.59 ms | 77.31 ms | 1442.52 ms | 63.20 s |
| Stable PA two-run mean | **0.231076** | **309.36 ms** | **77.38 ms** | **1434.66 ms** | - |

The two-run mean improves aggregate RTF by 16.10%, TTFP by 9.97%, TTFT by
2.21%, and E2E by 15.52% relative to the retained profile. The first PA
request after process startup incurred about 62 seconds of ATB compilation;
the competition's declared warmup requests absorb this one-time cost, while
subsequent requests were stable at roughly 1.0--1.75 seconds E2E.

This is the first Talker change in this series that removes the profiled
twenty-layer control-plane mechanism rather than making each update slightly
cheaper. It remains an experimental submission candidate until the matched
official Seed-TTS WER/SIM gate is rerun; the current server lacks `funasr`, so
the performance run cannot substitute for that accuracy result. An export-only
quality run nevertheless produced 10/10 official utterance-name WAV files with
zero failures or missing PCM. All files are 24-kHz mono WAV, span 3.28--8.48
seconds, and total 61.52 seconds, ready for the organizer's `cal_wer.sh` and
`cal_sim.sh`.

Artifacts:

```text
/tmp/lunanexa-bench/talker-stable-pa-official10/
/tmp/lunanexa-bench/talker-stable-pa-official10-repeat/
/tmp/lunanexa-bench/talker-stable-pa-export10/
/tmp/lunanexa-quality/talker-stable-pa-seed10/
/tmp/minicpmo-talker-stable-pa-host-slab.log
```

### Main-bottleneck result: fingerprinted prompt-state templates

Synchronized Stage-2 timing split the hot first-packet path into two pieces:
prompt setup consumed about 39--41 ms (roughly 12 ms Conformer plus 26--28 ms
prompt CFM), while the first live width-13 chunk consumed about 67--69 ms
(roughly 12 ms encoder, 27 ms first-packet CFM, and 28 ms HiFT).  Bounding the
DiT prompt suffix to 150 frames did not attack the dominant cost and was
rejected at 0.237681 aggregate RTF / 320.22 ms TTFP.  Deferring reduced-CFM
cache materialization was likewise rejected at 0.239588 RTF / 315.04 ms TTFP.

The competition path repeatedly uses the same default reference-audio prompt,
yet Stage 2 rebuilt its deterministic Conformer and prompt-CFM state for every
request.  The retained candidate caches one immutable state template keyed by
the complete prompt-content fingerprint.  Each request receives independent
copies of its Conformer cache, estimator attention/CNN storage, rolling fixed
slabs, and HiFT state; the live decoder never mutates the template or another
request's state.  Different prompt content or paths select different cache
entries, and the small LRU is bounded.  This changes neither solver steps nor
tensor values; it removes repeated setup computation.

Five focused cache/fixed-slab tests pass on the A2 host, including address
non-aliasing and mutation isolation.  The first request after service startup
still performs normal setup and graph compilation, which the declared warmup
absorbs.  All later fixed-prompt requests clone the resident template.

Three runs used the same Chinese Seed-TTS wrapper, two warmups, ten measured
requests, concurrency one, seed zero, and temperature zero.  Lower is better.

| Run | Aggregate RTF | Mean TTFP | Mean TTFT | Mean E2E | Audio duration |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prompt-state cache 1 | 0.228856 | 275.99 ms | 78.59 ms | 1409.24 ms | 61.60 s |
| Prompt-state cache 2 | 0.250856 | 271.95 ms | 79.12 ms | 1467.50 ms | 58.52 s |
| Prompt-state cache 3 | 0.230909 | 269.56 ms | 77.03 ms | 1383.11 ms | 59.92 s |
| Three-run median | **0.230909** | **271.95 ms** | **78.59 ms** | **1409.24 ms** | - |

Compared with the retained stable-PA two-run mean (0.231076 RTF, 309.36 ms
TTFP, 77.38 ms TTFT, and 1434.66 ms E2E), median TTFP improves by 12.09% and
median E2E by 1.77%.  Median RTF is effectively flat but 0.07% lower; TTFT is
1.56% higher and remains within normal host variation.  The second run
contained a 2.17-second single-request host tail, so promotion uses the
predeclared median rather than selecting the fastest run.  Every run completed
10/10 requests with zero failures and 100% streaming continuity.

Artifacts:

```text
/tmp/lunanexa-bench/prompt-state-cache-smoke/
/tmp/lunanexa-bench/prompt-state-cache-official10/
/tmp/lunanexa-bench/prompt-state-cache-official10-repeat/
/tmp/lunanexa-bench/prompt-state-cache-official10-third/
/tmp/minicpmo-stable-pa-prompt-state-cache.log
```

A follow-up pinned-host D2H candidate was rejected at the smoke gate.  It
copied the first 23,040-byte FP32 waveform through a dedicated NPU copy stream
into pinned CPU storage and synchronized that stream before EngineCore IPC.
The path was confirmed active, but TTFP rose from the prompt-cache smoke's
280.74 ms to 297.54 ms (+5.99%).  For this small payload, pinned allocation,
stream handoff, and explicit synchronization cost more than the pageable D2H
copy.  The implementation and profile were reverted; the artifact remains at
`/tmp/lunanexa-bench/prompt-cache-pinned-d2h-smoke/`.

### Leaderboard refresh and static steady-CFM ABI

The public vLLM-Omni leaderboard was refreshed at `2026-08-27 13:15:46`.
The team's last organizer result was seventh at 0.2423 RTF, 514.22 ms TTFP,
and 45.72 ms TTFT.  The best individual values were 0.1278 RTF, 156.03 ms
TTFP, and 6.37 ms TTFT.  These values come from different submissions, but
they make the remaining priorities explicit: our TTFT is already comparable
with the RTF leader's 47.24 ms, while RTF and TTFP remain the large gaps.

The previous five-step experiment did not rebuild the serving ABI.  It entered
the reduced-step eager path and expanded the result back into six cache slots,
so it was slower despite performing less model arithmetic.  The new CFM4 and
CFM3 profiles instead set the complete Token2Wav solver width before backend
construction.  Timeline tensors, all-step AdaLN, fixed estimator K/V slabs,
direct cache outputs and the outer steady NPUGraph are consequently native
four- or three-slot executables.  Prompt-cache prefill and the first live
packet remain one-step; Talker sampling and chunk boundaries are unchanged.

The first CFM4 official-shape run used two warmups followed by ten Chinese
Seed-TTS requests at concurrency one, seed zero and temperature zero.  The
first warmup absorbed 62 seconds of ATB/NPUGraph compilation.  All ten measured
requests completed with 100% streaming continuity.

| Variant | Aggregate RTF | Mean TTFP | Mean TTFT | Mean E2E | Audio duration |
| --- | ---: | ---: | ---: | ---: | ---: |
| CFM6 prompt-state-cache median | 0.230909 | 271.95 ms | 78.59 ms | 1409.24 ms | - |
| Native static CFM4 run 1 | **0.218146** | 279.26 ms | **76.77 ms** | **1228.11 ms** | 56.32 s |
| Native static CFM3 run 1 | **0.208385** | **275.25 ms** | 77.76 ms | **1173.12 ms** | 56.32 s |

Static CFM4 improves aggregate RTF by 5.53% and mean E2E by 12.85%.  Its
2.69% TTFP movement is treated as run variance because both profiles execute
the same prompt-CFM1 and first-packet-CFM1 path.  Unlike the prompt-state cache,
steady step reduction changes model numerics; CFM4 and CFM3 therefore remain
accuracy-gated until the paired Seed-TTS WER/SIM screen and the final
Daily-Omni and Video-MME gates pass.

Native CFM3 completed the same ten-request shape without a failed request or
stream discontinuity.  Relative to the retained CFM6 median, it lowers
aggregate RTF by 9.76% and mean E2E by 16.76%; relative to native CFM4 it lowers
aggregate RTF by another 4.47%.  It is the performance candidate, not yet the
submission candidate: the missing accuracy gates are intentionally not
inferred from transport success.

```text
/tmp/lunanexa-bench/cfm4-static-official10/
/tmp/minicpmo-cfm4-static.log
/tmp/lunanexa-bench/cfm3-static-official10/
/tmp/minicpmo-cfm3-static.log
```

### BF16-bounded final Addcmul gate

The final DiT modulation path had already qualified an exact `addcmul`
rewrite on a retained 32-row WER/SIM screen, but the startup parity gate still
used a fixed FP32 `1e-6` absolute threshold.  On the live BF16 estimator the
canonical multiply/add expression and `addcmul` differ by exactly one BF16
storage ULP (`0.0078125`), so every service start silently disabled the
qualified path.  The gate now keeps the FP32 bound unchanged and allows at
most one storage epsilon for FP16/BF16.  Startup logged:

```text
Validated dtype-bounded MiniCPM-o final Addcmul path;
max_abs_drift=0.0078125, limit=0.0078125
```

The controlled CFM3 A/B retained the same 56.32 seconds of output, 10/10
success and 100% continuity:

| CFM3 variant | Aggregate RTF | Mean TTFP | Mean TTFT | Mean E2E |
| --- | ---: | ---: | ---: | ---: |
| Fixed FP32 gate, fusion disabled | 0.208385 | 275.25 ms | 77.76 ms | 1173.12 ms |
| Dtype-aware gate, fusion active | **0.208119** | **273.43 ms** | 78.61 ms | **1171.62 ms** |

The 0.13% aggregate-RTF movement is too small to call a major service gain,
but the fix prevents an already-qualified fast path from being incorrectly
disabled.  It also confirms that the remaining leaderboard gap is dominated
by the Talker execution cycle, not this isolated Stage-2 launch.

```text
/tmp/lunanexa-bench/cfm3-static-addcmul-official10/
/tmp/minicpmo-cfm3-static-addcmul.log
```

### Exact n-gram speculative Talker experiment

The next isolated profile attacked the 83.5% idle fraction in the hot Talker
trace.  It proposed three codec tokens from matching one- to three-token
suffixes in the already-generated sequence, then verified them with the
ordinary Talker in one wider target forward.  Rejected drafts fall back to the
target token, so this was an exact speculative execution mechanism rather than
a trained or approximate multi-code head.

The ordinary CPU n-gram proposer was rejected by vLLM configuration because
the retained Talker requires asynchronous scheduling.  The device n-gram
proposer was compatible after rebuilding the Stage-1 graph ABI at width four:
it captured successfully as `FULL_AND_PIECEWISE` and completed 10/10 measured
requests with 100% streaming continuity.  It was nevertheless a decisive
service regression:

| Talker path | Aggregate RTF | Mean TTFP | Mean TTFT | Mean E2E | Audio duration |
| --- | ---: | ---: | ---: | ---: | ---: |
| CFM3 + one-token full-decode graph | **0.208119** | **273.43 ms** | **78.61 ms** | **1171.62 ms** | 56.32 s |
| CFM3 + NGram-GPU 3-token drafts | 0.459330 | 394.86 ms | 81.22 ms | 2705.69 ms | 58.92 s |

The speculative candidate regressed aggregate RTF by 120.71% and TTFP by
44.41%.  Codec suffix acceptance cannot amortize the wider target verification
and the loss of the specialized one-token `FULL_DECODE_ONLY` executable on
this A2 stack.  The profile and its test were removed rather than leaving a
dormant trap.  A useful multi-code path now requires a trained compatible
draft/MTP head or a backend device loop that preserves the efficient one-token
executable; prompt n-gram reuse is closed.

```text
/tmp/lunanexa-bench/cfm3-talker-ngram3-official10/
/tmp/minicpmo-cfm3-talker-ngram3-v3.log
```

### Current CFM3 Talker trace and chunk-boundary EOS candidate

A fresh Stage-1-only trace was captured on the current static-CFM3 service,
after stable PagedAttention and the fused inverse-CDF sampler had landed.  The
1.067-second Talker window contained 390.45 ms of device compute and 676.25 ms
of free time: the device was still idle for 63.4% of the observed interval.
MatMulV2 and PagedAttention accounted for 40.36% and 32.90% of device compute,
respectively, but the host trace exposed the more actionable serialization:

| Host/API event | Count | Total time |
| --- | ---: | ---: |
| `aclrtSynchronizeStreamWithTimeout` | 229 | 177.12 ms |
| `_local_scalar_dense` operator rows | about 141 | about one per codec step |
| `aclnnInplaceUniform` | 140 | 10.03 ms |
| `aclrtRandomNumAsync` | 140 | 6.65 ms |
| `_compute_slot_mapping_kernel` | 141 | 26.66 ms device time |

The ordinary Talker reads the sampled EOS scalar after every eligible codec
token.  Sparse transport, however, makes codec output visible to Code2Wav only
at the 10-frame initial and 25-frame steady boundaries.  The experimental
`VLLM_OMNI_MINICPMO45_NPU_DEFERRED_CHUNK_EOS` path therefore retains samples
on-device, reads one vector at the existing publish boundary, finds the first
EOS there, and publishes only the prefix before EOS.  The max-token boundary
still drops its sampled code.  Native-duplex and non-sparse paths retain the
ordinary immediate EOS behavior.

The matched test used the same static-CFM3 base, two warmups, ten fixed Chinese
Seed-TTS prompts, concurrency one, seed zero and temperature zero.  Both runs
completed 10/10 with 100% continuity.

| Variant | Aggregate RTF | Mean TTFP | Mean TTFT | Mean E2E | Serving duration | Audio duration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Per-token EOS control | 0.208119 | 273.43 ms | **78.61 ms** | 1171.62 ms | 11.721 s | 56.32 s |
| Chunk-boundary EOS | **0.171280** | **267.86 ms** | 79.47 ms | **1023.84 ms** | **10.243 s** | 59.80 s |

Lower is better except audio duration.  The candidate lowers aggregate RTF by
17.70%, mean E2E and total serving time by 12.61%, and TTFP by 2.04%; TTFT
moves +1.09%, inside the 2% guard.  Mean middle-chunk RTF computed directly
from raw samples falls from 0.15341 to 0.11607 (-24.34%), and continuity stays
at 100%.

This is a real execution-speed improvement, not merely a denominator effect:
wall-clock serving time fell even though the run generated 6.18% more audio.
That duration change also means the optimization is not bit-exact.  It remains
an accuracy-gated candidate until paired Seed-TTS WER/SIM, Daily-Omni and
Video-MME stay within the competition's two-point allowance.  The missing
derived summary fields in the candidate JSON are a benchmark-reporting issue;
the artifact contains all ten TTFP/E2E arrays and all 70 per-chunk RTF arrays,
which were used for the figures above.

```text
/tmp/vllm-omni-profiles/minicpmo45/a2-cfm3-current-stage1/
/tmp/lunanexa-bench/cfm3-deferred-eos-official10/
/tmp/minicpmo-cfm3-deferred-eos.log
```

The follow-up request-local RNG-slab experiment replaced 140 scalar
`uniform_` launches with one request-wide random fill and a per-step device
slice copy.  It generated the same 59.80 seconds / 1,435,200 frames as the
chunk-boundary-EOS control, but regressed aggregate RTF from 0.171280 to
0.173524 (+1.31%), mean E2E from 1023.84 to 1037.17 ms (+1.30%), middle-chunk
RTF from 0.11607 to 0.11746 (+1.19%), and TTFP by 0.50%.  The additional
per-step copy/index dependency costs more than the eliminated random launch
on this A2 stack.  The implementation and deployment switch were removed;
the negative artifact remains at:

```text
/tmp/lunanexa-bench/cfm3-deferred-eos-rng-slab-official10/
/tmp/minicpmo-cfm3-deferred-eos-rng-slab.log
```

### Post-EOS trace and fixed-codec-slab rejection

The retained chunk-boundary-EOS path was profiled again rather than assuming
the previous trace still described it.  For the same 140-code hot request,
`aclrtSynchronizeStreamWithTimeout` fell from 177.12 ms to 1.94 ms.  The
Talker device window fell from 1066.70 to 864.79 ms, and free time from 676.25
to 464.18 ms.  Device compute was 400.61 ms and free time remained 53.7%, so
the dominant remaining budget is the per-code vLLM scheduler/IPC round trip,
not scalar synchronization.  Slot mapping remained 26.69 ms, only 6.66% of
device compute; its previously rejected isolated graph cannot close the
remaining gap.

A Python stack sample exposed recurrent sampled-code clones plus 147 Cat
calls.  A fixed 16-code history ring and two 25-code ping-pong transport slabs
removed those allocations and concatenations.  It passed five focused
semantic tests and completed 10/10 requests with 100% continuity, but failed
both structure and performance gates:

| Variant | Aggregate RTF | Mean TTFP | Mean E2E | Middle-chunk RTF | Audio |
| --- | ---: | ---: | ---: | ---: | ---: |
| Retained deferred EOS | **0.171280** | **267.86 ms** | **1023.84 ms** | **0.11607** | 59.80 s |
| Fixed history/transport slabs | 0.176502 | 276.59 ms | 1046.48 ms | 0.11702 | 59.32 s |

The candidate regressed RTF by 3.05%, TTFP by 3.26%, E2E by 2.21%, and also
changed output length.  Per-token view copies/dependencies cost more than the
removed small Cats on this A2 stack.  The implementation and profile were
removed.  The next material architecture target is a Talker device loop or
backend multi-step execution that amortizes scheduler crossings while
preserving the efficient one-token Llama executable.

```text
/tmp/vllm-omni-profiles/minicpmo45/a2-cfm3-deferred-eos-stage1/
/tmp/minicpmo-cfm3-deferred-eos-stage1.raw
/tmp/lunanexa-bench/cfm3-fixed-codec-slabs-official10/
```

### Graph-internal codec embedding rejection

An additional Stage-1 candidate connected the standalone inverse-CDF sampler
to the existing one-token Talker graph through a fixed resident NPU scalar.
The graph performed the small codec embedding at ingress, while the runner
skipped its ordinary Python decode preprocess, eager embedding launch and
`inputs_embeds` copy.  The vocabulary-wide codec head deliberately remained
outside the Talker graph because the earlier fused-head experiment was much
slower.

The implementation passed its focused fixed-address and graph-switch tests,
and both measured runs completed 10/10 requests with 100% streaming
continuity.  It nevertheless failed the determinism and performance gates:

| Variant | Aggregate RTF | Mean reported RTF | Mean TTFP | Mean TTFT | Mean E2E | Audio duration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Retained deferred EOS | **0.171280** | - | **267.86 ms** | **79.47 ms** | 1023.84 ms | 59.80 s |
| Internal embedding run 1 | 0.170075 | 0.173272 | 276.73 ms | 79.94 ms | 1012.46 ms | 59.56 s |
| Internal embedding run 2 | 0.172009 | 0.175373 | 271.09 ms | 79.82 ms | **982.02 ms** | 57.12 s |

Aggregate RTF is serving duration divided by generated audio duration; lower
is better.  Its marginal movement changed sign across repeats, while TTFP
regressed in both runs.  More importantly, the same ten prompts, seed zero and
temperature zero produced different total frame counts in all three rows.
The candidate's resident sampled-code scalar crosses the asynchronous engine
state boundary and can be overwritten before every downstream consumer has
materialized the prior value.  This is not a safe fixed-address ABI even
though the next Talker replay itself reads the correct address.  The code and
profile were removed rather than retaining a nondeterministic fast path.

The result narrows the next architecture change: a multi-code device loop must
own sampling, history, EOS and the repeated Talker invocation together.  A
single mutable tensor cannot be exported through the current per-step engine
contract as both graph input and scheduler-visible request state.

```text
/tmp/lunanexa-bench/internal-codec-embed-official10/
/tmp/lunanexa-bench/internal-codec-embed-official10-repeat/
/tmp/minicpmo-cfm3-internal-codec-embed.log
```

### Synchronous Talker scheduler rejection

The post-EOS Python profile showed `EngineCore.step_with_batch_queue` on every
codec step, so a matched candidate disabled asynchronous scheduling only for
Stage 1.  The hypothesis was that a single-request benchmark had no second
request whose work could justify the queue handoff.  Startup confirmed that
Thinker remained asynchronous, Talker was synchronous, and the existing
one-token decode graph captured and replayed normally.

The queue was not wasted overhead: it overlaps CPU output processing with the
next NPU decode.  Removing it serialized those phases and regressed the full
10-request test despite 10/10 success and 100% streaming continuity:

| Variant | Aggregate RTF | Mean reported RTF | Mean TTFP | Mean TTFT | Mean E2E |
| --- | ---: | ---: | ---: | ---: | ---: |
| Retained asynchronous Talker | **0.171280** | - | **267.86 ms** | 79.47 ms | **1023.84 ms** |
| Synchronous Talker | 0.194153 | 0.196695 | 276.38 ms | **78.45 ms** | 1226.56 ms |

Aggregate RTF regressed about 13.3% and E2E about 19.8%.  The profile was
removed.  A future device-loop implementation should retain asynchronous
request handling around the loop; eliminating the outer batch queue is not a
substitute for amortizing multiple decode steps inside one worker execution.

```text
/tmp/lunanexa-bench/cfm3-sync-talker-official10/
/tmp/minicpmo-cfm3-sync-talker.log
```

### Retained five-code first-packet profile on the current CFM3 stack

The earlier five-code screen predated prompt-state reuse, native CFM3 and
chunk-boundary EOS, so its 566 ms result did not answer whether the minimum
packet helps the current stack.  The new profile changes only Stage 1's first
transport boundary from ten codes to five.  With the three-code left context,
HiFT receives ten mel frames and publishes its minimum non-empty 40 ms packet.
Width 10 intentionally stays eager because this A2 CANN release previously
aborted its captured graph at first replay; width-50 steady work retains the
proven graph.

Two independent runs used two warmups, ten measured Chinese Seed-TTS prompts,
concurrency one, seed zero and temperature zero.  Both completed 10/10 with no
failures and 100% streaming continuity:

| Variant | Aggregate RTF | Mean reported RTF | Mean TTFP | Mean TTFT | Mean E2E | Audio duration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Retained first-10 control | 0.171280 | - | 267.86 ms | **79.47 ms** | 1023.84 ms | 59.80 s |
| First-5 run 1 | 0.172712 | 0.176069 | **253.09 ms** | 79.36 ms | **970.83 ms** | 56.24 s |
| First-5 run 2 | **0.165697** | **0.168205** | 255.55 ms | 80.02 ms | 1046.71 ms | 63.20 s |
| First-5 pooled/mean | **0.169000** | 0.172137 | **254.32 ms** | 79.69 ms | **1008.77 ms** | 119.44 s total |

Pooled aggregate RTF is total serving duration divided by total generated
audio across both runs; other pooled-row latency values are two-run means.
Relative to the first-10 control, the new profile improves aggregate RTF by
1.33%, mean TTFP by 5.05%, and mean E2E by 1.47%.  TTFT moves by 0.27% and is
effectively unchanged.  The repeated TTFP gain shows that the saved five
Talker steps outweigh the eager width-10 Code2Wav launch on the current stack.

This becomes the retained low-TTFP/high-score profile.  It does not remove the
existing accuracy gate: native CFM3 and chunk-boundary EOS already require the
official Seed-TTS WER/SIM, Daily-Omni and Video-MME checks, and changing the
first HiFT partition must be covered by the same end-to-end audio screen.

```text
/tmp/lunanexa-bench/cfm3-deferred-eos-i5-official10/
/tmp/lunanexa-bench/cfm3-deferred-eos-i5-official10-repeat/
/tmp/minicpmo-cfm3-deferred-eos-i5.log
```

### Selective-BF16 HiFT rejection on A2

The next Stage-2 candidate moved HiFT's F0 predictor, pre-convolution,
upsamplers, source downsamplers, residual blocks and final convolution to
BF16.  Harmonic-source construction, phase accumulation, STFT/ISTFT and the
published waveform remained FP32 precision islands.  Startup confirmed that
the candidate was active rather than silently falling back.

Two matched runs used the retained native-CFM3, deferred-EOS and five-code
first-packet stack.  They produced exactly the same 56.24 and 63.20 seconds of
audio as the two retained control runs, completed 10/10 requests, and kept
100% streaming continuity.  Lower is better:

| Variant, pooled/two-run mean | Aggregate RTF | Mean reported RTF | Mean TTFP | Mean TTFT | Mean E2E |
| --- | ---: | ---: | ---: | ---: | ---: |
| Retained FP32 HiFT | **0.169000** | **0.172137** | 254.32 ms | 79.69 ms | **1008.77 ms** |
| Selective BF16 HiFT | 0.171575 | 0.174798 | **252.88 ms** | **77.66 ms** | 1024.16 ms |

Selective BF16 improves mean TTFP by only 0.57% and TTFT by 2.55%, while
regressing pooled aggregate RTF by 1.52%, mean reported RTF by 1.55%, and E2E
by 1.53%.  The first run is an especially clean paired comparison because its
audio length is identical: serving time rose from 9.713 to 9.944 seconds and
aggregate RTF regressed 2.37%.  A2's BF16 convolution path does not amortize
the boundary casts and backend/layout overhead at these small batch-one HiFT
shapes.  The implementation and profile were removed; future lower-precision
HiFT work must use a graph-visible, layout-propagated boundary rather than
per-module dtype conversion.

```text
/tmp/lunanexa-bench/cfm3-i5-hift-bf16-official10/
/tmp/lunanexa-bench/cfm3-i5-hift-bf16-official10-repeat/
/tmp/minicpmo-cfm3-i5-hift-bf16.log
```

### Native static CFM2 performance/accuracy candidate

The previous reduced-step rejection changed a loop bound but retained a wider
serving ABI.  This candidate instead sets the complete steady Code2Wav solver
width to two before backend construction.  Timeline tensors, all-step AdaLN,
fixed estimator cache slabs, direct outputs and the outer graph are therefore
native two-slot objects.  Prompt prefill and the first live packet retain
their one-step schedules.

Two matched runs completed 10/10 requests with 100% streaming continuity and
produced exactly the same 56.24 and 63.20 seconds of audio as the retained
CFM3 runs.  Lower is better:

| Variant, pooled/two-run mean | Aggregate RTF | Mean reported RTF | Mean TTFP | Mean TTFT | Mean E2E |
| --- | ---: | ---: | ---: | ---: | ---: |
| Retained native CFM3 | 0.169000 | 0.172137 | **254.32 ms** | **79.69 ms** | 1008.77 ms |
| Native static CFM2 | **0.161572** | **0.164286** | 260.21 ms | 80.69 ms | **964.40 ms** |

Native CFM2 lowers pooled aggregate RTF by 4.40%, mean reported RTF by 4.56%
and E2E by 4.40%.  TTFP regresses 2.32% despite using the same one-step first
packet, and TTFT moves 1.25%; those movements confirm that steady CFM is no
longer the dominant first-packet path.  CFM2 changes diffusion numerics and is
retained only as a performance/accuracy candidate until official Seed-TTS
WER/SIM passes within the two-point allowance.  It must not replace CFM3 in a
submission based on transport success alone.

```text
/tmp/lunanexa-bench/cfm2-deferred-eos-i5-official10/
/tmp/lunanexa-bench/cfm2-deferred-eos-i5-official10-repeat/
/tmp/minicpmo-cfm2-deferred-eos-i5.log
```

### In-process Talker EngineCore rejection

A Stage-1 architecture candidate removed the per-token ZMQ/msgpack boundary by
hosting Talker's EngineCore on a dedicated thread in the API process.  The
second revision also ran the complete single-request EngineCore loop on that
thread, so codec steps no longer round-tripped through the asyncio event loop;
only real streaming outputs were delivered back to the orchestrator.  Thinker
and Code2Wav retained their ordinary subprocess isolation, and Talker retained
asynchronous scheduling and the one-token decode graph.

The complete-loop revision passed its cancellation/output-race tests and the
real run completed 10/10 requests with 100% streaming continuity.  It reduced
the first per-step inline prototype slightly, but remained decisively slower
than the retained process-isolated stack.  Lower is better:

| Variant | Aggregate RTF | Mean chunk RTF | Mean TTFP | Mean TTFT | Mean E2E |
| --- | ---: | ---: | ---: | ---: | ---: |
| Retained native CFM3 | **0.169000** | **0.172137** | **254.32 ms** | 79.69 ms | **1008.77 ms** |
| Inline EngineCore, per-step event-loop handoff | 0.2733 | 0.28 | 429.28 ms | 81.83 ms | 1359.14 ms |
| Inline EngineCore, continuous worker drain | 0.265986 | 0.276008 | 422.67 ms | **76.95 ms** | 1283.70 ms |

The continuous loop proves that Python event-loop resubmission was only a
small part of the regression.  Moving Talker's NPU runtime into the API
process introduces more expensive GIL, host-thread and NPU-runtime contention;
the original EngineCore subprocess already keeps its own efficient scheduler
loop, so ZMQ is not the dominant Talker cost.  The inline runtime, profile and
tests were removed.  A future multi-code Talker optimization must stay inside
the isolated worker and fuse several model steps behind one engine command,
instead of moving the entire EngineCore across the process boundary.

```text
/tmp/lunanexa-bench/cfm3-i5-talker-inline-v2-official10/
/tmp/lunanexa-bench/cfm3-i5-talker-inline-drain-official10-valid/
/tmp/minicpmo-cfm3-i5-talker-inline-drain.log
```

### Evaluator-visible source policy and official-protocol qualification

The organizer installs the submitted Python source but supplies
`vllm_omni/deploy/minicpmo_4_5.yaml` from the current `minicpm-challenge`
baseline. Candidate-only deployment profiles therefore do not affect the
score. The single-chip source policy now fills only absent, output-preserving
Talker and Code2Wav settings: batched codec transport, deferred chunk EOS,
direct binary stop control, prompt-state templates, HiFT weight-normalization
materialization, and event-backed shared-memory wakeup.
Explicit deploy values retain authority, and
`VLLM_OMNI_MINICPMO45_SINGLE_CHIP_EXACT_DEFAULTS=0` provides a matched rollback
without disabling the existing Stage-0/1 decode graphs or connector events.

The official A3 YAML cannot initialize Stage 0 on the available 32-GiB 910B4,
so `minicpmo_4_5_1npu_a2_evaluator_compat.yaml` changes capacity planning only.
It deliberately carries no CFM, chunk-boundary, dtype, sampler, or model
numeric overrides. With the source defaults and native CFM6, ten fixed Chinese
Seed-TTS requests completed 10/10 with 100% continuity. A second service was
then launched with
`VLLM_OMNI_MINICPMO45_SINGLE_CHIP_EXACT_DEFAULTS=0`; it retained the same
Stage-0/1 decode graphs and event-backed shared memory, but removed only the
new exact producer/consumer defaults. Lower is better:

| CFM6 source policy | Prompts | Mean audio RTF | Mean chunk RTF | Mean TTFP | Mean TTFT | Mean E2E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Exact defaults | 10 | **0.387873** | **0.404148** | **594.21 ms** | **78.34 ms** | **1758.29 ms** |
| Exact-default rollback | 10 | 0.426696 | 0.451212 | 873.99 ms | 82.16 ms | 2035.45 ms |

This exploratory matched run suggested that the bundle could improve the
evaluator's primary all-chunk RTF by **10.43%**,
mean request RTF by **9.10%**, TTFP by **32.01%**, TTFT by **4.65%**, and E2E
by **13.62%**. The generated totals differ by one 1.32-second sampled audio
chunk (46.96 versus 48.28 seconds), but both per-request RTF and the official
flattened chunk statistic independently show a large win. This matched
rollback justified an official-protocol follow-up; it did not qualify the
bundle for submission.

These first A2 policy runs also inherited the exploratory server flag
`--interleave-mm-strings`. The official Seed-TTS server fixture intentionally
omits that flag because interleaving and TTS `ref_audio` must not share the
Daily-Omni request path. The relative exact/rollback A/B remains useful because
both sides used the same server, but final promotion metrics and all accuracy
screens must be repeated with only the official deploy config plus
`--trust-remote-code`.

The reduced-solver experiments use that retained exact policy:

| Native solver | Prompts | Mean audio RTF | Mean chunk RTF | Mean TTFP | Mean TTFT | Mean E2E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| CFM6 | 10 | 0.387873 | 0.404148 | 594.21 ms | 78.34 ms | 1758.29 ms |
| CFM5 | 32 | 0.310151 | 0.320209 | 556.73 ms | 78.44 ms | 1710.61 ms |
| CFM2 | 10 | 0.264708 | 0.268982 | 449.46 ms | 78.81 ms | 1209.94 ms |
| CFM2 | 32 | **0.257140** | **0.260838** | **440.75 ms** | **76.99 ms** | **1118.29 ms** |

CFM2 reduced matched ten-row RTF by 31.8%, TTFP by 24.4%, and E2E by
31.2%, confirming that solver arithmetic—not transport microseconds—is the
dominant remaining score budget. It remains an unqualified performance
candidate. The first 32-row Paraformer screens used the deterministic
performance setting `temperature=0` and reported mean WER `1.202915` for CFM2
and `1.246473` for CFM5. A subsequent CFM6 control under that same setting also
reported catastrophic WER (`1.3103`), proving that this screen measured an
unusable argmax Talker codec distribution rather than reduced-CFM accuracy.
Those WER numbers must not be used to accept or reject any solver width.

The official Seed-TTS accuracy test leaves temperature unset so the server's
model generation config controls Talker sampling, and runs concurrency four.
CFM2 and CFM5 therefore remain opt-in research controls until they are rerun
under that exact protocol. A distilled few-step flow map/student remains the
safer large-gain route if the correctly controlled native reductions fail.

The corrected CFM6 control, with no interleave flag, 32 fixed Chinese rows,
two warmups, concurrency four and server-default temperature, passed the
quality gate decisively. The first candidate also enabled Talker
`weight_nz_mode=2` and stable PA graph inputs. That pair was rejected: it
changed the sampled codec/audio distribution, failed WER catastrophically and
did not improve the concurrent run. Lower is better except audio throughput:

| Official-protocol CFM6 | Duration | Audio throughput | Mean chunk RTF | Mean TTFP | Mean TTFT | Mean WER |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Rollback control | 78.09 s | 2.0311 audio-s/s | 2.0222 | **3850.48 ms** | **111.20 ms** | **0.00865** |
| NZ2 + stable-PA candidate | 85.93 s | 1.9816 audio-s/s | 2.0531 | **2917.34 ms** | 111.45 ms | 1.22731 |
| Safe exact paths, no NZ/stable PA | **69.11 s** | **2.2950 audio-s/s** | **1.7978** | 3884.46 ms | 116.07 ms | **0.00865** |

The concurrent quality run includes queue stalls in per-chunk RTF and TTFP,
so those absolute values are not leaderboard numbers. They are valid for this
paired rejection. The candidate generated 170.28 seconds of audio versus
158.60 seconds for the control and missed the `0.0156` WER gate by two orders
of magnitude. NZ2 and stable PA were therefore removed from evaluator-visible
defaults. The remaining transport/cache/HiFT exact paths generated the same
158.60 seconds of audio as the control and exactly matched its mean WER. They
lowered benchmark duration by **11.50%**, mean chunk RTF by **11.10%**, and
mean E2E latency by **11.58%**, while increasing audio throughput by
**13.00%**. Mean TTFP moved 0.88% and TTFT 4.38% in the wrong direction, so
those first-response movements are treated as noise/regression rather than a
claimed win. The safe exact bundle is promoted; the rejected layout leaves
remain experimental-only.

A separate concurrency-one run of the promoted CFM6 candidate completed
32/32 requests with 100% continuity: mean chunk RTF `0.39525`, median chunk
RTF `0.25438`, mean TTFP `621.37 ms`, and mean TTFT `80.95 ms` on the available
910B4. These A2 absolute numbers are not compared directly with the single
910C leaderboard, but they provide the local target for reduced-solver
qualification.

Stage 2 previously consumed and discarded the complete lazy parent-checkpoint
iterator before loading its independently owned `flow.pt` and `hift.pt`.
Skipping that unowned 17.46-GiB safetensors scan reduced the observed Stage-2
initialization-to-API-ready interval from about 465 seconds to 148 seconds on
the GlusterFS-backed A2 host, a **68%** reduction (about 5 minutes 17 seconds).
The already running service imported this change when Stage 2 was spawned, so
the passing safe-exact performance/WER run above also exercised the corrected
loader. This changes startup only, not model tensors or scored inference.

At the 2026-08-27 23:58:38 leaderboard refresh, `向量贴贴` ranked ninth at
RTF `0.2423`, TTFP `514.22 ms` and TTFT `45.72 ms`. The RTF leader reported
`0.1066`, while the best observed first-response entry reported TTFP
`156.03 ms` and TTFT `6.37 ms`. The primary gap is now Stage-2 throughput:
closing the RTF gap requires about a 56% reduction from the submitted score,
not another microsecond-scale transport fusion.

The quality run also found that FunASR's default `AutoModel` initialization
performs an update check even when the Paraformer checkpoint is already
cached. The evaluator now passes `disable_update=True`, with compatibility
fallbacks for older FunASR releases. This removes an external-network hang
from repeatable accuracy qualification without changing ASR results.

### Correct-protocol two-step CFM promotion

The reduced native solvers were rerun without `--interleave-mm-strings`, with
server-default temperature, two warmups, concurrency four, and the same 32
fixed Chinese Seed-TTS rows. Each run generated the same 158.60 seconds of
audio as the retained CFM6 control and completed 32/32 requests with 100%
streaming continuity. Higher is better for throughput and SIM; lower is better
for the other columns:

| Solver | Duration | Audio throughput | Mean TTFT | Mean WER | WavLM-base-plus SIM |
| --- | ---: | ---: | ---: | ---: | ---: |
| CFM6 retained control | 69.11 s | 2.295 audio-s/s | 116.07 ms | 0.00865 | prior retained screens about 0.845 |
| CFM3 | 41.55 s | 3.82 audio-s/s | 96.39 ms | 0.0101 | **0.84858** |
| CFM2 | **40.60 s** | **3.91 audio-s/s** | **96.26 ms** | **0.0072** | 0.84240 |

Relative to CFM6, CFM2 reduced the complete concurrent batch duration by
**41.25%** and increased generated-audio throughput by **70.37%**, without
shortening the output. Its WER is below the organizer's `0.0156` gate and its
WavLM-base-plus similarity is 15.34 points above the `0.689` gate. CFM3 passed
too, but was slower and had worse WER on the same screen, so it is not the
submission default.

The upstream Seed-TTS fine-tuned WavLM-SV protocol was also run for diagnostic
parity. Its absolute scores are not comparable with the competition's `0.689`
WavLM-base-plus threshold: CFM2 scored `0.25287` and CFM3 scored `0.27367`,
while historical MiniCPM-o controls on other retained subsets were also near
zero. Those figures are retained as cross-checks, not used as the competition
admission metric.

The first evaluator-visible one-chip promotion filled an absent Stage-2
`VLLM_OMNI_MINICPMO45_TOKEN2WAV_N_TIMESTEPS` with `2`. An explicit stage or
launch environment value remains authoritative. Setting
`VLLM_OMNI_MINICPMO45_SINGLE_CHIP_CFM2_DEFAULT=0` disables only this numerical
default; setting `VLLM_OMNI_MINICPMO45_SINGLE_CHIP_EXACT_DEFAULTS=0` retains the
broader matched rollback. The complete official Seed-TTS pool remains a final
release gate before submission; this 32-row promotion is the fail-fast screen,
not a claim that the full pool has already run.

A fresh source-visibility run then launched the generic evaluator-compatible
deploy with no timestep override. The policy log included `code2wav-cfm2`, and
Stage 2 reported `token2wav_n_timesteps=2`, proving that the organizer-visible
source path—not a private profile or benchmark environment—selected the new
solver. Its 32-row concurrency-one result is directly comparable with the
retained CFM6 run on the same A2 host:

| Metric | CFM6 | Source-default CFM2 | Change |
| --- | ---: | ---: | ---: |
| Mean flattened chunk RTF | 0.39525 | **0.32233** | **-18.45%** |
| Median flattened chunk RTF | 0.25438 | **0.23707** | **-6.80%** |
| P99 flattened chunk RTF | not retained | 0.72869 | — |
| Mean TTFP | 621.37 ms | **491.92 ms** | **-20.83%** |
| Mean E2E | 1797.41 ms | **1504.50 ms** | **-16.29%** |
| Mean TTFT | 80.95 ms | **76.01 ms** | **-6.10%** |
| Audio throughput | 2.7066 audio-s/s | **3.2933 audio-s/s** | **+21.67%** |

The smaller concurrency-one gain relative to the concurrency-four duration
gain shows that Talker and per-chunk HiFT/orchestration are now a larger share
of the critical path. A further solver-only reduction cannot close the full
leaderboard gap by itself.

### Quality-gated one-step CFM promotion

The same official-protocol screen was then repeated with native CFM1. It
completed 32/32 requests with 100% streaming continuity and passed both quality
gates:

| Solver | Concurrent duration | Mean WER | WavLM-base-plus SIM |
| --- | ---: | ---: | ---: |
| CFM6 retained control | 69.11 s | 0.00865 | about 0.84485 |
| CFM2 | **40.60 s** | **0.0072** | **0.84240** |
| CFM1 | 41.48 s | 0.0087 | 0.83694 |

CFM1's SIM loss is only 0.79 percentage points relative to the retained CFM6
proxy, inside the two-point allowance, and its WER remains below `0.0156`.
Concurrency four did not improve because the reduced solver is no longer the
dominant resource at that load. At concurrency one, where the leaderboard's
per-request latency metrics are visible, it retained a smaller but measurable
win over CFM2:

| Metric | CFM2 | CFM1 | Change |
| --- | ---: | ---: | ---: |
| Mean flattened chunk RTF | 0.32233 | **0.30273** | **-6.08%** |
| Median flattened chunk RTF | 0.23707 | **0.22418** | **-5.44%** |
| P99 flattened chunk RTF | 0.72869 | **0.63560** | **-12.77%** |
| Mean TTFP | 491.92 ms | **455.93 ms** | **-7.32%** |
| Mean E2E | 1504.50 ms | **1392.11 ms** | **-7.47%** |
| Complete batch duration | 48.16 s | **44.56 s** | **-7.46%** |

The CFM1 performance run generated 155.72 seconds of audio versus CFM2's
158.60 seconds, a 1.82% length difference that is inside the quality allowance
but means the latency delta is not purely compute. The source default therefore
uses a rollback ladder rather than deleting the safer candidate. With no
explicit timestep value, CFM1 is selected. Setting
`VLLM_OMNI_MINICPMO45_SINGLE_CHIP_CFM1_DEFAULT=0` falls back to CFM2; also
setting `VLLM_OMNI_MINICPMO45_SINGLE_CHIP_CFM2_DEFAULT=0` restores CFM6.
An explicit launch or Stage-2 timestep value still wins over every default.
The complete 2020-row official pool remains the release gate.

### First-five packet rejection under the official chunk-mean RTF

With CFM1 established, synchronized hot timing measured the default first
Code2Wav packet at 75.30 ms: 12.77 ms encode, 34.38 ms CFM, 27.85 ms HiFT and
0.31 ms state publication. Client TTFP was 479.79 ms, leaving about 404.5 ms
before or around Stage 2, principally the wait for the first 25 Talker codes.

An official-protocol candidate lowered only the first transport boundary to
five new codes, the HiFT continuity minimum. It completed 32/32 requests,
retained 100% continuity and 158.60 seconds of audio at concurrency four, and
passed WER at `0.0102`. At concurrency one it reduced mean TTFP from 455.93 ms
to **271.47 ms** and P99 TTFP to **280.44 ms**, while E2E was unchanged.

It is nevertheless rejected for the ranked submission. The organizer defines
RTF as the arithmetic mean over every audio chunk and ranks RTF first. The
40-ms first packet makes its full request wait the numerator of a very small
first-chunk denominator. Mean flattened chunk RTF therefore regressed from
`0.30273` to `1.33619`, despite unchanged whole-audio throughput. This is a
real scoring consequence, not a throughput regression. A submission-oriented
first boundary must amortize pre-audio work over a longer packet and preferably
land on an efficient static Stage-2 width.

```text
/tmp/lunanexa-bench/a2-evaluator-source-default-cfm1-stage2-timing-zh1/
/tmp/lunanexa-bench/a2-evaluator-source-default-cfm1-stage2-timing-hot-zh1/
/tmp/lunanexa-bench/a2-evaluator-cfm1-first5-official-quality-zh32/
/tmp/lunanexa-bench/a2-evaluator-cfm1-first5-official-export-zh32/
/tmp/lunanexa-bench/a2-evaluator-cfm1-first5-official-perf-zh32-conc1/
/tmp/minicpmo-a2-evaluator-source-default-cfm1-stage2-timing.log
/tmp/minicpmo-a2-evaluator-cfm1-first5-official-server.log
```

### Ranked first-packet RTF optimization

The official rules rank the arithmetic mean of all chunk RTF values before
TTFP. The opposite scheduling direction was therefore screened: retain more
Talker codes in the first packet so the one-time pre-audio latency is divided
by a longer audio duration. First-47 also combines the three-code left context
into a natural width-50 Stage-2 input.

All performance rows use the same 32 Chinese prompts, two warmups,
concurrency one and server-default sampling. Lower is better:

| Initial codec frames | Mean chunk RTF | P99 chunk RTF | Mean TTFP | Mean E2E | Decision |
| ---: | ---: | ---: | ---: | ---: | --- |
| 25, CFM1 control | 0.30273 | 0.63560 | **455.93 ms** | 1392.11 ms | Lower-TTFP rollback |
| 47 | **0.27479** | **0.53501** | 646.86 ms | 1405.55 ms | Promoted after quality gate |
| 72 | 0.26916 | 0.51272 | 879.68 ms | **1399.39 ms** | Rejected: WER failure |

First-47 improves the primary mean-chunk RTF by **9.23%** and P99 RTF by
15.82% relative to first-25. First-72 improved RTF by another 2.05%, but its
official-protocol WER was `0.0172`, above the `0.0156` admission threshold, so
the numerically fastest ranked candidate is not eligible.

First-47 completed 32/32 quality requests, retained 100% continuity and
158.60 seconds of audio, scored WER `0.0087`, and scored `0.83727` with the
competition WavLM-base-plus proxy. The SIM change is about -0.76 percentage
points from the retained CFM6 proxy, inside the two-point allowance.

The evaluator-visible single-chip policy now defaults an absent
`VLLM_OMNI_MINICPMO45_INITIAL_CODEC_CHUNK_FRAMES` to `47`. An explicit launch
or Stage-1 value remains authoritative. Setting
`VLLM_OMNI_MINICPMO45_SINGLE_CHIP_RTF_FIRST47_DEFAULT=0` restores the native
25-frame first boundary without disabling the other exact defaults. This is a
ranked-profile tradeoff: it improves the primary RTF metric at the cost of
about 191 ms TTFP on this A2 host.

```text
/tmp/lunanexa-bench/a2-evaluator-cfm1-first47-official-perf-zh32-conc1/
/tmp/lunanexa-bench/a2-evaluator-cfm1-first47-official-quality-zh32/
/tmp/lunanexa-bench/a2-evaluator-cfm1-first47-official-export-zh32/
/tmp/lunanexa-bench/a2-evaluator-cfm1-first72-official-perf-zh32-conc1/
/tmp/lunanexa-bench/a2-evaluator-cfm1-first72-official-quality-zh32/
/tmp/lunanexa-bench/a2-evaluator-cfm1-first72-official-export-zh32/
/tmp/minicpmo-a2-evaluator-cfm1-first47-quality-server.log
/tmp/minicpmo-a2-evaluator-cfm1-first72-official-server.log
```

The evaluator-visible source default was then restarted without either an
explicit timestep or chunk-boundary override.  The same 32-row, two-warmup,
concurrency-one protocol measured mean flattened chunk RTF `0.27780`, P99
`0.52572`, mean TTFP `648.94 ms`, and mean E2E `1418.63 ms`.  This is within
1.10% of the explicit first-47 candidate and proves that the submitted source,
not only the exploratory launch environment, selects CFM1 and first-47.

```text
/tmp/lunanexa-bench/a2-evaluator-source-default-cfm1-first47-official-perf-zh32-conc1/
/tmp/minicpmo-a2-evaluator-source-default-cfm1-first47-server.log
```

### Steady-47 fixed-width rejection

The next candidate also changed every steady Talker publication from 25 to 47
new codec frames.  With the three-frame left context this makes the normal
Stage-2 input width 50, but it did not improve the ranked metric.  The exact
environment was confirmed in the Stage-1 process before measurement.

| Candidate | Chunks | Mean chunk RTF | P99 chunk RTF | Mean TTFP | Mean E2E |
| --- | ---: | ---: | ---: | ---: | ---: |
| Source default: first 47, steady 25 | 141 | **0.27780** | **0.52572** | **648.94 ms** | **1418.63 ms** |
| First 47, steady 47 | 98 | 0.33763 | 1.11675 | 651.50 ms | 1493.27 ms |

Steady-47 regressed mean RTF by **21.54%**.  Its 66 non-terminal chunks
already averaged `0.29689`, so the wider Stage-2 work is not cheaper per audio
second on this A2 stack.  In addition, the 32 terminal chunks averaged
`0.42168` and reached `1.19613`: a large publication boundary leaves a shorter
terminal remainder whose fixed launch cost is heavily amplified by the
organizer's unweighted mean-over-chunks metric.  Static width alone is
therefore insufficient.  Future chunk-shape work must eliminate or cheaply
complete short terminal packets and pass WER/SIM; this launch candidate is
rejected and is not a source default.

```text
/tmp/lunanexa-bench/a2-evaluator-cfm1-first47-steady47-official-perf-zh32-conc1/
/tmp/minicpmo-a2-evaluator-cfm1-first47-steady47-server.log
```

### First-60 boundary rejection

An intermediate first-packet boundary was screened after first-47 passed and
first-72 failed WER.  It retained the 25-frame steady boundary and changed
only the initial publication to 60 new codec frames.  The same source tree,
32 Chinese rows, two warmups and concurrency one produced 158.60 seconds of
audio with 100% continuity.

| Initial frames | Chunks | Mean chunk RTF | First-chunk mean RTF | Terminal-chunk mean RTF | Mean TTFP |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 47 explicit control | 141 | **0.27479** | 0.37608 | - | **646.86 ms** |
| 60 | 127 | 0.28092 | **0.33850** | 0.35453 | 758.24 ms |

First-60 amortized the first packet as intended, but its mean ranked RTF was
2.23% worse than the explicit first-47 control.  The longer first boundary
changes the modulo-25 terminal remainder and makes the final short packet more
expensive under the unweighted chunk mean.  Because it failed the performance
screen, WER/SIM were not run and first-47 remains the source default.

A raw official-shape diagnostic request made the mechanism concrete.  Its
first-60 packet contained 2.24 seconds of PCM, three normal steady packets
contained exactly 1.00 second each, and its terminal packet contained only
0.56 second.  The terminal arrival interval was 217.30 ms, so that final packet
alone scored RTF `0.38319`.  This motivates a quality-gated terminal-duration
floor rather than further initial-boundary integer search.

```text
/tmp/lunanexa-bench/a2-evaluator-cfm1-first60-steady25-official-perf-zh32-conc1/
/tmp/minicpmo-a2-evaluator-cfm1-first60-steady25-server.log
```

### Quality-gated terminal-packet floor

The first-60 diagnostic identified the short terminal remainder as a ranked
RTF outlier.  Stage 2 can now extend only a shorter final packet with digital
silence to an explicit minimum duration.  The synthesized prefix, request
state, and every non-terminal packet are unchanged.  A 1000-ms screen improved
the ranked RTF by 9.30% but failed WER at `0.0172`, so it was rejected.

The safer 600-ms candidate passed two identical 32-row WER screens and the
competition WavLM-base-plus proxy:

| Metric | Source default | Terminal 600 ms | Change / gate |
| --- | ---: | ---: | ---: |
| Mean flattened chunk RTF | 0.27780 | **0.26327** | **-5.23%** |
| P99 flattened chunk RTF | 0.52572 | **0.39779** | **-24.33%** |
| Mean TTFP | 648.94 ms | **645.91 ms** | -0.47% |
| Mean E2E | 1418.63 ms | **1383.28 ms** | -2.49% |
| Mean WER, repeat 1 / 2 | 0.0087 | **0.0153 / 0.0153** | pass, <= 0.0156 |
| WavLM-base-plus SIM | about 0.84485 | **0.8310** | -1.39 pp, pass |

Two narrower offline boundary screens confirmed that 600 ms is the largest
jointly admissible floor on this sample.  They appended only digital silence
to the exact terminal-600 exports, then reran the same per-row Paraformer and
WavLM-base-plus code paths:

| Floor | Mean WER | WavLM SIM | Result |
| --- | ---: | ---: | --- |
| 700 ms | 0.01587 | 0.82620 | reject: WER > 0.0156 |
| 800 ms | 0.01391 | 0.81891 | reject: SIM is -2.59 pp vs CFM6 |

The two metrics bind in opposite directions near the boundary, so neither
candidate is promoted without a larger official rerun demonstrating margin.

The one-chip policy therefore defaults
`VLLM_OMNI_MINICPMO45_TERMINAL_MIN_AUDIO_MS=600`.  An explicit launch or
stage value remains authoritative.  Setting
`VLLM_OMNI_MINICPMO45_SINGLE_CHIP_RTF_TERMINAL600_DEFAULT=0` disables only
this ranked-output policy.  The more aggressive 1000-ms candidate is not a
source default.

```text
/tmp/lunanexa-bench/a2-evaluator-cfm1-first47-terminal600-official-perf-zh32-conc1/
/tmp/lunanexa-bench/a2-evaluator-cfm1-first47-terminal600-official-quality-zh32/
/tmp/lunanexa-bench/a2-evaluator-cfm1-first47-terminal600-repeat-quality-sim-zh32/
/tmp/minicpmo-a2-evaluator-cfm1-first47-terminal600-server.log
```

### Isolated stable-PagedAttention rejection

The earlier stable-input PagedAttention screen combined that graph contract
with `weight_nz_mode=2`, so its catastrophic accuracy result did not identify
which switch changed Talker sampling.  A new candidate isolated stable PA on
the current first-47, CFM1 and terminal-600 source defaults without NZ weight
preformatting.

Its apparent performance was large but invalid: 32-row flattened chunk RTF
fell from `0.26327` to `0.20155` and TTFP from `645.91` to `570.99` ms, while
total generated audio increased from 159.68 to 200.20 seconds.  Wall-clock
duration improved only 1.16%.  The official quality screen confirmed that the
longer output was not a valid speedup: mean WER was `1.2856` across 31
evaluable rows, with one ASR failure, versus the `0.0156` admission limit.
WavLM-base-plus SIM was `0.7797`, but speaker similarity cannot compensate for
incorrect spoken content.

Stable PA is therefore rejected independently of NZ2 and is not enabled by
the evaluator-visible policy.  Future Talker graph-input work must preserve
the fused-attention numerical path and the codec/EOS distribution, not merely
reduce graph-parameter maintenance.

```text
/tmp/lunanexa-bench/a2-evaluator-cfm1-first47-terminal600-stable-pa-official-perf-zh32-conc1/
/tmp/lunanexa-bench/a2-evaluator-cfm1-first47-terminal600-stable-pa-official-quality-sim-zh32/
/tmp/minicpmo-a2-evaluator-cfm1-first47-terminal600-stable-pa-server.log
```

### FIA sequence-length bucket: initial rejection and corrected promotion

A second Talker experiment retained fused-infer-attention rather than changing
to PagedAttention.  It rounded each decode KV length to a 16-token bucket,
masked the not-yet-valid tail, and attempted to reuse the twenty captured FIA
tasks inside a bucket.  Standalone 910B4 probes were bit-identical in BF16
(maximum absolute error 0.0), including the real Talker shape (12 heads,
64-dimensional heads, a three-dimensional KV-cache view and a 4096-slot block
table).

The full graph did not preserve that result.  The candidate completed the
32-row performance run, but generated 202.32 seconds of audio in 186 chunks,
versus 159.68 seconds and 141 chunks for the safe control.  Its apparent
flattened chunk RTF of 0.19541 is therefore invalid; the Talker sampling/EOS
path had drifted.  This isolates the failure to captured CANN task reuse rather
than the eager attention arithmetic.  The bucket profile remains experimental
and is not part of evaluator-visible defaults.

A follow-up fixed the original mask-producer race by enqueueing the tail-mask
writes on the task-update stream before signaling each captured FIA group.  A
hot single request then matched the control's five chunks and 5.68-second
audio duration, while E2E fell from 1.504 to 1.266 seconds.  The complete
32-row run nevertheless reproduced 202.32 seconds and 186 chunks.  Its mean
chunk RTF was 0.19643 (25.4% below the safe 0.26327), but the official WER-only
screen rejected it decisively: mean WER was 1.7680, only seven rows were
evaluable and 25 ASR/WER rows failed.  Stream ordering was therefore a real
bug, but not the numerical cause of the invalid autoregressive trajectory.

FIA-v2 separately demonstrated bit-identical eager output while accepting a
fixed-address device sequence-length tensor.  On the installed A2 CANN 9.0 /
torch-npu stack, both its workspace helper and the FIA-v2 task itself extract a
local scalar from that tensor.  NPUGraph capture rejects the required stream
synchronization with error 107027 (`stream is captured`), so this path cannot
provide dynamic device lengths on the competition runtime and is also not
promoted.

The final correction identified two independent full-graph bugs that the
operator-only probes could not expose. First, the reuse key contained only the
rounded sequence length, so a new request could reuse tasks bound to the prior
request's block-table buffers. The key now includes every captured layer's
block-table address. Second, the initial tail-mask fill ran while the outer
NPUGraph was capturing; replay therefore overwrote every runtime mask with the
capture-time mask immediately before FIA. Capture now allocates the stable mask
without recording a write, and the task-update stream produces the runtime mask
before signaling the captured FIA groups.

Real 910B4 checks then proved both levels independently: rounded FIA with the
three-dimensional tail mask was bit-identical to exact-length sparse-mode-3
FIA, and a captured single-FIA task remained bit-identical while crossing
16-token buckets and reusing a bucket. The corrected complete service restored
the expected output geometry instead of the rejected 202.32-second trajectory.
The evaluator-visible source policy, using the generic A2 compatibility YAML
with no private bucket override, reproduced the win:

| Metric, lower is better | Safe control | Source-default bucket16 | Improvement |
| --- | ---: | ---: | ---: |
| Mean flattened chunk RTF | 0.263267 | **0.222028** | **15.66%** |
| P99 flattened chunk RTF | 0.397793 | **0.334418** | **15.93%** |
| Mean audio TTFP | 645.915 ms | **563.890 ms** | **12.70%** |
| Mean E2E | 1383.281 ms | **1166.872 ms** | **15.64%** |

The source-default run completed 32/32 requests, generated 158.20 seconds / 142
chunks, and retained 100% streaming continuity. Two independent performance
launches measured mean chunk RTF 0.22295 and 0.22203, so the result is not tied
to the isolation profile.

Three official-protocol 32-row WER screens scored `0.017020`, `0.013400`, and
`0.017020`; all completed 32/32 with no request, PCM, ASR, or scoring failure.
The stricter local `0.0156` early-screen boundary lies inside that run-to-run
range, while the maximum degradation from the matched safe `0.0153` control is
only 0.17 percentage points, inside the competition's two-point allowance.
The WavLM-base-plus proxy scored `0.8260` versus the safe control's `0.8310`, a
0.50-point loss, with 32/32 embeddings and no failures. The one-chip source
policy therefore defaults `fia_graph_seq_len_bucket_size=16` only for Stage 1.
An explicit Stage-1 additional-config value remains authoritative, and
`VLLM_OMNI_MINICPMO45_SINGLE_CHIP_FIA_BUCKET16_DEFAULT=0` restores exact-length
task updates.

torch-npu's outer `auto_dispatch_capture` FIA handler was also evaluated and
removed. The fused unified-attention custom-op boundary produced zero native
dispatch records even though twenty Talker layers were captured. Its apparent
0.188 chunk RTF came from replaying capture-time KV lengths and failed WER
catastrophically; a larger outer handler cannot see through that compiled-op
boundary on this stack.

A wider 32-token bucket was then screened twice to test whether remaining task
rebinding was still the dominant cost. Against bucket16 with the same
161.04-second output signature, bucket32 reduced mean chunk RTF from 0.22295 to
0.22222, only 0.33%, while P99 RTF regressed by 0.15%. With the alternate
158.20-second signature it reduced mean RTF from 0.22203 to 0.22003, but TTFP
regressed from 563.89 to 572.39 ms and P99 RTF from 0.33442 to 0.33973. The
sub-one-percent mean gain is below the promotion threshold and loses both tail
latency guards, so bucket16 remains the source default. Task rebinding is no
longer the primary Talker bottleneck; the next trace targets in-block layout
conversion and small operators.

### Bucket16 hot trace and scalar decode slot mapping

The refreshed Stage-1-only trace covered ten official-shape requests after the
capture-safe bucket16 promotion. Unlike the older layout trace, it contained no
material `TransData` or `Transpose` budget. `MatMulV2` accounted for 45.821% of
device time and FIA for 20.732%. The next discrete hotspot was instead the
generic `_compute_slot_mapping_kernel`: 1,288 launches, 243.294 ms total,
188.892 us average and 8.972% of all Stage-1 device time. The kernel clears the
maximum slot slab on every call even though batch-one Talker decode consumes
only slot zero; the runner separately pads the much smaller active graph view.

The retained vLLM-Ascend path is an opt-in scalar Triton kernel for exactly one
request, one live token and DCP world size one. It reads the stable position,
performs the same physical-to-logical hybrid-block mapping, writes slot zero,
and leaves prefill, batching, DCP and every disabled case on the canonical
kernel. `VLLM_ASCEND_SINGLE_TOKEN_SLOT_GRAPH=0` or
`VLLM_OMNI_MINICPMO45_SINGLE_CHIP_SLOT_FASTPATH_DEFAULT=0` restores the generic
path. This is distinct from the previously rejected nested-NPUGraph prototype:
the scalar kernel has no inner graph replay boundary.

Three 32-row runs completed without request or streaming failures. The two
runs with the same 158.20-second / 142-chunk output signature measured mean
chunk RTF 0.219961 and 0.218859, versus 0.222028 for the retained bucket16
control. Their mean is 0.219410, 1.18% lower. Server-side Stage-1 ITL over the
last 32 requests fell from 7.3313 ms to 7.1977 ms, or 1.82%. The best matched
run also reduced mean TTFP from 563.890 to 562.778 ms and mean E2E from
1166.872 to 1154.089 ms. P99 chunk RTF was 0.335229 versus 0.334418, a 0.24%
tail change; the alternate 160.92-second signature improved P99 from the
matching 0.350188 control to 0.343646.

The official quality gates preserved the output trajectory: WER was 0.0170
over 32/32 rows with zero request, PCM, ASR or scoring failures, and WavLM
speaker SIM was 0.82583 over 32/32 embeddings with zero failures. Those match
the retained bucket16 ranges (WER 0.0134--0.01702 and SIM 0.8260) and remain
well inside the competition's two-percentage-point allowance. The single-chip
source policy therefore enables the scalar path only for the Talker stage and
keeps both an explicit rollback and an isolation profile.

```text
/tmp/lunanexa-bench/a2-evaluator-cfm1-first47-terminal600-fia-bucket16-v2-official-perf-zh32-conc1/
/tmp/minicpmo-a2-evaluator-fia-bucket16-v2-server.log
/tmp/lunanexa-bench/a2-evaluator-cfm1-first47-terminal600-fia-bucket16-update-stream-official-perf-zh32-conc1/
/tmp/lunanexa-bench/a2-evaluator-cfm1-first47-terminal600-fia-bucket16-update-stream-wer-only-zh32/
/tmp/minicpmo-a2-evaluator-fia-bucket16-update-stream-server.log
/tmp/minicpmo-a2-evaluator-fia-v2-stable-v3-server.log
/tmp/fia-bucket16-capture-safe-perf-20260828/
/tmp/fia-bucket16-capture-safe-wer-20260828/
/tmp/source-default-fia-bucket16-perf-20260828/
/tmp/source-default-fia-bucket16-wer-repeat-20260828/
/tmp/source-default-fia-bucket16-sim-20260828/
/tmp/minicpmo-a2-source-default-fia-bucket16-server.log
/tmp/fia-bucket32-perf-20260828/
/tmp/fia-bucket32-perf-repeat-20260828/
/tmp/minicpmo-a2-fia-bucket32-server.log
/tmp/vllm-omni-profiles/minicpmo45/a2-fia-bucket16-stage1/
/tmp/slotfast-official-perf-20260828/
/tmp/slotfast-official-wer-20260828/
/tmp/slotfast-official-sim-20260828/
/tmp/minicpmo-a2-fia-bucket16-slotfast-server.log
```

### Persistent Talker decode metadata

The Stage-1 trace also showed that the batch-one decode runner retransmitted
metadata whose values do not change between codec tokens. In particular, it
uploaded the active KV block-table row on every step even though allocation
changes occur only at block boundaries, and it recopied the same
`query_start_loc`, request index, query offset, one-token schedule length,
discard mask, and all-ones accepted-token slab. These tiny copies create host
dispatch, H2D copy, fill, and event work around a decode graph whose tensor
addresses are already stable.

The retained vLLM-Ascend implementation has two independently reversible
parts. Block-table rows are dirty-tracked by every append, clear, move and swap
mutation and are uploaded only when an active row changed. The batch-one,
non-speculative, non-DCP decode path fingerprints its invariant metadata and
leaves matching slabs resident at their graph-visible NPU addresses. Prefill,
batching, speculative decode, DCP, GDN, shape transitions, and disabled cases
invalidate the fingerprints and use the canonical uploads. The source-default
single-chip policy scopes both switches to the Talker stage. Set
`VLLM_OMNI_MINICPMO45_SINGLE_CHIP_DECODE_METADATA_DEFAULT=0` to disable both,
or set `VLLM_ASCEND_DIRTY_BLOCK_TABLE_COMMIT=0` and
`VLLM_ASCEND_SINGLE_REQUEST_DECODE_METADATA_CACHE=0` independently.

Dirty block-table submission alone reduced matched Stage-1 ITL from 7.1977 to
7.0845 ms, but its 158.20-second whole-service run was neutral: mean chunk RTF
was 0.219477 versus the slot-fast two-run mean of 0.219410. It therefore was
not promoted by itself. Adding invariant metadata residency produced two
same-direction official-shape runs and improved both output signatures.

| Metric, lower is better | Slot-fast matched control | Persistent metadata | Improvement |
| --- | ---: | ---: | ---: |
| Mean chunk RTF, 158.20 s / 142 chunks | 0.219410 | **0.216392** | **1.38%** |
| P99 chunk RTF | 0.336446 | **0.333836** | **0.78%** |
| Mean audio TTFP | 562.852 ms | **553.566 ms** | **1.65%** |
| Mean E2E | 1155.853 ms | **1138.786 ms** | **1.48%** |
| Stage-1 ITL | 7.1977 ms | **6.9858 ms** | **2.94%** |

The independent 161.04-second / 145-chunk repeat also improved every guarded
metric versus dirty-only: mean chunk RTF 0.222004 to 0.216280, P99 0.346446 to
0.334312, TTFP 568.236 to 556.054 ms, E2E 1188.152 to 1158.536 ms, and
Stage-1 ITL 7.1777 to 7.0034 ms. All 96 measured requests across performance
and quality runs completed with continuous streaming and no request or PCM
failure.

The exact quality gate remained unchanged. Seed-TTS WER was 0.0170 over 32/32
utterances with zero ASR failures. Offline-cache WavLM SIM was 0.8259 over
32/32 embeddings with zero failures, matching the prior 0.82583 result. The
first combined WER/SIM invocation recorded 32 SIM infrastructure failures
because Hugging Face HEAD requests were reset; forcing `HF_HUB_OFFLINE=1`
loaded the already cached identical WavLM checkpoint and completed the gate.

```text
/tmp/dirtybt-official-perf-20260828/
/tmp/dirtybt-official-perf-repeat-20260828/
/tmp/metacache-official-perf-20260828/
/tmp/metacache-official-perf-repeat-20260828/
/tmp/metacache-official-quality-20260828/
/tmp/metacache-official-sim-offline-20260828/
/tmp/minicpmo-a2-fia-bucket16-slotfast-dirtybt-metacache-server.log
```

### Fused batch-one Talker metadata

The next retained step replaces the remaining batch-one decode scalar chain
with one graph-visible Triton/Ascend program. After the single dynamic
`num_computed_tokens` upload, the kernel writes position, sequence length, and
the first KV group's slot mapping together; every additional non-Mamba KV
group updates only its own slot slab. The implementation retains the exact
integer arithmetic and stable graph input addresses. It is gated by
`VLLM_ASCEND_SINGLE_REQUEST_DECODE_SCALAR_STAGING=1` under the same batch-one,
one-token, non-prefill, non-speculative, non-DCP, non-GDN and non-multiaxis-RoPE
conditions as the resident metadata path.

The first integration attempt exposed that MiniCPM-o uses
`MultiGroupBlockTable`; the missing wrapper method killed Stage 1 before any
scored request (0/32). That run was rejected, the wrapper was implemented to
cover every KV group, and both six focused unit tests and a cold real request
then passed before the official-shape run.

| Metric, lower is better | Persistent metadata control | Fused metadata | Improvement |
| --- | ---: | ---: | ---: |
| Mean chunk RTF, 161.04 s / 145 chunks | 0.216280 | **0.208516** | **3.59%** |
| P99 chunk RTF | 0.334312 | **0.331753** | **0.77%** |
| Mean audio TTFP | 556.054 ms | **544.965 ms** | **1.99%** |
| Mean E2E | 1158.536 ms | **1120.094 ms** | **3.32%** |
| Stage-1 ITL | 7.0034 ms | **6.7004 ms** | **4.33%** |

The quality run completed 32/32 requests with continuous streaming and zero
request, PCM, ASR or embedding failures. WER remained 0.0170 and WavLM SIM
remained 0.8259, exactly matching the previous accepted quality gate.

```text
/tmp/fusedscalar-multigroup-official-perf-20260828/
/tmp/fusedscalar-multigroup-official-quality-20260828/
/tmp/minicpmo-a2-fia-bucket16-slotfast-dirtybt-metacache-fusedscalar-multigroup-server.log
```

### Isolated Talker FRACTAL_NZ rejection

The current hot trace attributes 45.821% of Stage-1 device time to
`MatMulV2`, so Stage 1 was rerun with only its immutable BF16 linear weights
preformatted as FRACTAL_NZ (`weight_nz_mode=2`).  CFM1, first-47,
terminal-600, FIA bucket16, scalar slot mapping and fused resident metadata
were unchanged.  The 32-request result exactly matched the control's 161.04
seconds of audio and 145 chunks, but did not improve the primary metric:

| Metric, lower is better | ND control | FRACTAL_NZ | Change |
| --- | ---: | ---: | ---: |
| Mean chunk RTF | **0.208516** | 0.209138 | +0.30% |
| P99 chunk RTF | 0.331753 | **0.327603** | -1.25% |
| Mean audio TTFP | **544.965 ms** | 549.391 ms | +0.81% |
| Mean E2E | **1120.094 ms** | 1124.911 ms | +0.43% |
| Stage-1 ITL | **6.7004 ms** | 6.7046 ms | +0.06% |

The small P99 movement does not compensate for regressions in the ranked
mean, TTFP, E2E and total duration.  On this graph, GE's existing weight-format
selection is already at least as effective as whole-layer NZ preformatting.
The isolation profile was removed and the source default remains unchanged.

```text
/tmp/fusedscalar-nz-smoke-20260828/
/tmp/fusedscalar-nz-official-perf-20260828/
/tmp/minicpmo-a2-fia-bucket16-fusedscalar-nz-server.log
```

### Graph-wide FIA gate rejection

Bucket16 avoids task-parameter rebinding for fifteen out of every sixteen
Talker decode tokens, but its captured graph still contains one
`ExternalEvent` gate per FIA layer.  An opt-in lower-layer candidate replaced
those twenty gates with one gate before the first FIA layer.  At a bucket
transition the update stream first queued all twenty task updates and then
released the graph; inside a bucket it updated the tail mask and recorded only
the single gate.  This preserved exact attention math and the 161.04-second,
145-chunk output signature, but lost useful layer-by-layer update/compute
overlap:

| Metric, lower is better | Per-layer gates | Single graph gate | Change |
| --- | ---: | ---: | ---: |
| Mean chunk RTF | **0.208516** | 0.212717 | +2.01% |
| P99 chunk RTF | 0.331753 | **0.331458** | -0.09% |
| Mean audio TTFP | **544.965 ms** | 554.493 ms | +1.75% |
| Mean E2E | **1120.094 ms** | 1142.809 ms | +2.03% |
| Stage-1 ITL | **6.7004 ms** | 6.8430 ms | +2.04% |

The event calls are therefore not pure host overhead: their granularity lets
the replay stream start earlier while later FIA tasks are being rebound.  The
implementation and profile were fully removed; bucket16 retains per-layer
external events.

```text
/tmp/fusedscalar-singlegate-smoke-20260828/
/tmp/fusedscalar-singlegate-official-perf-20260828/
/tmp/minicpmo-a2-fia-bucket16-fusedscalar-singlegate-server.log
```

### ENPU update-before-replay rejection

The safe FIA configuration was also launched with vLLM-Ascend's internal
`ENPU_ENABLE=true` lifecycle path.  This preserves the exact attention
operator and sequence lengths, but synchronizes the current stream, updates
captured task parameters and only then enqueues graph replay.  That ordering is
covered by upstream graph-mode tests for other models, but it is incompatible
with MiniCPM-o's asynchronous three-stage execution on this stack.  The first
request remained in Stage-1 replay for more than two minutes with only 2%
AICore utilization and never produced a first audio packet.  The control's
cold request completes in about 68 seconds and subsequent requests in about
1.3 seconds.  The ENPU process was stopped and the safe post-replay external-
event ordering restored.

The next exact-math direction is therefore reducing the number of host task-
group begin/end/update operations without changing their event ordering, not
moving all updates ahead of replay.

```text
/tmp/minicpmo-a2-evaluator-source-default-enpu-server.log
```

### Talker token-only IPC coalescing

The retained sparse codec path already advances sampling and scheduler state
inside the Stage-1 engine process on every codec token, but it still serialized
and sent an otherwise empty `EngineCoreOutput` through ZMQ after every step.
The promoted scheduler path accumulates only those token ids locally and
attaches them to the next real codec payload, stop, or transfer boundary.  It
does not batch model execution, speculate tokens, reorder sampling, or change
the codec/CFM/HiFT tensors.  Requests that ask for token log probabilities and
all non-audio/non-Stage-1 configurations remain on the canonical path.

Two consecutive 32-row, concurrency-one runs completed 32/32 requests with
100% streaming continuity.  Talker output length is stochastic on this stack:
the runs landed in the already-observed 160.92-second/144-chunk and
158.20-second/142-chunk clusters respectively.  The former cluster also
occurred in the retained slot-fast control and the latter in the dirty-table
and metadata-cache controls, so the length change is not attributed to IPC
coalescing.

| Metric, lower is better | Fused-metadata control | IPC coalesce run 1 | IPC coalesce run 2 |
| --- | ---: | ---: | ---: |
| Mean chunk RTF | 0.208516 | **0.206408** | 0.207337 |
| P99 chunk RTF | 0.331753 | 0.327476 | **0.323383** |
| Mean audio TTFP | 544.965 ms | 543.215 ms | **542.227 ms** |
| Mean E2E | 1120.094 ms | 1112.290 ms | **1095.828 ms** |
| Benchmark duration | 35.857 s | 35.609 s | **35.083 s** |
| Stage-1 ITL | 6.7004 ms | - | **6.6502 ms** |

Against the strongest prior run, the first candidate lowers mean chunk RTF by
1.01%, P99 by 1.29%, audio TTFP by 0.32%, E2E by 0.70%, and total duration by
0.69%.  The last 32 server-side Stage-1 samples lower ITL by 0.75%.  Because
the optimization changes transport frequency only, the previously accepted
WER/SIM model-output gate remains applicable; the two performance runs also
validate request completion and PCM streaming.

```text
/tmp/fusedscalar-ipccoalesce-smoke-20260828/
/tmp/fusedscalar-ipccoalesce-official-perf-20260828/
/tmp/fusedscalar-ipccoalesce-official-perf-repeat-20260828/
/tmp/minicpmo-a2-fusedscalar-ipccoalesce-server.log
```

### Two-layer FIA event grouping rejection

The earlier single-gate experiment showed that removing all twenty per-layer
ExternalEvents loses valuable task-update/attention overlap.  A finer
experiment kept all twenty FIA task updates but shared one gate per adjacent
layer pair, reducing event waits/signals from twenty to ten while releasing
compute after every two updates.  Unit tests and a cold smoke passed, but the
same-signature 160.92-second/144-chunk official run regressed against the
IPC-only result:

| Metric, lower is better | Per-layer gates | Two-layer gates | Change |
| --- | ---: | ---: | ---: |
| Mean chunk RTF | **0.206408** | 0.208611 | +1.07% |
| P99 chunk RTF | **0.327476** | 0.330033 | +0.78% |
| Mean audio TTFP | **543.215 ms** | 551.661 ms | +1.56% |
| Mean E2E | **1112.290 ms** | 1125.262 ms | +1.17% |
| Benchmark duration | **35.609 s** | 36.024 s | +1.16% |
| Stage-1 ITL | **6.6502 ms** | 6.7619 ms | +1.68% |

Even one extra layer of release delay costs more than ten event operations on
this graph.  The implementation and profile were removed, and the retained
path continues to signal one event per FIA layer.

```text
/tmp/fusedscalar-ipccoalesce-eventgroup2-smoke-20260828/
/tmp/fusedscalar-ipccoalesce-eventgroup2-official-perf-20260828/
/tmp/minicpmo-a2-fusedscalar-ipccoalesce-eventgroup2-server.log
```

### Talker runtime W8A16 rejection

The Stage-1 trace attributes 45.821% of device time to `MatMulV2`, so a
runtime-only W8A16 experiment quantized all 80 Talker Llama linear weights
with symmetric per-output-channel scales.  Activations, RMSNorm, attention
probabilities, codec projection, Stage 0, CFM and HiFT stayed BF16/FP32.  The
post-load conversion reduced Stage-1 model weights from about 0.93 GB to
0.4674 GB.

The first launch exposed a stale-cache hazard: runtime quantization was not
represented in vLLM's AOT/NPUGraph cache key, so a cached BF16 graph expected
the pre-quantization weight layout.  A cache-disabled diagnostic forced fresh
compilation.  That compilation then exposed a torch_npu 2.10 packaging defect:
`npu_fx_compiler` disables `enable_view_optimize` for
`npu_weight_quant_batchmatmul`, but the matching NPUGraph experimental config
omits the option.  vLLM-Ascend now repairs that missing option only on affected
packages.  With the compatibility repair, W8A16 compiled, completed cold
warmup and served 32/32 requests with 100% streaming continuity.

It was nevertheless slower than the retained full-precision path:

| Metric, lower is better | IPC-coalesced BF16 | Talker W8A16 | Change |
| --- | ---: | ---: | ---: |
| Overall audio RTF | **0.221284** | 0.238195 | +7.64% |
| Mean chunk RTF | **0.206408** | 0.223413 | +8.24% |
| P99 chunk RTF | **0.329015** | 0.345386 | +4.98% |
| Mean audio TTFP | **543.215 ms** | 575.535 ms | +5.95% |
| Mean E2E | **1112.290 ms** | 1203.883 ms | +8.23% |
| Benchmark duration | **35.609 s** | 38.540 s | +8.23% |

The candidate generated 161.80 seconds of audio and 145 chunks versus the
control's already-observed 160.92-second/144-chunk stochastic cluster; the
normalized RTF and per-chunk regressions independently reject it.  At
concurrency one, A2's small decode GEMMs do not amortize weight dequantization
and layout overhead.  The runtime quantization code and deploy profile were
fully removed, and no quality run was spent on a performance-losing candidate.
The generic NPUGraph compatibility repair remains because it fixes fresh
compilation for any legitimate weight-quantized graph.

```text
/tmp/talker-w8a16-smoke-20260828/
/tmp/talker-w8a16-official-perf-20260828/
/tmp/minicpmo-a2-talker-w8a16-viewcompat-20260828-server.log
```

### Talker static-kernel rejection

The fixed token-one Stage-1 graph was also compiled with NPUGraph_ex static
ACLNN kernels while retaining BF16 weights, FIA bucket16, scalar-slot mapping,
fused resident metadata, and token-only IPC coalescing.  The launch log
confirmed a fresh graph cache key with `enable_static_kernel=true`, built and
installed the generated kernel package, and completed a real cold request.
The package was removed automatically when the candidate process exited.

Static compilation did not preserve the Talker output trajectory on this A2
stack.  The same 32 prompts produced 202.32 seconds / 186 chunks instead of the
control's 160.92 seconds / 144 chunks.  Consequently a lower normalized
per-chunk number was a false win: the service performed substantially more
autoregressive work and regressed request latency and total duration.

| Metric, lower is better | NPUGraph_ex control | Static kernel | Change |
| --- | ---: | ---: | ---: |
| Mean chunk RTF | 0.206408 | **0.185415** | -10.17% |
| P99 chunk RTF | 0.329015 | **0.328123** | -0.27% |
| Mean audio TTFP | **543.215 ms** | 549.840 ms | +1.22% |
| Mean E2E | **1112.290 ms** | 1279.052 ms | +14.99% |
| Benchmark duration | **35.609 s** | 40.945 s | +14.98% |
| Generated audio | **160.92 s / 144 chunks** | 202.32 s / 186 chunks | +25.73% audio |

The output-shape mismatch fails the exact-math gate before any official
WER/SIM run.  The deploy profile was removed and the retained Stage-1 graph
continues to use NPUGraph_ex without static-kernel materialization.

```text
/tmp/talker-static-kernel-smoke-20260828/
/tmp/talker-static-kernel-official-perf-20260828/
/tmp/minicpmo-a2-talker-static-kernel-20260828-server.log
```

### Talker FP16 rejection

The next isolation run changed only the Stage-1 model and decode graph from
BF16 to FP16.  Stage 0 remained BF16 and Stage 2 retained its established
mixed-precision path.  Startup logs confirmed `dtype=torch.float16` for the
Talker, a fresh graph capture completed, and a second fully hot four-request
smoke excluded the first run's lazy-shape compilation outlier.

The complete 32-request run passed serving continuity but regressed the ranked
mean and every aggregate/first-packet latency guard.  Its 159.96 seconds / 145
chunks are close to the control's existing stochastic output clusters, so the
normalized RTFs—not raw output length—reject the candidate.

| Metric, lower is better | BF16 control | FP16 Talker | Change |
| --- | ---: | ---: | ---: |
| Overall audio RTF | **0.221284** | 0.225080 | +1.72% |
| Mean chunk RTF | **0.206408** | 0.212361 | +2.88% |
| P99 chunk RTF | 0.329015 | **0.327209** | -0.55% |
| Mean audio TTFP | **543.215 ms** | 545.474 ms | +0.42% |
| Mean E2E | **1112.290 ms** | 1124.623 ms | +1.11% |
| Benchmark duration | **35.609 s** | 36.004 s | +1.11% |

The small P99 movement does not offset regressions in the primary mean, TTFP,
E2E, and total duration.  The experimental profile was removed.  Lower
precision for the Talker's dominant small GEMMs now requires a calibrated
static W8A8 graph that uses A2's INT8 Cube path; merely exchanging BF16 for
FP16 does not provide that acceleration.

```text
/tmp/talker-fp16-smoke-20260828/
/tmp/talker-fp16-smoke-repeat-20260828/
/tmp/talker-fp16-official-perf-20260828/
/tmp/minicpmo-a2-talker-fp16-20260828-server.log
```

### Talker dynamic-W8A8 projection rejection

The next experiment attacked the latest Stage-1 trace's dominant
`MatMulV2` budget with graph-visible A2 INT8 Cube operations.  Symmetric
per-output-channel INT8 weights were prepared once after checkpoint loading;
activations were quantized per token inside the graph.  RMSNorm, attention,
row projections, codec sampling, Stage 0 and Stage 2 retained their proven
BF16/FP32 paths.  The Ascend compiler confirmed that `norm_quant` fusion was
enabled and a fresh cache-disabled graph was captured.

Two target sets were measured.  Quantizing all 20 fused QKV and 20 fused
gate/up projections reduced Talker model memory to 0.5221 GB, with 124.39 MiB
of persistent INT8 parameters.  A second candidate retained BF16 QKV and only
quantized the larger `[6144,768]` gate/up projection in each layer: 20
projections, 90.47 MiB INT8, and 0.5549 GB total Talker model memory.  Both
served 4/4 requests with 100% streaming continuity after a 68-second first
graph compile.

Fully hot four-request smoke results rejected both candidates:

| Metric, lower is better | BF16 smoke control | QKV + gate/up W8A8 | Gate/up-only W8A8 |
| --- | ---: | ---: | ---: |
| Overall audio RTF | **0.237039** | 0.247373 | 0.238995 |
| Mean chunk RTF | 0.224442 | 0.227800 | **0.223403** |
| Mean audio TTFP | **551.967 ms** | 590.037 ms | 573.866 ms |
| Mean E2E | **1084.984 ms** | 1152.197 ms | 1151.511 ms |
| Benchmark duration | **4.343 s** | 4.611 s | 4.608 s |
| Generated audio / chunks | 18.32 s / 17 | 18.64 s / 17 | 19.28 s / 18 |

Gate/up-only recovered most of the dual-target loss and improved mean chunk
RTF by 0.46%, but regressed overall RTF by 0.83%, TTFP by 3.97% and E2E by
6.13%.  Dynamic per-token activation scale calculation and the additional
graph work still exceed the Cube saving for concurrency-one autoregressive
decode.  No 32-request or quality run was spent on either losing candidate.
The runtime conversion and deploy profile were removed.

The remaining INT8 path is offline calibrated static W8A8: precompute
activation scales, retain higher precision for RMSNorm/attention/sampling, and
compile the quantized projection graph without a per-layer dynamic-scale
kernel.  That requires a MiniCPM Talker calibration/export adapter rather than
another runtime-only dtype experiment.

```text
/tmp/talker-selective-w8a8-smoke-20260828/
/tmp/talker-selective-w8a8-smoke-repeat-20260828/
/tmp/talker-gateup-w8a8-smoke-20260828/
/tmp/talker-gateup-w8a8-smoke-repeat-20260828/
/tmp/minicpmo-a2-talker-selective-w8a8-20260828-server.log
/tmp/minicpmo-a2-talker-gateup-w8a8-fixedpath-20260828-server.log
```

### Talker calibrated static-W8A8 rejection

An eager-only calibration run collected nonzero gate/up input maxima for all
20 Talker layers over 32 Seed-TTS requests.  The observed maxima increased
plausibly with residual depth, from 3.359375 in layer 0 to 30.25 in layer 19.
Graph-mode calibration was explicitly discarded because Python forward hooks
do not execute during NPUGraph replay.  The valid eager artifact then drove a
per-tensor static activation scale (5% headroom) and per-output-channel INT8
weights for each `[6144,768]` gate/up projection.

The first candidate used graph-visible fixed quantization and the second also
enabled vLLM-Ascend's RMSNorm/activation-quant fusion passes.  Startup
confirmed 20 converted projections, 91.03 MiB of persistent parameters, a
fresh compile-cache-disabled graph, and successful NPUGraph capture.  Both
served 4/4 requests with 100% streaming continuity.  Fusion did not recover
the small batch-one GEMM overhead:

| Metric, lower is better | BF16 smoke control | Static W8A8 | Static W8A8 + norm-quant |
| --- | ---: | ---: | ---: |
| Overall audio RTF | **0.237039** | 0.277713 | 0.280000 |
| Mean audio TTFP | **551.967 ms** | 589.660 ms | 594.580 ms |
| Mean E2E | **1084.984 ms** | 967.960 ms | 974.330 ms |
| Benchmark duration | 4.343 s | 3.870 s | 3.900 s |
| Generated audio | **18.32 s** | 13.96 s | 13.96 s |

The normalized RTF regressed by 18.1% and TTFP by 7.7% in the fully hot fused
run.  More importantly, both static candidates changed the output trajectory
and generated 23.8% less audio than the exact BF16 smoke.  They therefore fail
both the performance and exact-math promotion gates; no 32-request quality run
was spent.  The inference conversion and fusion profile were removed.  The
eager calibration/export adapter remains as development tooling for future
offline-quantized checkpoints, but it is inactive in serving.

```text
/tmp/minicpmo45-talker-gateup-calibration-20260828-v2.json
/tmp/talker-static-w8a8-gateup-smoke-repeat-20260828/
/tmp/talker-static-w8a8-normquant-smoke-20260828/
/tmp/talker-static-w8a8-normquant-smoke-repeat-20260828/
/tmp/minicpmo-a2-talker-static-w8a8-calibration-eager-20260828-server.log
/tmp/minicpmo-a2-talker-static-w8a8-normquant-20260828-server.log
```

### Talker direct decode-embedding rejection

An opt-in runner path replaced each single-token Talker embedding temporary
plus graph-input copy with `torch.index_select(..., out=stable_input_slice)`.
The installed torch-npu 2.10 stack supported the BF16 `out=` operation and the
candidate completed two hot four-request runs and the 32-request performance
gate without fallback or streaming failure.

The tiny smoke runs showed an unstable 2.6--6.0 ms TTFP reduction, but the
official-shape run reversed it and regressed every normalized ranked metric:

| Metric, lower is better | Retained BF16 | Direct embedding | Change |
| --- | ---: | ---: | ---: |
| Overall audio RTF | **0.221284** | 0.234876 | +6.14% |
| Mean chunk RTF | **0.206408** | 0.218432 | +5.83% |
| P99 chunk RTF | **0.329015** | 0.340267 | +3.42% |
| Mean audio TTFP | **543.215 ms** | 559.205 ms | +2.94% |
| Mean TTFT | **75.057 ms** | 80.533 ms | +7.30% |

The candidate generated a different audio-length cluster (140.88 seconds / 125
chunks versus 160.92 seconds / 144 chunks), so the shorter raw duration and E2E
are not wins. RTF and per-chunk normalization independently reject the path.
The runtime code and deploy profile were removed.

The first real warmup also exposed a separate wrapper regression: insertion of
the experimental method had split the existing Stage-0 `preprocess` body and
left the ordinary LLM path returning an unbound `embeds`. The original control
flow was restored and a plain-LLM preprocessing regression test was retained;
that correctness fix is independent of the rejected optimization.

```text
/tmp/talker-direct-decode-embed-fixed-smoke-v2-20260828/
/tmp/talker-direct-decode-embed-fixed-smoke-v3-20260828/
/tmp/talker-direct-decode-embed-official-perf-20260828/
/tmp/minicpmo-a2-direct-decode-embed-fixed-20260828-server.log
```

### Talker replay-fence rejection and clean-cache A2 compatibility

A fresh isolated vLLM-Ascend tree first exposed a deployment correctness
problem hidden by the long-lived control's compiled graph cache.  CANN 9.0 on
this 910B4 registers the custom-op wrapper, but its operator package does not
contain `aclnnAddRmsNormBias`; a fresh Stage-0 graph therefore failed during
warmup.  A narrow `VLLM_ASCEND_ENABLE_ADD_RMS_NORM_BIAS=0` control now
decomposes only that operation to `npu_add_rms_norm`.  It does not disable the
MiniCPM QKV, convolution, AdaLN, or other custom kernels.

The subsequent performance experiment attempted to replace the broad
per-token model-stream synchronize before Talker FULL-graph replay.  A first
device-only update-stream fence deadlocked because `graph_task_update_begin`
mutates captured tasks on the host immediately; queuing a device wait cannot
delay that host mutation.  The corrected candidate retained a host barrier but
waited on an event recorded immediately after the prior replay, excluding
later model-stream work.  It served 32/32 requests with 100% streaming
continuity.

The first comparison against the historical cached control was confounded:
the historical run generated 160.92 seconds / 144 chunks, while the new
isolated runs generated roughly 137 seconds / 119 chunks.  That output-length
change must not be attributed to the A2 compatibility decomposition without a
matched control; it can also arise from the sampled Talker trajectory and
benchmark protocol.  The candidate was therefore compared with two no-fence
runs from the same isolated source, graph cache, request settings and
compatibility path.  That strict A/B rejected it as benchmark noise rather than
a repeatable mean-speed improvement:

| Metric, lower is better | No-fence control 1 | No-fence control 2 | Precise replay event |
| --- | ---: | ---: | ---: | ---: |
| Overall audio RTF | 0.234059 | **0.231018** | 0.232503 |
| Mean chunk RTF | 0.220769 | **0.215947** | 0.219432 |
| P99 chunk RTF | 0.329353 | 0.329028 | **0.322421** |
| Mean audio TTFP | 544.587 ms | 542.012 ms | **539.777 ms** |
| Mean E2E | 1003.611 ms | **990.620 ms** | 992.775 ms |
| Mean TTFT | 76.782 ms | 76.787 ms | **76.209 ms** |

The candidate's primary RTF values fall inside the control's 1.3--2.2%
run-to-run spread.  TTFP improved by only 0.4--0.9% and P99 chunk RTF by about
2%, while mean chunk RTF was 1.6% slower than the faster control repeat.  That
is not enough to justify another execution path.  Both replay-fence variants
and their deploy profile were removed.

The experiment confirms that graph-task rebinding is an opaque host mutation
boundary on this torch-npu release; eliminating the barrier requires fixed
graph-visible attention inputs or an upstream task-update API with explicit
asynchronous lifetime semantics, not a different stream wait.

```text
/tmp/talker-async-replay-fence-smoke-v3-20260829.log
/tmp/talker-precise-replay-fence-smoke-v4-20260829/
/tmp/talker-precise-replay-fence-official-v4-20260829/
/tmp/cleancompile-control-official-v5-20260829/
/tmp/cleancompile-control-repeat-official-v5-20260829/
/tmp/minicpmo-a2-talker-precise-replay-fence-v4-20260829-server.log
/tmp/minicpmo-a2-cleancompile-control-v5-20260829-server.log
```

### FIA-v2 device-length rejection and graph-cache isolation

A follow-up attempted to remove FIA task rebinding entirely by binding Stage
1 attention to the model runner's persistent NPU sequence-length tensor, then
skipping the conservative pre-replay host synchronize.  This is not supported
by the installed torch-npu 2.10 operator contract.  Both
`actual_seq_qlen` and `actual_seq_kvlen` are `SymInt[]`, not Tensor inputs.
Passing the NPU tensor makes the wrapper extract a scalar through
`LocalScalarDenseNpu`; capture then fails because that extraction calls
`aclrtSynchronizeStream` on the captured stream (`107027`, followed by
capture-end `107033`).  The experimental async-replay code and profile were
removed.

The failed launch exposed a separate persistent-cache correctness issue.  The
A2 capability switch for the unavailable `aclnnAddRmsNormBias` changes the FX
graph, but neither vLLM's global AOT key nor vLLM-Ascend's backend key included
that switch.  A graph traced with the optional A3-capable path could therefore
be reused on A2.  The normalized capability is now included in both cache
layers.  Focused remote tests passed (8 passed), and clean Stage-0 and Stage-1
compilation produced distinct A2 cache keys and completed both FULL decode
graph captures.

The fully warm 32-request control below uses the same evaluator-compatible
source policy, Stage-1 bucket16 FIA, conservative replay fence, local model
overlay, concurrency one and deterministic benchmark sampling as the strict
controls above.  It completed 32/32 with 100% streaming continuity:

| Metric, lower is better | Clean-cache control 1 | Clean-cache control 2 | Capability-key control |
| --- | ---: | ---: | ---: |
| Overall audio RTF | 0.234059 | **0.231018** | 0.233979 |
| Mean chunk RTF | 0.220769 | **0.215947** | 0.220094 |
| P99 chunk RTF | 0.329353 | 0.329028 | **0.317693** |
| Mean audio TTFP | 544.587 ms | 542.012 ms | **537.419 ms** |
| Mean E2E | 1003.611 ms | **990.620 ms** | 1003.283 ms |
| Mean TTFT | 76.782 ms | 76.787 ms | **75.469 ms** |

The primary means remain inside ordinary run variance, as expected for a cache
correctness fix.  The first request after process startup spent 68.40 seconds
in lazy Code2Wav/operator initialization, so that cold run is recorded but not
mixed into the fully warm serving comparison.  Local checkpoint staging also
reduced Stage-0 weight loading from roughly 350--390 seconds on the remote
filesystem to 3--12 seconds; this is deployment startup improvement, not a
ranked steady-state RTF claim.

```text
/tmp/cachefixed-control-official-v11-20260829/
/tmp/cachefixed-control-repeat-official-v11-20260829/
/tmp/minicpmo-a2-stable-fia-v2-async-v8-local-20260829-server.log
/tmp/minicpmo-a2-cachefixed-control-v11-local-20260829-server.log
```

### Bucket-stable FIA replay preflight

The rejected replay-fence experiments above still synchronized before every
FULL graph replay because they could not prove whether `graph_task_update`
would rebind FIA metadata.  The revised opt-in path performs a conservative
preflight using the rounded sequence-length bucket plus every block-table
address.  It skips the host fence only when all captured FIA tasks can be
reused.  A ping-pong completion event protects the shared tail mask; bucket,
request, speculative and unrecognized transitions keep the existing host
synchronize.

Focused vLLM-Ascend tests pass (64 passed).  In the Stage-1 trace this reduced
`aclrtSynchronizeStreamWithTimeout` calls from 386 to 29, but the removed calls
accounted for only about 1.5 ms per request.  Consequently this remains an
experimental, explicitly enabled optimization rather than a claimed major RTF
gain.  The resident repetition-penalty fix below dominates the measured
combined result.

```text
vllm_omni/deploy/minicpmo_4_5_1npu_a2_evaluator_fia_bucket16_async_replay_experimental.yaml
vllm_omni/deploy/minicpmo_4_5_1npu_a2_evaluator_fia_bucket16_async_replay_talker_profile.yaml
```

### Resident Talker repetition-penalty scalar

A Stage-1 stack trace found that 140 of 153 host stream synchronizations in a
representative Talker request came from the codec repetition-penalty helper.
Every generated codec token called `torch.as_tensor(1.05, device="npu")`,
which synchronously materialized the same Python scalar on the NPU.  Those 140
calls accounted for 145.628 ms of host-side synchronization time.

The Talker already owns `_fused_codec_penalty` as a registered tensor buffer.
The hot path now reuses that resident buffer and retains the scalar fallback
for standalone callers.  The penalty value, frequency accumulation, sampling
distribution, CFM step count and model weights are unchanged.  A focused test
also fails if the resident-tensor path calls `torch.as_tensor`; the four
relevant remote tests pass.

Two fully warm 32-request Seed-TTS runs completed 32/32 with 100% streaming
continuity.  The comparison uses the three-run median of the immediately
preceding capability-key controls as the baseline.  For the two new runs, the
reported candidate is their median (the average of the two middle values):

| Metric, lower is better | Control median | Resident scalar median | Improvement |
| --- | ---: | ---: | ---: |
| Overall audio RTF | 0.233979 | 0.191745 | 18.05% |
| Mean chunk RTF | 0.220094 | 0.179135 | 18.61% |
| P99 chunk RTF | 0.329028 | 0.282424 | 14.16% |
| Mean audio TTFP | 542.012 ms | 474.117 ms | 12.53% |
| Mean E2E | 1003.283 ms | 963.326 ms | 3.98% |
| Mean TTFT | 76.782 ms | 75.673 ms | 1.44% |

The individual candidate runs produced mean chunk RTF 0.180403 / 0.177868
and mean TTFP 475.414 / 472.820 ms, so the gain reproduced across both full
runs.  Output durations were 162.24 and 159.44 seconds; per-request RTF and
chunk RTF remain normalized for output duration.  This optimization is exact
at the repetition-penalty expression level, but the official three-benchmark
accuracy gate still remains required before submission.

```text
/tmp/lunanexa-bench/resident-penalty-v20-official32/
/tmp/lunanexa-bench/resident-penalty-v20-repeat-official32/
/tmp/vllm-omni-profiles/minicpmo45/a2-fia-bucket16-async-stage1/stage1_rank0/9af131f15bd4_1670738_20260828213004818_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json
```

A post-fix warm trace confirms the intended mechanism.  Only 13 plain stream
synchronizations and 9 timed stream synchronizations remain; their combined
runtime is 0.480 ms.  The old 140 per-token scalar-materialization syncs are
absent.  The remaining Stage-1 budget is dominated by autoregressive launch
gaps plus the eager head/filter/multinomial chain rather than that scalar.

### Resident scalar plus inverse-CDF sampler graph

The existing fixed-shape inverse-CDF sampler was then layered on the new
resident-scalar and bucket-async baseline.  It captures the codec head,
frequency penalty, bounded top-k/top-p, draw and rolling frequency update in a
small graph.  One focused deploy-config test passes, and three fully warm
32-request runs completed 32/32 with 100% streaming continuity.  The final
run enabled the complete official metric set:

| Metric, lower is better | Resident scalar median | Sampler graph | Improvement |
| --- | ---: | ---: | ---: |
| Overall audio RTF | 0.191745 | 0.159988 | 16.56% |
| Mean audio RTF | 0.194141 | 0.164302 | 15.37% |
| Mean chunk RTF | 0.179135 | 0.146645 | 18.14% |
| P99 chunk RTF | 0.282424 | 0.263500 | 6.70% |
| Mean audio TTFP | 474.117 ms | 436.852 ms | 7.86% |
| Mean E2E | 963.326 ms | 938.292 ms | 2.60% |
| Mean TTFT | 75.673 ms | 74.779 ms | 1.18% |

This is a real normalized speed gain, but it is not yet the submission
default.  Inverse-CDF preserves the categorical distribution but changes the
seed-to-code mapping relative to `torch.multinomial`; the measured run
generated 187.76 seconds of audio versus 159.44--162.24 seconds for the eager
sampler runs.  The competition's Seed-TTS WER/SIM, Daily-Omni and Video-MME
accuracy gate must therefore approve it before promotion.  The profile name
keeps the risk explicit.

```text
vllm_omni/deploy/minicpmo_4_5_1npu_a2_evaluator_fia_bucket16_async_replay_sampler_graph_experimental.yaml
/tmp/lunanexa-bench/resident-sampler-v22-official32/
/tmp/lunanexa-bench/resident-sampler-v22-repeat-official32/
/tmp/lunanexa-bench/resident-sampler-v22-metrics-official32/
/tmp/vllm-omni-profiles/minicpmo45/a2-fia-bucket16-async-stage1/stage1_rank0/9af131f15bd4_1681277_20260828220213179_ascend_pt/ASCEND_PROFILER_OUTPUT/trace_view.json
```

```text
/tmp/lunanexa-bench/a2-evaluator-exact-defaults-zh10/
/tmp/lunanexa-bench/a2-evaluator-cfm2-zh10/
/tmp/lunanexa-bench/a2-evaluator-cfm2-zh32/
/tmp/lunanexa-bench/a2-evaluator-cfm2-wer-fixed-zh32/
/tmp/lunanexa-bench/a2-evaluator-cfm5-zh32/
/tmp/lunanexa-bench/a2-evaluator-cfm5-wer-zh32/
/tmp/lunanexa-bench/a2-evaluator-cfm6-rollback-zh10/
/tmp/lunanexa-bench/a2-evaluator-cfm6-rollback-official-quality-zh32/
/tmp/lunanexa-bench/a2-evaluator-cfm6-exact-official-quality-zh32/
/tmp/lunanexa-bench/a2-evaluator-cfm6-safe-exact-official-quality-zh32/
/tmp/lunanexa-bench/a2-evaluator-cfm6-safe-exact-official-perf-zh32-conc1/
/tmp/lunanexa-bench/a2-evaluator-cfm2-safe-exact-official-quality-zh32/
/tmp/lunanexa-bench/a2-evaluator-cfm2-safe-exact-official-export-zh32/
/tmp/lunanexa-bench/a2-evaluator-cfm3-safe-exact-official-quality-zh32/
/tmp/lunanexa-bench/a2-evaluator-cfm3-safe-exact-official-export-zh32/
/tmp/lunanexa-bench/a2-evaluator-source-default-cfm2-official-perf-zh32-conc1/
/tmp/lunanexa-bench/a2-evaluator-cfm1-safe-exact-official-quality-zh32/
/tmp/lunanexa-bench/a2-evaluator-cfm1-safe-exact-official-export-zh32/
/tmp/lunanexa-bench/a2-evaluator-cfm1-safe-exact-official-perf-zh32-conc1/
/tmp/minicpmo-a2-evaluator-exact-defaults.log
/tmp/minicpmo-a2-evaluator-cfm2.log
/tmp/minicpmo-a2-evaluator-cfm5.log
/tmp/minicpmo-a2-evaluator-cfm6-rollback.log
```

## A2 inverse-CDF accuracy rejection

The resident-scalar eager sampler and the inverse-CDF sampler graph were run
against the same first 32 Chinese Seed-TTS rows on the same A2 service image.
Both runs used seed zero, disabled shuffle and oversampling, temperature zero,
concurrency four, the local Paraformer checkpoint, and the same WavLM-base-plus
proxy scorer.  All 32 serving requests completed and streaming continuity was
100% in both runs.

| Metric | Resident eager control | Inverse-CDF graph | Gate decision |
| --- | ---: | ---: | --- |
| Serving duration | 21.91 s | 19.70 s | inverse-CDF is 10.1% faster |
| Audio throughput | 6.32 audio-s/s | 7.25 audio-s/s | inverse-CDF is 14.7% higher |
| WER evaluated | 32 / 32 | 29 / 32 | inverse-CDF has 3 empty-ASR rows |
| Mean WER | 4.8315% | 65.1398% | fail; organizer limit is 1.56% |
| Median WER | 0.0000% | 68.7500% | fail |
| Mean proxy SIM | 0.831314 | 0.781522 | both exceed 0.689, but this cannot rescue WER |

The control's 32-row WER is only a screen and is not substituted for its prior
full qualification.  The paired result is nevertheless decisive for this
candidate: inverse-CDF increases mean WER by 60.31 percentage points and turns
the median sample from exact recognition into 68.75% error.  Per-item ASR shows
truncated, unrelated, and empty transcriptions, confirming that this is not
normal sample variance.  Distribution equivalence is insufficient because the
changed seed-to-code mapping materially changes the generated utterance.

The release gate is fail-fast.  Daily-Omni and Video-MME were not rerun after
the mandatory Seed-TTS WER failure; neither text-only benchmark can make a
WER-failing audio candidate releasable.  The previously qualified full results
(Daily-Omni 78.279% and Video-MME 70.259%) remain evidence for the accepted
lineage, not a waiver for this rejected sampler.  The inverse-CDF profile stays
explicitly experimental and the resident eager sampler remains the A2 default.

An initial post-replay synchronization diagnostic did not recover output
equivalence: all eight whole-audio hashes and all eight per-request chunk-hash
sequences differed from the resident eager sampler, and the synchronized graph
run took 7.17 seconds versus 6.75 seconds for the safe control. A second run
with both pre-replay and post-replay fences took 7.21 seconds and produced the
same 32.84 seconds of audio as the post-only run, while the safe control
produced 34.04 seconds. However, a same-service repeat produced 38.84 seconds,
so end-to-end WAV hashes are not stable enough to attribute the divergence to
the sampler alone. The next diagnostic compares graph inverse-CDF and resident
multinomial draws inside the same forward pass from an identical generator
state and fails on the first codec-token mismatch. All synchronization and
shadow profiles remain ineligible for submission.

The same-forward shadow then found the first actionable divergence at codec
step 1: uniform `0.916076303`, graph token `1303`, eager inverse-CDF token
`4218`, and resident multinomial token `4218`. This proves that the RNG mapping
and eager inverse-CDF arithmetic agree with the resident sampler for the real
runtime distribution. A second diagnostic fenced before replay, immediately
after replay, and before copying graph outputs. It still returned token `1303`
instead of `4218`. Cross-stream output ordering is therefore rejected as the
root cause: in the real FULL_DECODE service context this nested graph retained
the capture-time scalar rather than consuming the fixed-address uniform update.

Graph construction now includes a post-capture runtime-input probe, but the
same nested-graph failure allowed that capture-context probe to pass. The
authoritative gate therefore runs on the first real codec step from an
identical pre-draw generator state. If graph inverse-CDF, eager inverse-CDF and
native multinomial do not return the same token, serving returns the native
token, advances the canonical rolling frequency state and permanently disables
the sampler graph. The gate costs one shadow draw once and fails closed without
killing Stage 1. A future graph-internal RNG or multi-code Talker graph must
pass this real-runtime canary before any WER budget is spent.

```text
b64aa0073cee42b804a9742ebc0b9cb24e563a410dec034909510a120d78156e  resident-control-zh32.json
a13e96ee73dd2d0680c083d000f4203d759a455ba108003396d58b83ec43ad54  inverse-cdf-zh32.json
/tmp/lunanexa-bench/resident-control-v24-quality-zh32-local/
/tmp/lunanexa-bench/inverse-cdf-v26-quality-zh32-local/
```

## 2026-08-29 leaderboard refresh and next primary target

The public vLLM-Omni leaderboard reported an update time of
`2026-08-29 00:16:25`.  The retained submission was ninth:

| Rank | Team | RTF | TTFP | TTFT |
| ---: | --- | ---: | ---: | ---: |
| 1 | 李炎彬 / UCAS | 0.1066 | 568.96 ms | 49.20 ms |
| 2 | grounds / USTC | 0.1258 | 266.07 ms | 108.44 ms |
| 3 | 田峻钢 | 0.1274 | 678.33 ms | 46.66 ms |
| 4 | LinguistWantsTech | 0.1398 | 207.62 ms | 47.30 ms |
| 5 | 榴莲大王 | 0.1546 | 237.11 ms | 43.11 ms |
| 6 | KuaaMU | 0.1572 | 240.69 ms | 122.85 ms |
| 7 | 味蕾 | 0.1856 | 156.03 ms | 6.37 ms |
| 8 | 奶龙必胜 | 0.2191 | 371.90 ms | 160.20 ms |
| 9 | 向量贴贴 | 0.2423 | 514.22 ms | 45.72 ms |
| 10 | deltax | 0.2502 | 582.05 ms | 60.66 ms |

RTF remains the ranking axis.  Matching rank one requires a 56.00% reduction
from 0.2423; matching rank five requires 36.19%; matching rank seven requires
23.40%.  TTFT is already competitive, so more Stage-0 work is not the primary
score target.  The latest safe local resident-scalar result has not yet been
reflected in the public score and measured mean chunk RTF 0.179135, but local
and evaluator protocols are not interchangeable.

The last valid Stage-1 trace showed that the Talker is launch-bound and that
the remaining exact eager head/filter/multinomial tail is repeated for every
codec token.  The safe path already reduces the multinomial domain from 6,562
to the checkpoint's top-k of 25.  Therefore repeating the earlier full-vocab
exponential-race rejection would be the wrong experiment.

The first new screen combined that 25-value bounded distribution with an
exponential-race random input.  On desktop PyTorch 2.12.1 CPU, both sampled
token and generator state matched `torch.multinomial` for all 3,000 tested
shape/seed pairs.  That assumption did not hold on the actual A2 stack:

| A2 parity screen (`[1, 25]`, FP32) | Result | Gate |
| --- | ---: | --- |
| torch / torch_npu | 2.10.0 / 2.10.0 | recorded |
| Sampled-code parity | 106 / 1,000 | fail |
| Generator-state parity | 1,000 / 1,000 | pass |

Ascend's `MultinomialWithReplacement` therefore advances the generator by the
same amount but does not implement the PyTorch exponential-race token mapping.
The candidate failed before performance measurement, exactly as required by
the fail-fast gate.  Its runtime path and deploy YAML were removed; the safe
default is unchanged.  The NPU benchmark now uses the live top-k of 25 and
reports both token and generator-state parity so this result is reproducible.

This removes sampler-algorithm substitution from the near-term plan.  The
primary target is now a graph-visible multi-code/device-loop Talker executable:
the trace's roughly 83.5% device-idle fraction cannot be removed by another
isolated microkernel.  A useful prototype must keep the native multinomial
boundary or prove the complete accuracy gate, advance fixed-address KV and
metadata state on device, and amortize one host replay across multiple codec
steps.  Stage-1/Stage-2 overlap remains the second target, but it must first be
quantified with a two-process NPU timeline because both stages contend for the
same single-chip Cube/Vector resources.

### Follow-up: sampler math is exact; service state ordering is the suspect

The initial exponential-race failure still revealed that native multinomial
and a same-shaped random operation advance the NPU generator identically.
Additional A2 screens then recovered the actual one-sample mapping:

- CDF with the first value of a `[1, 25]` uniform slab matched native
  multinomial for 300/300 independently seeded draws.
- Both `[1, 25]`-slab and `[1, 1]`-scalar uniform CDF matched token and RNG
  state for 1,000/1,000 independently seeded draws.
- More importantly, both paths matched 1,000/1,000 draws while continuously
  advancing one request generator from seed 42, with no state mismatch.

The bounded helper was then captured alone in NPUGraph and compared with its
eager form for 300 synthetic BF16-derived logit rows. Candidate IDs were
300/300 identical, probabilities were bitwise identical with maximum error
zero, sampled codes were 300/300 identical, and generator state matched.
Plain eager and graph `topk` also returned identical IDs for 300/300 rows.

These results overturn the narrower RNG hypothesis: inverse-CDF and bounded
sampler math are exact relative to the current eager bounded path.  The
remaining service-only difference is the asynchronous lifetime of sampled-code
and rolling-frequency outputs across autoregressive steps.  A diagnostic
profile now places a device-wide fence after replay and state copies:

```text
vllm_omni/deploy/minicpmo_4_5_1npu_a2_evaluator_fia_bucket16_async_replay_sampler_graph_sync_diagnostic.yaml
```

It is not a performance candidate.  If it restores paired codec/audio hashes,
the fence will be replaced with a narrow recorded-event/current-stream wait.
If hashes still differ, stream ordering is rejected and the investigation
moves to full service state capture.  The profile cannot run concurrently with
the resident safe service: that service currently consumes 27,735 MiB of the
32-GiB A2, so the diagnostic requires an explicitly approved short restart.

A separate synthetic comparison found only 174/300 finite-mask parity and
43/300 sampled-code parity between the original full-vocabulary warper and the
bounded helper.  This is primarily a BF16 top-k boundary-tie concern and does
not explain bounded-eager versus bounded-graph divergence.  It does mean the
bounded optimization must retain its own official quality evidence; it must
not be described as unconditionally bitwise-equivalent to the original full
sort.

### 2026-08-29 12:09 leaderboard refresh

The public vLLM-Omni table moved materially again. The new top ten is:

| Rank | Team | RTF | TTFP | TTFT |
| ---: | --- | ---: | ---: | ---: |
| 1 | 纠纠在努力 | 0.0768 | 158.96 ms | 6.47 ms |
| 2 | 李炎彬 | 0.0976 | 506.12 ms | 48.03 ms |
| 3 | iin | 0.0987 | 214.37 ms | 48.20 ms |
| 4 | 奶龙必胜 | 0.1058 | 263.45 ms | 95.72 ms |
| 5 | 5.6-sol | 0.1133 | 212.41 ms | 104.99 ms |
| 6 | grounds | 0.1258 | 266.07 ms | 108.44 ms |
| 7 | 田峻钢 | 0.1274 | 678.33 ms | 46.66 ms |
| 8 | 榴莲大王 | 0.1290 | 698.39 ms | 47.28 ms |
| 9 | yesoryes | 0.1360 | 509.06 ms | 370.98 ms |
| 10 | LinguistWantsTech | 0.1398 | 207.62 ms | 47.30 ms |

The table refresh time is `2026-08-29 12:09:56`. The last published
`向量贴贴` result, RTF `0.2423`, is no longer in the truncated top ten. It is
68.30% above the leader and 42.29% above the tenth-place boundary. Even the
latest qualified local A2 path cannot close that gap with another Stage-2 or
metadata micro-optimization.

The ranking movement strengthens the architectural conclusion from the hot
trace. The next large experiment must execute several dependent codec steps
inside one isolated Talker worker command, with fixed-address KV/metadata,
graph-internal RNG or a proven native-sampler boundary, and device EOS masks.
It must retain the current asynchronous scheduler around that inner loop. The
minimum useful screen is a static two-step unroll: it should nearly halve the
per-code scheduler crossing count while preserving every sampled code and RNG
state. Only after two-step parity should the unroll grow to four or eight.

### Real-runtime sampler gate result

The first real Seed-TTS request confirmed that the capture-context probe was
not sufficient. On codec step 1 the nested graph returned code `1303`, while
eager inverse-CDF and the resident native multinomial both returned `4218` for
the same uniform value `0.916076303`. The real-runtime gate disabled the graph,
returned the native code, and allowed the request to finish. Stage 1 produced
122 stream units with mean ITL `5.789 ms`, the request returned HTTP 200, and
the service remained healthy. This is the required fail-closed behavior: the
experimental sampler can no longer terminate Stage 1 or silently reach an
accuracy run after its first observed mismatch.

The graph sampler remains diagnostic-only. The result also sharpens the design
requirement for a merged Talker graph: runtime RNG/state updates must be part of
the outer executable or use an Ascend graph-task update hook with explicit event
ordering. A nested capture that merely copies a scalar into a fixed-address
buffer is not a valid substitute in the FULL_DECODE service context.

### More aggressive host-bound plan

The Ascend multi-step graph RFC describes the closest upstream analogue to our
Talker bottleneck: capture one merged multi-step graph, update attention runtime
parameters once, and allocate distinct slot-mapping buffers for each internal
step. Applying that design to MiniCPM-o means a two-code Talker command first,
with two fixed KV/metadata slabs and one graph-visible EOS/commit decision.
Parity is checked after every internal code, including generator state. A
successful two-code screen is then extended to four codes. This attacks the
roughly 83.5% device-idle share in the earlier Talker trace rather than the
already-short device kernels.

The current three-stage single-NPU deployment also disabled CPU binding. Simply
enabling the stock per-NPU binder is unsafe because Thinker, Talker and Code2Wav
would all receive the same NPU-local CPU pool. The experimental profile below
instead partitions that pool into disjoint `1:2:1` slices, gives the host-bound
Talker half, keeps every slice NUMA-local, and deliberately leaves the shared
NPU IRQ placement untouched:

```text
vllm_omni/deploy/minicpmo_4_5_1npu_a2_evaluator_stage_cpu_slice_experimental.yaml
```

The supporting vLLM-Ascend implementation uses
`VLLM_ASCEND_CPU_BINDING_STAGE_SLICE=index:weight,weight,...`. It is opt-in and
must not replace the submission profile until a matched official-shape A/B run
improves RTF, P99 and TTFP. This is a high-leverage screen because the upstream
A2 CPU-binding experiment improved total token throughput from 124.08 to
146.93 token/s and TPOT from 11.78 to 9.90 ms in a host-bound decode workload;
the exact gain is workload-specific and is not assumed here.

The A2 screen confirmed the intended topology: Talker received CPUs 128-159
on NUMA 4, while Thinker and Code2Wav received disjoint 16-core slices on NUMA
5. The container did not permit `migratepages`; this is handled as a warning,
so affinity remained active without claiming page migration. The complete CPU
binding test file passed 89/89 tests. The 2-warmup + 8-request Seed-TTS run then
completed 8/8 requests with mean TTFP `489.319 ms`, mean chunk RTF `0.192519`
over 31 chunks, median chunk RTF `0.151948`, and Stage-1 ITL between roughly
`5.56` and `5.99 ms`. The immediately following safe control used the same
prompts, seed, 2 warmups and 8 measured requests. It completed in `6.8967 s`
with mean TTFP `479.319 ms`, mean TTFT `74.611 ms`, and mean chunk RTF
`0.187680` over the same 31 chunks. The stage-sliced candidate was worse by
2.60% in duration, 2.09% in TTFP, 5.04% in TTFT and 2.58% in mean chunk RTF.
It is therefore rejected as a submission default and retained only as an
opt-in experiment. The safe service was restored and returned HTTP 200 after
the control run.

After merged Talker execution, the next architecture screen is to deepen and
measure the overlap already enabled by `async_chunk`: prove on a two-process
timeline that Code2Wav chunk `n` overlaps Talker generation of chunk `n+1`,
then remove any remaining connector-side host fence. Inside Code2Wav, CFM and
HiFT can similarly use events and ping-pong buffers if the timeline shows idle
space rather than Cube/Vector contention. This follows the useful SGLang-Omni
separation between an AR scheduler and a streaming vocoder scheduler. Same-NPU
replicas are not the first move on this 32-GiB A2: the resident process already
uses about 27.7 GiB, so independent weight copies do not fit. Weight sharing
could revisit that idea later, but a single merged Talker executable removes
the same dispatch bubbles without a second model replica.

### Outer Talker graph: distribution and graph-owned codec state

The next accepted change moves the codec-head projection, repetition penalty,
bounded top-k/top-p filtering and softmax into the existing outer FULL_DECODE
graph. Native `torch.multinomial` deliberately remains outside the graph. Real
runtime gates at codec steps 1, 16 and 50 compare candidate logits,
probabilities and the sampled token against eager execution with an identical
cloned generator state. All three gates passed and the complete quality run
reported zero fail-closed events. The implementation is opt-in through
`VLLM_OMNI_MINICPMO45_NPU_FUSED_CODEC_DISTRIBUTION` and the profile:

```text
vllm_omni/deploy/minicpmo_4_5_1npu_a2_evaluator_fia_bucket16_async_replay_fused_distribution_experimental.yaml
```

On the matched 2-warmup + 8-request, concurrency-1 screen, this reduced request
duration from `6.868741 s` to `6.392827 s`, mean TTFP from `477.994 ms` to
`454.110 ms`, mean chunk RTF from `0.186887` to `0.170914`, and mean E2E from
`858.076 ms` to `798.406 ms`. Audio throughput improved from `4.955867` to
`5.362559` audio-seconds/s. TTFT regressed by `1.859 ms`, so this optimization
does not solve the Thinker first-token path.

A first attempt to replace the growing codec history with a host-maintained
16-code ring was rejected. Against its immediately adjacent control it
regressed duration by 4.70%, TTFP by 4.01%, mean chunk RTF by 7.42%, P99 chunk
RTF by 6.09%, and E2E by 4.70%. The external scalar ring write formed another
opaque producer/consumer boundary; fixed shape alone is not sufficient when
the state transition is still outside the executable. The implementation and
profile remain diagnostic-only and are not submission defaults.

The accepted replacement advances both the fixed 16-code FIFO and the
6562-entry repetition-frequency vector at the start of the next outer Talker
graph replay. Only the previous native sampled scalar crosses the graph
boundary. This is enabled by `VLLM_OMNI_MINICPMO45_NPU_GRAPH_CODEC_STATE` and:

```text
vllm_omni/deploy/minicpmo_4_5_1npu_a2_evaluator_fia_bucket16_async_replay_graph_codec_state_experimental.yaml
```

Two repeated concurrency-1 runs reproduced the favorable direction. The
stronger repeat used exactly the same `34.52 s` of generated audio as the
control and produced the following matched result:

| Metric | Control | Graph-owned state | Change |
| --- | ---: | ---: | ---: |
| Duration | 6.2393 s | 5.9242 s | -5.05% |
| Audio throughput | 5.5327x | 5.8269x | +5.32% |
| Mean TTFP | 441.24 ms | 423.79 ms | -3.95% |
| Mean chunk RTF | 0.16544 | 0.15691 | -5.16% |
| Median chunk RTF | 0.12946 | 0.12030 | -7.07% |
| P99 chunk RTF | 0.26530 | 0.25739 | -2.98% |
| Mean E2E | 779.45 ms | 740.07 ms | -5.05% |

The 32-item, concurrency-4 Seed-TTS quality gate then completed 32/32 requests
with WER `0.0500515` and SIM `0.832275`. The safe control was WER `0.0483154`
and SIM `0.831314`: WER changed by only +0.174 percentage points, comfortably
inside the +2-point rule, while SIM improved by `0.000961`. On this same
official-shape local protocol, duration fell from `21.9106 s` to `18.5383 s`
(-15.39%), audio throughput rose from `6.3166x` to `7.4700x` (+18.26%), mean
TTFP fell from `2204.40 ms` to `1869.83 ms` (-15.18%), and mean E2E fell from
`2585.15 ms` to `2191.32 ms` (-15.23%). These concurrency-4 values include
scheduler contention and must not be compared directly with the public
leaderboard's hidden evaluation result.

This closes the codec-distribution and codec-state host boundary, but it still
executes one dependent codec step per scheduler command. The remaining large
Talker target is therefore a parity-gated two-code merged command with two
fixed KV/slot-mapping slabs, followed by four/eight-code unrolling only if the
native sampler state and EOS commit decisions remain exact. That change can
remove scheduler crossings; another isolated elementwise fusion cannot.

### Rejected graph-embedding and cross-stage composition screens

Moving the previous-code `emb_code` lookup behind a resident placeholder into
the outer Talker graph was not semantics-safe. The three distribution gates
passed, but the same fixed eight prompts produced `64.52 s` instead of the
matched graph-state run's `34.52 s` of audio. Duration regressed from
`5.9242 s` to `14.4525 s`, mean TTFP from `423.79 ms` to `1090.77 ms`, and
mean E2E from `740.07 ms` to `1805.32 ms`. The graph was reading a stale or
incorrectly ordered cross-replay producer even though its local distribution
was valid. The implementation, tests and profile were removed. A future
multi-code graph must carry the sampled-code commit and dependent embedding in
one scheduler-owned executable with distinct KV slots; a resident input
placeholder is not sufficient.

The first attempt to compose graph-owned Talker state with the older complete
CFM3 profile also failed. The mixed inheritance chain silently retained old
Stage-1 sampler/chunk/PA settings, and the two independently fast graph paths
contended on one NPU. Its fully warm 8-request repeat generated `43.92 s` of
audio with mean RTF `0.30626`, TTFP `914.93 ms`, TTFT `732.26 ms` and E2E
`1610.43 ms`; it was substantially slower than either isolated path.

Per-process torch-npu stream priorities were then tested as a preemption
mechanism: Thinker `-2`, Talker `-1`, Code2Wav `0` (lower is higher priority).
All workers confirmed installation, but the matched `43.92 s` repeat worsened
RTF by `0.74%`, TTFP by `2.47%`, TTFT by `2.53%` and E2E by `0.46%`.
Priority streams do not preempt an already-submitted opaque CFM graph at the
granularity needed here, so the worker change was removed.

A final single-variable screen kept the qualified graph-owned Talker profile
unchanged and overlaid only the CFM3 Stage-2 environment. This exposed a
configuration mistake in the proposed composition: the current source policy
already defaults to the quality-qualified one-step CFM solver, so forcing
three steps increased Stage-2 work rather than reducing it. The fully warm
eight-request repeat generated `55.92 s` of audio with mean chunk RTF about
`0.28`, mean TTFP `1104.25 ms`, mean TTFT `733.68 ms` and mean E2E
`1966.93 ms`. The overlay and its test were removed. The accepted submission
path remains source-default CFM1 plus the graph-owned Talker distribution and
codec state; stage-local wins must not be composed without checking the
effective inherited policy and whole-request contention.

```text
/tmp/lunanexa-bench/graph-codec-embed-v57-zh8/
/tmp/lunanexa-bench/cfm3-graph-state-v62-repeat-zh8/
/tmp/lunanexa-bench/cfm3-graph-state-priority-v65-repeat-zh8/
/tmp/lunanexa-bench/cfm3-graph-state-clean-v68-repeat-zh8/
```

### CFM1 graph composition and isolated HF32 screen

A follow-up candidate preserved source-default CFM1 and combined the accepted
graph-owned Talker state with the legacy Stage-2 CFM graph, graph cache, BF16
FFN, HF32 MatMul and prompt-state switches. Runtime evidence showed that this
was not a valid graph composition: BF16 FFN disabled itself because the model
did not satisfy its BF16/channel-major preconditions, and CFM capture failed
during warmup because the current causal convolution still reached the ACLop
`Conv2D` implementation, which cannot execute during NPU graph capture. The
first warmup consequently took about 68 seconds. The fully warm 8-request run
did improve the immediately adjacent control on the identical 940,800-frame /
39.20-second output signature, but the only newly active numeric variable was
HF32 MatMul:

| Metric | Safe graph-state control | Mixed candidate | Change |
| --- | ---: | ---: | ---: |
| Duration | 6.8860 s | 6.7600 s | -1.83% |
| Audio throughput | 5.6927x | 5.7988x | +1.86% |
| Mean TTFT | 87.01 ms | 82.23 ms | -5.49% |
| Mean TTFP | 451.71 ms | 447.47 ms | -0.94% |
| Mean chunk RTF | 0.162626 | 0.159423 | -1.97% |
| Median chunk RTF | 0.125377 | 0.117793 | -6.05% |
| P99 chunk RTF | 0.273865 | 0.271914 | -0.71% |
| Mean E2E | 860.25 ms | 844.46 ms | -1.84% |

The mixed profile was removed rather than retaining two disabled or failed
features. HF32 was then isolated in
`minicpmo_4_5_1npu_a2_graph_codec_state_hf32_experimental.yaml`. Logs proved
that this profile enabled only Stage-2 HF32 MatMul and did not attempt a CFM
graph. Its two hot 8-request repeats averaged mean chunk RTF `0.157576`,
median chunk RTF `0.119796`, P99 chunk RTF `0.270899`, TTFT `86.32 ms`, TTFP
`452.54 ms`, and E2E `841.30 ms`. Relative to the adjacent control this is a
3.10% mean-RTF improvement, but TTFP is neutral/slightly worse (+0.18%). The
two repeats also generated 938,880 and 935,040 frames rather than the
control's 940,800, so their shorter wall time is not a strict output-matched
speed proof. HF32 therefore remains an explicit quality-gated experiment and
does not replace the accepted graph-state profile. The failed CFM graph and
BF16 flags are not carried by that profile.

This screen narrows the next major change further: the current CFM1 path is
already too small for another collection of Stage-2 flags to close the public
TTFP gap. The remaining high-leverage work is the scheduler-managed Talker
multi-code command described above, using preallocated KV/slot slabs and a
device-side sampled-token advance between dependent full-model replays.

```text
/tmp/lunanexa-bench/cfm1-graph-codec-state-v70-zh8/
/tmp/lunanexa-bench/graph-codec-state-v71-control-zh8/
/tmp/lunanexa-bench/graph-codec-state-hf32-v72-zh8/
/tmp/lunanexa-bench/graph-codec-state-hf32-v72-repeat2-zh8/
```

### Calibrated Talker static-W8A8 kernel rejection

The earlier dynamic-W8A8 screen left one important uncertainty: whether its
loss came mainly from recomputing the activation scale for every batch-one
Talker token. A second implementation therefore loaded the existing
32-request activation calibration, quantized only the 20 `gate_up_proj`
weights once, and used a fixed per-layer activation scale. Focused profile and
operator tests passed, but the real A2 `[1,768] x [768,6144]` kernel screen
rejected the path before an expensive service restart:

| Gate/up implementation | A2 latency | Relative to BF16 |
| --- | ---: | ---: |
| BF16 linear | 29.15 us | 1.000x |
| Fixed-scale W8A8, divide-mode quantize | 134.70 us | 0.216x |
| Fixed-scale W8A8, multiply-mode quantize | 105.10 us | 0.277x |
| Dynamic W8A8 | 124.04 us | 0.235x |

The fixed-scale result was numerically plausible (`MAE 0.00427`, maximum
absolute error `0.02148`, reference mean absolute value `0.26367`), but it was
not a speed optimization. Even the faster multiply-mode form would add roughly
`(105.10 - 29.15) * 20 = 1.52 ms` to every Talker token before counting any
other operator. That cannot improve the current roughly 6.7-ms Talker token
interval. On this batch-one shape, quantize launch and small INT8 matmul
overheads dominate the saved Cube work. The implementation, profile and
calibration copy were removed; the accepted graph-owned-codec profile remains
unchanged.

This closes the isolated precision/quantization branch. The next major target
is scheduler-owned multi-code Talker execution: reserve lookahead KV/slot
state, execute dependent replays under one device command, advance the sampled
code and position on device, and return two parity-checked codec tokens per
scheduler crossing. That architecture targets the launch/IPC/device-idle
budget that remains visible in the trace, rather than making the already-small
batch-one matrix arithmetic more expensive.

### Rejected graph-native codec sampler

The next experiment registered a model-owned `torch.Generator` with the
Stage-1 FULL ACL graph and moved native `torch.multinomial`, sampled-code
commit, repetition-frequency advance and the fixed 16-code FIFO advance into
the captured executable. This was technically successful: the A2 runtime
captured one explicit generator, replayed without the earlier generator-offset
error, and passed distribution/sample gates at codec steps 1, 16 and 50. The
generic vLLM-Ascend generator-registration hook and the MiniCPM-o integration
also passed their focused tests.

Two concurrency-1 repeats showed a real steady-state benefit relative to two
adjacent safe controls:

| Metric | Safe control mean | Graph-native sampler mean | Change |
| --- | ---: | ---: | ---: |
| Audio throughput | 5.6184x | 5.8711x | +4.50% |
| Mean chunk RTF | 0.16351 | 0.15515 | -5.11% |
| Median chunk RTF | 0.12685 | 0.11795 | -7.02% |
| P99 chunk RTF | 0.25850 | 0.25757 | -0.36% |
| Mean TTFP | 434.21 ms | 432.19 ms | -0.47% |
| Mean E2E | 762.23 ms | 765.18 ms | +0.39% |

The experiment nevertheless failed the mandatory accuracy gate. On the exact
same first 32 Chinese Seed-TTS rows used by the qualified graph-state control,
with `--disable-shuffle --no-oversample`, the candidate produced WER
`0.0787393` versus `0.0500515` (+2.869 percentage points) and SIM `0.789568`
versus `0.832275` (-0.042706). The WER loss exceeds the two-point limit. At
concurrency four, normalized audio throughput improved only from `7.46996x`
to `7.57522x` (+1.41%), while TTFP worsened from `1869.83 ms` to `1929.03 ms`
and E2E from `2191.32 ms` to `2274.12 ms` because of the longer sampled audio
and queueing.

The implementation, profile and capture hook were therefore removed rather
than promoted. This result also strengthens the next multi-code requirement:
distribution parity at sparse checkpoints is insufficient. A future device-
side sampler must prove request-seed and draw-offset parity for the complete
codec sequence, including capture warmup, request reset, EOS and scheduler
replay boundaries, before its performance result is considered eligible.

```text
/tmp/lunanexa-bench/graph-native-codec-v74-zh8/
/tmp/lunanexa-bench/graph-native-codec-v74-repeat2-zh8/
/tmp/lunanexa-bench/graph-native-codec-v74-adjacent-control-zh8/
/tmp/lunanexa-bench/graph-native-codec-v74-adjacent-control-repeat-zh8/
/tmp/lunanexa-bench/graph-native-codec-v74-quality-exact-control-zh32/
```

### Complete codec-sequence audit and cold Stage-2 isolation

An explicit `async_scheduling: true` Stage-1 overlay was first screened and
removed. The retained A2 profile already logs `OmniARAsyncScheduler` and
`Asynchronous scheduling is enabled`, so the overlay changed neither the
scheduler nor the executable. Apparent changes between repeated eight-prompt
runs were accompanied by different generated frame counts and could not be
claimed as speedups.

A diagnostic-only codec audit was then added behind
`VLLM_OMNI_MINICPMO45_CODEC_PARITY_TRACE`. The ordinary production path pays
no clone, host read or file-I/O cost when the variable is absent. An enabled
run records every sampled and published code, the exact distribution step,
the candidate IDs/probabilities immediately around native multinomial, and a
SHA-256 digest of the final request generator state. Request IDs are stored
only as SHA-256 digests; prompts and reference audio are never written.

The first same-prompt comparison produced 122 samples / 4.96 seconds of audio
on the first request after service start and 147 samples / 5.32 seconds on the
next request. The first six samples were equal and the first divergence was
zero-based step six. An initial analysis incorrectly aligned distribution row
zero with codec step zero even though the eligible full-decode distribution
begins at step one; trace format v2 now carries `distribution_steps` and binds
the pre/post snapshots to the sample at the multinomial call site.

Two controls ruled out a mutable graph-output race:

- device-wide fences before sampling and after the diagnostic copies produced
  exactly the same two sequences and generator digests as the no-fence run;
- two later requests with different request IDs were bit-for-bit identical:
  both produced 147 samples, 123 published codes and 5.32 seconds of audio,
  with identical candidate distributions and final generator digest.

Disabling the one-time fused-distribution runtime gate also left the first and
second sequences unchanged, so that unsafe bypass was removed. The boundary
instead matches the five-code first packet: after codec step five, Stage 2
starts its first real CFM/HiFT invocation on the colocated NPU. The cold run
took 67.90 seconds while the next run took 1.00 second because the former
absorbed Stage-2 compile/warmup. All subsequent unique-ID requests were
sequence-exact. This is therefore a cold colocated-runtime effect, not a
steady codec nondeterminism or a speed optimization. Competition measurements
already use warmup requests; production cold-start work should precompile the
real first-packet Stage-2 shape rather than weaken sampler parity gates.

```text
/tmp/lunanexa-bench/codec-parity-v76-nosync.jsonl
/tmp/lunanexa-bench/codec-parity-v77-sync.jsonl
/tmp/lunanexa-bench/codec-parity-v78-gate-off.jsonl
```

### Rejected deeper Stage-1 async batch queue

vLLM 0.26 normally permits two in-flight batches for single-chip async
scheduling. An experimental, MiniCPM-o-only profile raised this limit to four
before `EngineCore` construction so that the KV reservation and scheduler
queue used the same depth. The A2 runtime confirmed the exact path was active
(`Stage-1 async batch queue depth raised from 2 to 4`) and completed all eight
requests without an OOM or deadlock.

The fixed Chinese Seed-TTS screen nevertheless regressed against the adjacent
safe run:

| Metric | Safe run | Queue depth 4 | Change |
| --- | ---: | ---: | ---: |
| Total audio frames | 828,480 | 816,960 | -1.39% (not comparable) |
| Benchmark duration | 5.9543 s | 6.1682 s | +3.59% |
| Audio throughput | 5.7975x | 5.5186x | -4.81% |
| Mean chunk RTF | 0.15665 | 0.16634 | +6.19% |
| Mean TTFP | 432.15 ms | 443.82 ms | +2.70% |
| Mean E2E | 743.76 ms | 770.52 ms | +3.60% |

The shorter output makes the regression stronger, not weaker: the candidate
performed less audio work while taking longer. MiniCPM-o's next codec state is
model-worker dependent, so increasing the host batch queue cannot create a
valid multi-token executable. It also moves stop/EOS observation relative to
already queued work, explaining the changed output geometry. The code and
profile were removed. Any future lookahead must advance the codec state and
terminal mask inside one parity-proven device executable rather than enqueue
dependent one-token calls speculatively.

```text
/tmp/lunanexa-bench/queue4-v79-zh8/
```

### Accepted RTF-first fixed-slab CFM1 steady graph

The next candidate kept the accuracy-qualified source-default one-step CFM
solver, native sampling, 25-code streaming geometry and the default two-batch
Stage-1 queue.  It changed only recurrent Stage 2 work: the width-50/cache-402
CFM invocation now uses fixed-address planar KV slabs, BSH attention,
cache-major causal-pack state and two raw-NPUGraph output slots.  Prompt,
cache-fill and variable-width terminal work remain eager.  Runtime logs proved
that the fixed slabs were active and that both steady slots captured and
replayed; all startup numerical gates stayed below approximately `1e-6`.

The strict adjacent comparison completed the same eight requests and emitted
exactly `828,480` PCM frames / `34.52 s` of audio on both paths.  Excluding the
first and terminal packet of each request isolates the fixed steady shape for
the `steady chunk RTF` row:

| Metric (lower is better except throughput) | Adjacent safe control | Fixed-slab CFM1 graph | Change |
| --- | ---: | ---: | ---: |
| Benchmark duration | 6.059480 s | 6.028794 s | -0.51% |
| Audio throughput | 5.696858x | 5.725855x | +0.51% |
| Mean all-chunk RTF | 0.160083 | 0.159014 | -0.67% |
| Steady chunk RTF | 0.116831 | 0.106959 | -8.45% |
| Mean E2E | 756.92 ms | 753.08 ms | -0.51% |

A hot 32-request, concurrency-one stability run completed 32/32 requests,
emitted `3,323,520` frames / `138.48 s` of audio, and sustained `5.745616x`
audio throughput.  Its mean all-chunk RTF was `0.158998`, steady chunk RTF
was `0.109759`, mean TTFP was `438.61 ms`, and mean E2E was `752.70 ms`.
One cold eight-request run encountered a variable-width terminal compilation
outlier; it is not a steady-graph failure, but terminal shapes must remain a
separate cold-start concern.

The first same-protocol Seed-TTS quality gate also passed.  On the fixed first
32 Chinese rows, the candidate completed 32/32 with WER `0.0658364` and SIM
`0.832320`.  The matching qualified graph-state baseline was WER `0.0500515`
and SIM `0.832275`; the WER change is +1.5785 percentage points, inside the
two-point budget, while SIM is effectively unchanged.  Daily-Omni and
Video-MME were not rerun because this profile changes only the TTS Code2Wav
stage; that architectural isolation is not a substitute for the organizer's
full pre-submission gate.

Decision: retain
`minicpmo_4_5_1npu_a2_graph_codec_state_cfm1_static_graph_experimental.yaml`
as the fastest currently qualified local RTF candidate.  The 8.45% steady
CFM gain becomes only a 0.51% same-output request-duration gain, which proves
that further isolated Stage-2 kernel work is no longer the primary lever.
The next RTF work must reduce the dependent Talker-token cadence and the eager
first/tail packet tax without changing the native sampled codec sequence.

```text
/tmp/lunanexa-bench/graph-native-codec-v74-adjacent-control-repeat-zh8/
/tmp/lunanexa-bench/rtf-cfm1-static-q2-zh8-r2/
/tmp/lunanexa-bench/rtf-cfm1-static-q2-zh32/
/tmp/lunanexa-bench/rtf-cfm1-static-q2-quality-zh32/
/tmp/lunanexa-bench/rtf-cfm1-static-q2-sim-zh32/
```

A one-slot output-pool diagnostic was then rejected.  Its fully hot run
emitted the same `828,480` frames / `34.52 s` as the accepted two-slot run,
and logs proved `slot=1/1` capture plus one-slot replay.  Mean all-chunk RTF
regressed from `0.159014` to `0.162756` (+2.35%), steady chunk RTF from
`0.106959` to `0.109450` (+2.33%), and duration from `6.028794 s` to
`6.144140 s` (+1.91%).  The second output set is therefore earning its HBM
cost by avoiding a replay/output-consumer dependency.  The one-slot profile
was removed and the two-slot service restored.

```text
/tmp/lunanexa-bench/rtf-cfm1-static-slots1-zh8-r1/
/tmp/lunanexa-bench/rtf-cfm1-static-slots1-zh8-r2/
```

### Accepted graph-owned Talker control residency

The next exact-math change removes two per-codec host launches from the
graph-owned Talker path without changing its native sampler.  The captured
EOS-mask input is now written only when its Python boolean changes (normally
once at the minimum-code boundary), rather than before every replay.  The
legacy expired-code slab is not filled at all when the graph-owned FIFO is
active because that branch does not read it.  Request transitions invalidate
the cached boolean and retain the existing fixed tensor address.  Native
`torch.multinomial`, candidate probabilities, the repetition FIFO, EOS rules,
codec chunking and every model weight are unchanged.

The focused remote suite passed 8/8 tests for fused-distribution staging,
graph-owned controls and codec transport.  After correcting a parameter
wiring issue found only by a non-deferred-EOS fallback test, all 39 tests that
do not require the unavailable remote `pytest-mock` plugin passed.  The real
runtime distribution gates passed at codec steps 1, 16 and 50 with no
fail-closed event.  The first adjacent eight-request run also emitted the same
`816,960` frames / `34.04 s` as its control and improved mean chunk RTF from
`0.233439` to `0.231540` (-0.81%).

The retained decision uses the longer same-trajectory comparison.  Both
32-request paths emitted `3,323,520` frames / `138.48 s` of audio:

| Metric (lower is better except throughput) | Fixed-slab CFM1 control | Control residency | Change |
| --- | ---: | ---: | ---: |
| Benchmark duration | 24.101855 s | **23.815255 s** | **-1.19%** |
| Audio throughput | 5.745616x | **5.814760x** | **+1.20%** |
| Mean audio RTF | 0.175709 | **0.173685** | **-1.15%** |
| Mean all-chunk RTF | 0.158998 | **0.157028** | **-1.24%** |
| Median all-chunk RTF | 0.120768 | **0.120263** | **-0.42%** |
| P99 all-chunk RTF | 0.275375 | **0.258882** | **-5.99%** |
| Mean TTFP | 438.61 ms | **433.73 ms** | **-1.11%** |
| Mean E2E | 752.70 ms | **743.80 ms** | **-1.18%** |

Because this changes only redundant writes into already captured control
slabs, the preceding CFM1 Seed-TTS WER/SIM gate remains the relevant model
quality evidence.  It does not replace the organizer's final Daily-Omni,
Video-MME and Seed-TTS submission gate.

An additional attempt retained the newly allocated native
`multinomial + gather` scalar directly until the chunk boundary instead of
cloning it.  Its two hot 32-request runs had mean chunk RTF `0.171393` and
`0.152871`; their mean, `0.162132`, was 3.25% slower than control residency,
and both P99 values (`0.279031` / `0.266401`) were worse than `0.258882`.
The faster second run did not compensate for the large variance and first-run
regression on the ranked mean, so direct retention was removed.  The service
and submission source keep the explicit ownership copy.

```text
/tmp/lunanexa-bench/rtf-control-q2-20260830/
/tmp/lunanexa-bench/rtf-control-residency-q2-candidate-20260830/
/tmp/lunanexa-bench/rtf-control-residency-q2-candidate-zh32-20260830/
/tmp/lunanexa-bench/rtf-control-residency-no-clone-q2-zh32-20260830/
/tmp/lunanexa-bench/rtf-control-residency-no-clone-q2-zh32-repeat-20260830/
```
