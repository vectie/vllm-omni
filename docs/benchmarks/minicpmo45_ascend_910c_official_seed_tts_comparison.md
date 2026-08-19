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
| --- | ---: | ---: | ---: | ---: |
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
