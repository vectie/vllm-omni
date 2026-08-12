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
