# MiniCPM-o 4.5 on Ascend 910C: real-data pilot

Date: 2026-08-09

This report records a small, real-data validation run for the competition
workload. It is a pilot, not a full competition score. Its purpose is to prove
the end-to-end dataset, request, response, metric, and accuracy paths before
running the full suites and before comparing optimization candidates.

## Environment

- Model: `OpenBMB/MiniCPM-o-4_5`
- Serving stack: source checkouts of vLLM, vLLM Ascend, and vLLM-Omni
- Host: Ascend 910C machine
- Active service allocation: two NPUs, with all three MiniCPM-o stages healthy
- API: OpenAI-compatible `/v1/chat/completions`
- Request concurrency: 1
- Decoding: temperature 0, thinking disabled
- Output modality: text for Daily-Omni and Video-MME; audio for Seed-TTS

The server was launched with Ascend custom ops disabled because the installed
CANN 9 environment did not expose the required custom-op package. This is a
compatibility baseline, not the intended final optimized configuration.

## Results

| Dataset | Coverage | Success | Accuracy | Mean TTFT | P90 TTFT | Mean E2E | P90 E2E |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Daily-Omni | 20 QA, seed 1 | 20/20 | 15/20 (75.0%) | 3739 ms | 5024 ms | 3778 ms | 5052 ms |
| Video-MME | 30 QA from 10 videos | 30/30 | 15/30 (50.0%) | 1368 ms | 2399 ms | 1391 ms | 2424 ms |
| Seed-TTS EN | 30 utterances | 30/30 | not evaluated | 314 ms | 375 ms | 2450 ms | 3192 ms |

Seed-TTS audio metrics:

| Metric | Mean | P50 | P90 | P99 |
| --- | ---: | ---: | ---: | ---: |
| Audio TTFP | 1107 ms | 1045 ms | 1474 ms | 1716 ms |
| Whole-audio RTF | 0.586 | 0.532 | 0.858 | 1.103 |
| Per-chunk RTF | 0.626 | 0.350 | 1.249 | 1.971 |

The Seed-TTS run generated 126.8 seconds of audio with 100% request
continuity. Whole-audio RTF below 1 means generation was faster than real time
on average; the P99 above 1 shows a latency tail that still needs optimization.

## Dataset and request details

- Daily-Omni used real local videos and the MiniCPM interleaved packing mode,
  which samples one frame per second with a maximum of 64 frames and inserts
  the video's audio between the frame groups.
- Video-MME used the first ten available ordered video IDs and all three
  questions for each video.
- Seed-TTS used the English voice-cloning split with real reference audio.
- Daily-Omni and Video-MME produced deterministic multiple-choice outputs that
  were scored in the benchmark runner.
- The Daily-Omni scorer accepts both the official one-letter answer and mirrors
  that store the complete selected option text.

Raw results on the benchmark host:

```text
/workspace/user_data/lunanexa-stack/results/minicpmo45-real-pilot-daily-omni-interleave-seed1-n20-20260809T075419Z.json
/workspace/user_data/lunanexa-stack/results/minicpmo45-real-pilot-seed-tts-en-c1-n30-20260809T075642Z.json
/workspace/user_data/lunanexa-stack/results/minicpmo45-real-pilot-video-mme-10videos-n30-20260809T075842Z.json
```

## What this run proves

- Real Daily-Omni video and audio can be decoded on the aarch64 server even
  when `decord` is unavailable; the OpenCV fallback preserves the same frame
  indices and sampling policy.
- The MiniCPM-specific multimodal interleave reaches the deployed service.
- Text accuracy, TTFT, E2E latency, audio TTFP, whole-audio RTF, and per-chunk
  RTF are emitted and persisted by one serving benchmark path.
- The service remains healthy after all three workloads.

## What this run does not prove

- It does not cover the full Daily-Omni, Seed-TTS, or Video-MME suites.
- It does not establish the competition's at-most-two-percentage-point accuracy
  regression requirement because an official framework baseline has not been
  run with the same samples and settings.
- Seed-TTS WER and speaker-similarity accuracy are not present. The host still
  needs the official ASR and similarity evaluator dependencies/checkpoints.
- The run used two NPUs, so it is not a one-NPU minimum-footprint result.
- Concurrency-one numbers do not establish capacity or saturation throughput.
- Warm-up requests may populate multimodal caches, so these latency numbers
  must not be compared with a cold-cache run unless the cache policy matches.
- The installed vLLM and vLLM-Omni source versions report a major/minor mismatch
  warning; the final competition image must pin a validated compatible set.

## Next benchmark gate

Run the unmodified framework baseline and the optimized fork from clean service
starts with the same manifests and sample IDs. Record cold and warm cache runs
separately, add concurrency sweeps, and reject any optimization whose full-suite
accuracy drops by more than two percentage points. Before claiming a competition
result, run all 1,196 Daily-Omni questions, the complete selected Seed-TTS split
with WER and speaker similarity, and all 2,700 Video-MME questions.
