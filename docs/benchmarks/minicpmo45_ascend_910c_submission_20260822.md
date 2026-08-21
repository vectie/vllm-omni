# MiniCPM-o 4.5 Ascend 910C challenge submission

Date: 2026-08-22

## Candidate

- Track: vLLM-Omni
- Repository: `git@github.com:vectie/vllm-omni.git`
- Branch: `challenge/minicpm-910c-submit-clean`
- Benchmarked code commit:
  `8b704e22811252c787eb38480440f85b4af48f9d`
- Upstream base:
  `a964efc55b6c36ed6a9214a8cf4bb131f368183d`
- Hardware: Atlas A3 / Ascend 910C, one visible NPU
- Image: `quay.io/ascend/vllm-omni:v0.25.0-a3`
- Model: OpenBMB MiniCPM-o 4.5

The complete source includes `.git`, has no Git object alternate, and passes
`git fsck --full` after independent archive extraction. Evaluator-style
`pip install -e . --no-build-isolation` and official-config
`vllm serve --omni` startup were reproduced in the organizer image.

## Full accuracy qualification

| Benchmark / metric | Organizer gate | Candidate | Evaluated | Result |
| --- | ---: | ---: | ---: | --- |
| Daily-Omni accuracy | >= 77.5% | 78.279% | 1,197 | pass |
| Video-MME accuracy | >= 67.0% | 70.259% | 2,700 | pass |
| Chinese Seed-TTS WER | <= 1.56% | 1.4424% | 2,020 | pass |
| Chinese Seed-TTS ASV SIM | >= 0.689 | 0.848011 | 2,020 | pass |

Daily-Omni completed with 937/1,197 correct and zero HTTP failures.
Video-MME completed with 1,897/2,700 correct and zero HTTP failures. The
Chinese Seed-TTS run completed 2,020/2,020 requests in 2,182.63 seconds at
concurrency four. Median WER was 0.0 and median SIM was 0.852046. Request,
PCM, ASR, missing-reference, and SIM-embedding failure counts were all zero.

Chinese Seed-TTS raw result SHA-256:

```text
585c4493743a9c08d328239f397fc055b528b6d97bb92425d19ab8f3199760d7  official-accuracy-zh2020-cfm6.json
```

The full benchmark and optimization history, including the Daily-Omni and
Video-MME raw-result checksums, is in
`docs/benchmarks/minicpmo45_ascend_910c_official_seed_tts_comparison.md`.

## Chinese Seed-TTS accuracy command

```bash
SEED_TTS_SIM_EVAL=1 vllm bench serve --omni \
  --host 127.0.0.1 --port 8099 \
  --model /models/MiniCPM-o-4_5 \
  --backend openai-chat-omni \
  --endpoint /v1/chat/completions \
  --dataset-name seed-tts \
  --dataset-path /data/seed-tts-eval-zh \
  --seed-tts-locale zh \
  --num-prompts 2020 --num-warmups 0 \
  --max-concurrency 4 --request-rate inf \
  --no-oversample --disable-shuffle --trust-remote-code \
  --extra-body '{"modalities":["text","audio"],"chat_template_kwargs":{"enable_thinking":false,"use_tts_template":true}}' \
  --seed-tts-wer-eval --seed-tts-wer-save-items \
  --save-result
```

## Official-shape performance preflight

Lower is better for all three organizer metrics.

| Metric | Organizer F16 baseline | Candidate | Improvement |
| --- | ---: | ---: | ---: |
| RTF | 0.4423 | 0.383662 | 13.26% |
| TTFP | 986.47 ms | 871.67 ms | 11.64% |
| TTFT | 333.27 ms | 301.41 ms | 9.56% |

The retained performance run used Chinese Seed-TTS, concurrency one, two
warmups, and 32 measured requests. It completed 32/32 with no failures,
generated 133.68 seconds of audio, and finished in 51.00 seconds.

Raw performance result SHA-256:

```text
98eb07b0a2bf72aee36c2550b4e44ca6603429a789c1cfe0dd11c56f89510e07  official-a3-c1-n32-r2.json
```

## Promoted optimizations

1. Six-step Code2Wav CFM after paired speed and quality screening.
2. Cached invariant CFM timestep embeddings and exact Euler deltas.
3. Ascend DiT MLP graph partition for CFG batch 2, width 50, hidden 512.
4. Cached projected prompt-speaker state and optimized Talker repetition
   penalty.
5. Competition-qualified Stage-0 c4 admission and 16K Thinker prefill budget.
6. Eager MiniCPM-o registry initialization for the official multi-process
   architecture.
7. A benchmark-only Ascend fast exit after persisted results to avoid native
   ACL destructor corruption in the supplied container.

Experiments that improved isolated kernels but regressed end-to-end latency,
P99, or quality were not promoted. See
`docs/design/minicpmo_4_5_ascend_910c_optimization.md` for the architecture,
rejected candidates, and promotion gates.
