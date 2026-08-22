# MiniCPM-o 4.5 Ascend 910C submission profile

Date: 2026-08-22

This branch is the evaluator-facing candidate. The organizer installs the
candidate source but supplies its own deploy YAML, so the release policy must
live in source rather than in a candidate-only profile.

## Default policy

On Ascend NPU, Code2Wav uses six CFM evaluations by default. This is the only
new Stage-2 default. The deeper BF16, fixed/planar KV slab, BSH-attention,
TorchAir partition, native causal-pack, and HiFT graph implementations remain
available for controlled experiments, but are not silently activated by the
official one-device deployment.

Set `VLLM_OMNI_MINICPMO45_NPU_OPTIMIZED_DEFAULTS=0` to restore the checkpoint
step count. Set `VLLM_OMNI_MINICPMO45_NPU_AGGRESSIVE_EXPERIMENTS=1` to enable
the retained all-feature research bundle. Explicit deploy values continue to
override source defaults.

## Why the larger bundle is not the default

The final screen used the official Chinese Seed-TTS request contract, 32
deterministic requests, two warmups, concurrency one, all three stages on one
logical 910C device, and the organizer deploy YAML. Lower is better.

| Variant | Mean whole-audio RTF | Mean TTFP | Decision |
| --- | ---: | ---: | --- |
| CFM6-only hot control | 0.396 | 856 ms | promote |
| Complete aggressive bundle | 0.4166 | 853 ms | retain as opt-in |
| BF16-only hot control | 0.495 | 1,029 ms | reject as default |
| Final no-private-env repeat | 0.420 | 890 ms | evaluator-path verified |

The 0.396 and 0.420 repeats show material host variance, so they are not used
to claim a leaderboard-equivalent score. They do show that BF16 and the full
bundle do not provide a robust one-device RTF win. The final service log
confirmed `token2wav_n_timesteps=6`, FP32 Stage 2, no BSH/fixed-slab activation,
and no HiFT graph activation without candidate-specific environment flags.

## Release gates

The six-step release lineage passed the full required quality gates:

| Gate | Result |
| --- | ---: |
| Daily-Omni | 937 / 1,197 = 78.279% |
| Video-MME | 1,897 / 2,700 = 70.259% |
| Chinese Seed-TTS | 2,020 / 2,020 completed |
| Seed-TTS WER | 1.4424% |
| Seed-TTS SIM | 0.848011 |

The final default-policy unit suite passed 177 tests on the target environment.
The retained aggressive bundle has not independently passed all three complete
release gates and must not be described as the qualified submission path.

The observed one-device result is still above the leaderboard's 0.2964. That
gap requires another measured architecture change; it is not closed by
enabling every existing kernel switch.
