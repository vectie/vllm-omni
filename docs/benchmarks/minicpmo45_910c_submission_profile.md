# MiniCPM-o 4.5 Ascend 910C submission profile

Date: 2026-08-22

This branch is the evaluator-facing candidate. The organizer installs the
candidate source but supplies its own deploy YAML, so the release policy must
live in source rather than in a candidate-only profile.

## Default policy

The organizer allocates one Atlas 800I A3 card.  The card exposes two logical
910C chips, so this is a dual-chip target even though the allocation count is
one card.  When the evaluator's untouched `minicpmo_4_5.yaml` and two visible
NPU devices are detected, source policy now applies the measured topology and
execution path:

- Thinker and Talker remain on logical chip 0; Code2Wav moves to chip 1;
- Thinker and Talker use `FULL_DECODE_ONLY` capture sizes 1, 2, and 4;
- Code2Wav uses six CFM evaluations, homogeneous BF16 DiT/CFM integration,
  fixed planar KV slabs, and bounded graph-visible partitions;
- bounded TorchAir partitions remain visible to GE.  The rejected opaque
  custom-op, BSH/static-CFM, and complete GE-monolith experiments are not
  enabled.

The policy is deliberately conditional.  A single-chip host retains the
one-device path, and any explicit non-baseline stage placement or compile mode
retains operator authority.

Set `VLLM_OMNI_MINICPMO45_NPU_OPTIMIZED_DEFAULTS=0` to restore the checkpoint
step count.  Set `VLLM_OMNI_MINICPMO45_A3_DUAL_CHIP=0` to disable the complete
automatic A3 topology policy.  The planar and graph components can be screened
independently with `VLLM_OMNI_MINICPMO45_NPU_PLANAR_DEFAULTS=0`; the rejected
static path remains explicit opt-in through
`VLLM_OMNI_MINICPMO45_NPU_CFM_GRAPH=1` and
`VLLM_OMNI_MINICPMO45_NPU_CFM_GRAPH_CACHE=1`.  Explicit deploy values continue
to override source defaults.

## Dual-chip qualification

The final screen used the official Chinese Seed-TTS request contract, 32
deterministic requests, two wrapper warmups, concurrency one, seed zero,
temperature zero, and the organizer deploy YAML.  Each candidate received a
complete 32-request graph/kernel warm run before its independent measurement.
Lower is better except throughput.

| Variant | Mean RTF | Mean E2E | Mean TTFT | Mean TTFP | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Previous dual-chip CFM6 control | 0.37981 | 1,830.42 ms | 363.92 ms | 927.82 ms | control |
| Fixed planar BF16 with evaluator `PIECEWISE` | 0.382 | 1,843.82 ms | 362.39 ms | 899.81 ms | reject incomplete integration |
| Planar BF16 + full decode, official YAML | 0.30786 | 1,480.20 ms | 340.09 ms | 806.57 ms | accepted base |
| Add BSH layout only | 0.3089 | 1,486.65 ms | 339.38 ms | 801.50 ms | reject on mean RTF |
| Add two-slot steady CFM graph, explicit profile run 1 | 0.30398 | 1,468.16 ms | 336.82 ms | 793.83 ms | promising isolation |
| Add two-slot steady CFM graph, explicit profile run 2 | 0.30445 | 1,469.45 ms | 340.82 ms | 797.72 ms | promising isolation |
| Same graph auto-enabled through organizer YAML | 0.567 | 2,734.98 ms | 476.50 ms | 1,161.87 ms | reject as default |

The accepted evaluator path is therefore the planar/full-decode row at
0.307862 RTF.  Relative to the previous dual-chip CFM6 control, mean RTF fell
18.94%, mean E2E fell 19.13%, mean TTFT fell 6.55%, and mean TTFP fell 13.07%.
It completed 32/32, preserved 559 output tokens and 3,737,280 audio frames,
reported 100% streaming continuity, and had zero underrun.

This closes most of the gap to the reported 0.2964 leaderboard result without
changing the checkpoint.  It is not a multiple because isolated kernel wins
are bounded by Talker decoding, prompt/tail work, HiFT, and inter-stage
latency.  The source-policy A/B also proved why the earlier fusion result was
hidden: leaving Stage 0/1 on `PIECEWISE` consumed the time saved by Stage 2.

The explicit static-graph profile remains useful research evidence, but it is
not the submission default.  Besides the evaluator-entry slowdown, its two
P99 RTF values were 0.36016 and 0.36056 versus 0.34876 for the accepted planar
base.  The source-level promotion gate correctly rejected a candidate that
won only behind its private profile boundary.

## Release gates

The six-step release lineage passed the full required quality gates:

| Gate | Result |
| --- | ---: |
| Daily-Omni | 937 / 1,197 = 78.279% |
| Video-MME | 1,897 / 2,700 = 70.259% |
| Chinese Seed-TTS | 2,020 / 2,020 completed |
| Seed-TTS WER | 1.4424% |
| Seed-TTS SIM | 0.848011 |

The accepted fixed-planar increment also passed a matched 32-row offline gate:
mean WER stayed exactly at 0.016588 and mean WavLM proxy SIM changed by only
0.010 percentage points.  The rejected BSH/static-graph profiles separately
passed their small paired quality screens, so their rejection is strictly a
performance/integration decision.

The complete combined layout/graph candidate has not yet rerun all 2,020
Seed-TTS rows plus full Daily-Omni and Video-MME after this promotion.  The
six-step lineage has passed those full gates, and the incremental paired
screens are strong regression evidence, but the full organizer quality run
remains the final submission gate.

Performance artifacts are under:

```text
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-dual-source-default-final-official-yaml-20260822
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-dual-full-decode-planar-bsh-20260822
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-dual-bsh-cfm-graph-20260822
/workspace/user_data/lunanexa-stack/experiments/minicpmo45-dual-source-default-bsh-cfm-final-official-yaml-20260822
```
