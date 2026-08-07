# MiniCPM-o 4.5

> Online serving and offline inference for omni multimodal chat
> (text / image / audio / video → text + 24 kHz speech)

## Summary

- Vendor: OpenBMB
- Model: [`openbmb/MiniCPM-o-4_5`](https://huggingface.co/openbmb/MiniCPM-o-4_5)
- Task: Omni multimodal chat — accepts text / image / audio / video input;
  emits text and 24 kHz mono speech in the same response
- Mode: Online serving via the OpenAI-compatible `/v1/chat/completions`
  API (plus Gradio demo), and offline inference via `Omni.generate`
- Maintainer: [`@tc-mb`](https://github.com/tc-mb) (MiniCPM-V / MiniCPM-o team)

## When to use this recipe

Use this recipe as a known-good starting point for serving
`openbmb/MiniCPM-o-4_5` on vLLM-Omni. MiniCPM-o 4.5 is the omni member
of the MiniCPM-o family — it runs a multimodal thinker, a streaming
MiniCPMTTS codec talker, and a separate batched Code2Wav stage so a single
`/v1/chat/completions` call can return text and 24 kHz speech in one
shot. The recommended batching deploy isolates the Thinker on GPU 0 and
co-locates Talker and Code2Wav on GPU 1; 1-GPU, 3-GPU, and 8x4090 layouts are
also provided.

## References

- Default deploy configs (auto-loaded by HF `model_type=minicpmo` +
  `hf_config.version="4.5"`):
  - Default single-GPU compatibility layout (auto-loaded):
    [`vllm_omni/deploy/minicpmo_4_5.yaml`](../../vllm_omni/deploy/minicpmo_4_5.yaml)
  - 2-GPU and 3-GPU layouts:
    [`vllm_omni/deploy/minicpmo_4_5_2gpu.yaml`](../../vllm_omni/deploy/minicpmo_4_5_2gpu.yaml),
    [`vllm_omni/deploy/minicpmo_4_5_3gpu.yaml`](../../vllm_omni/deploy/minicpmo_4_5_3gpu.yaml)
  - Development-only 2-NPU profiler overlay:
    [`vllm_omni/deploy/minicpmo_4_5_2npu_profile.yaml`](../../vllm_omni/deploy/minicpmo_4_5_2npu_profile.yaml)
  - Atlas A3 / Ascend 910C 2-NPU candidate overlay:
    [`vllm_omni/deploy/minicpmo_4_5_2npu_910c.yaml`](../../vllm_omni/deploy/minicpmo_4_5_2npu_910c.yaml)
  - 8x RTX 4090 layout:
    [`vllm_omni/deploy/minicpmo_4_5_8x4090.yaml`](../../vllm_omni/deploy/minicpmo_4_5_8x4090.yaml)
- Online example + Gradio demo:
  [`examples/online_serving/minicpmo/`](../../examples/online_serving/minicpmo/)
- Offline end-to-end example:
  [`examples/offline_inference/minicpmo/`](../../examples/offline_inference/minicpmo/)
- Pipeline / talker source:
  [`vllm_omni/model_executor/models/minicpmo_4_5/`](../../vllm_omni/model_executor/models/minicpmo_4_5/)
- Stage-input processors (thinker → talker and talker → Code2Wav):
  [`vllm_omni/model_executor/stage_input_processors/minicpmo_4_5_omni.py`](../../vllm_omni/model_executor/stage_input_processors/minicpmo_4_5_omni.py)
- Upstream model card:
  [`openbmb/MiniCPM-o-4_5`](https://huggingface.co/openbmb/MiniCPM-o-4_5)
- Ascend compatibility and A3 topology:
  [vLLM-Ascend installation](https://docs.vllm.ai/projects/ascend/en/latest/installation.html),
  [vLLM-Ascend quick start](https://docs.vllm.ai/projects/ascend/en/latest/quick_start.html)
- Performance architecture and 910C promotion gates:
  [MiniCPM-o 4.5 streaming optimization on Ascend 910C](../../docs/design/minicpmo_4_5_ascend_910c_optimization.md)
- Integration PR:
  [vllm-project/vllm-omni#3642](https://github.com/vllm-project/vllm-omni/pull/3642)

## Hardware Support

Four hardware layouts ship with deploy configs. Every layout uses the
same strict three-stage topology. The Talker emits codec chunks only;
Code2Wav consumes them through a shared-memory async connector.

| Layout | Thinker | Talker | Code2Wav | Typical hardware |
| --- | --- | --- | --- | --- |
| 1-GPU (default) | GPU 0 | GPU 0 | GPU 0 | 1x large-memory GPU |
| 2-GPU | GPU 0 | GPU 1 | GPU 1 | 2x large-memory GPU |
| 3-GPU | GPU 0 | GPU 1 | GPU 2 | 3x GPU |
| 2-NPU candidate | NPU 0 | NPU 1 | NPU 1 | Atlas A3 / 2x Ascend 910C |
| 8x RTX 4090 24GB | GPU 0–3 (TP=4) | GPU 4 | GPU 5 | 8x RTX 4090 consumer |

### Migration from the fused deployment

MiniCPM-o 4.5 now requires the three-stage topology: the Talker owns
request-local codec generation and Code2Wav owns waveform state and
reference-voice prompt features. `minicpmo_4_5.yaml` remains the stable
single-GPU entry point; `minicpmo_4_5_2gpu.yaml` is the recommended
two-GPU profile. The removed fused two-stage implementation is not retained as
a fallback because it would duplicate state machines and correctness paths.

## Ascend NPU / 910C qualification

MiniCPM-o 4.5 is listed as supported on Ascend NPU and ships an NPU-aware
Talker/Code2Wav adapter. The NPU platform overlay runs the Thinker and Talker
with PIECEWISE ACL graphs while keeping the dynamic, request-owned Code2Wav
stage eager.

Treat a reported "910C" chip name as hardware discovery, not as the complete
runtime selection. Confirm the server product and topology with `npu-smi info`.
If it is an Atlas A3 product, use the A3 vLLM-Ascend image/software row and
allocate at least two NPUs as required by the vLLM-Ascend A3 quick start. Keep
vLLM, vLLM-Ascend, PyTorch, torch-npu, CANN, and NNAL on one documented
compatibility row; for source builds, use the vLLM commit recorded by the
checked-out vLLM-Ascend tree in `.github/vllm-main-verified.commit`.

On a two-NPU host, start with the supplied split layout:

```bash
npu-smi info
git -C /workspace/vllm-ascend rev-parse HEAD
git -C /workspace/vllm-omni rev-parse HEAD

VLLM_OMNI_MINICPMO45_INITIAL_CODEC_CHUNK_FRAMES=25 \
VLLM_OMNI_MINICPMO45_CODEC_CHUNK_FRAMES=25 \
VLLM_OMNI_MINICPMO45_CODEC_LEFT_CONTEXT_FRAMES=3 \
VLLM_OMNI_MINICPMO45_TOKEN2WAV_N_TIMESTEPS=10 \
vllm serve openbmb/MiniCPM-o-4_5 --omni \
    --deploy-config vllm_omni/deploy/minicpmo_4_5_2npu_910c.yaml \
    --trust-remote-code \
    --allowed-local-media-path /data/benchmarks \
    --interleave-mm-strings \
    --host 0.0.0.0 --port 8099
```

The baseline intentionally preserves checkpoint-quality settings. Qualify one
change at a time in this order: initial chunk (TTFP), steady chunk (per-chunk
RTF), Code2Wav steps (RTF/quality), then concurrency and stage memory limits
(TTFT/throughput). Run each candidate from a fresh server process so graph and
prompt caches do not contaminate the comparison. Promote only candidates that:

1. complete every request with no missing audio chunks or server errors;
2. improve the target latency distribution over three repeated runs;
3. keep Daily-Omni, Seed-TTS, and Video-MME evaluated counts identical to the
   baseline; and
4. pass the fail-closed two-percentage-point quality gate below.

The vLLM-Ascend fork should remain model-agnostic until an NPU profiler trace
attributes material time to a generic Ascend operator or graph/runtime path.
MiniCPM codec policy, vocoder steps, and inter-stage chunking belong here in
vLLM-Omni. This prevents unprofiled model-specific patches from leaking into
the hardware plugin.

### Optimization decision matrix

The optimization boundary is settled as follows. "Remote gate" means the
mechanism is implemented, but its winning value must be measured on the target
910C/Atlas product rather than guessed on another accelerator.

| Optimization | Current decision | Owner / promotion evidence |
| --- | --- | --- |
| Thinker, Talker, Code2Wav stage isolation | Shipped; retain the three-stage topology | vLLM-Omni; end-to-end correctness and stage metrics |
| Local inter-stage transfer | Async SHM shipped; raw-tensor header/buffer format and event wakeup are opt-in 910C candidates | vLLM-Omni; transfer metrics, CPU utilization, and profiler trace |
| Thinker/Talker graph execution on NPU | Shipped as PIECEWISE ACL graph mode | vLLM-Omni platform overlay; 910C startup and accuracy gates |
| Code2Wav graph/compile | Eager default; exact-shape CFM graph replay is implemented behind `VLLM_OMNI_MINICPMO45_NPU_CFM_GRAPH=1` | Long-session cache correctness, trace attribution, and all quality gates |
| Initial codec chunk | Remote gate: sweep 8/12/16 against 25 | vLLM-Omni policy; TTFP target plus RTF/quality guards |
| Steady codec chunk | Remote gate after initial chunk settles | vLLM-Omni policy; per-chunk RTF target plus TTFP/quality guards |
| Code2Wav diffusion steps | Remote gate: sweep 8/6 against 10 | vLLM-Omni policy; RTF target plus official Seed-TTS quality |
| Concurrency and stage memory | Remote gate after single-request latency | Deploy config; throughput target plus TTFT/TTFP/RTF guards |
| Prefix/Radix caching | Do not enable for this pipeline yet | Requires proof that request-owned audio state and multimodal keys are safe |
| Async compute/transfer overlap and dedicated NPU streams | Profiler-triggered, not speculative | vLLM-Ascend only when the bottleneck is generic across models |
| Ascend operator/kernel changes | Profiler-triggered, model-agnostic only | vLLM-Ascend trace, operator table, and cross-model regression evidence |

The native-duplex Stage 0 already preserves a resumable request and its
scheduler-owned KV state across audio appends. Do not introduce a second
session abstraction. Append-only validation, TTL/reaping, cancellation,
pending-input limits, and max-session admission already exist. The next
multi-session production boundary is fair scheduling and explicit metrics for
session-held KV memory. The full
mechanism-level rationale—including the performance ideas extracted from
Thinking Machines' work and the SGLang streaming-session implementation—is in
the [910C optimization design](../../docs/design/minicpmo_4_5_ascend_910c_optimization.md).

### Deterministic shadow comparison

For qualification runs, capture exact audio fingerprints in saved benchmark
JSON without retaining production audio:

```bash
VLLM_OMNI_BENCH_CAPTURE_OUTPUT_HASHES=1 \
vllm bench serve --omni ... --save-result
```

Compare the same prompts and seed across concurrency, graph, and chunking modes:

```bash
python -m vllm_omni.benchmarks.determinism_gate \
  baseline.json candidate.json \
  --field generated_texts \
  --field audio_content_sha256s \
  --field audio_chunk_sha256s \
  --report-json determinism-report.json
```

This gate is exact and fail-closed. It supplements rather than replaces the
Daily-Omni, TTS-Seed, and Video-MME two-percentage-point quality gate.

This incorporates the useful SGLang/SGLang-Omni ideas—independent stage
scheduling, continuous batching, direct local transfer, and overlapping
communication—without copying Radix caching or static-graph assumptions across
MiniCPM-o's request-local audio state boundary. Exact-shape-compatible
Code2Wav batching is already enabled; batches with incompatible cache shapes
must remain separate for correctness.

### Capture an actionable 910C profile

Latency measurements and profiler measurements are separate runs because the
profiler changes timing. Start a fresh server with the development-only
overlay; it writes one trace and operator workbook per stage:

```bash
mkdir -p /data/profiles/minicpmo45/{stage0,stage1,stage2}

vllm serve openbmb/MiniCPM-o-4_5 --omni \
    --deploy-config vllm_omni/deploy/minicpmo_4_5_2npu_profile.yaml \
    --trust-remote-code \
    --allowed-local-media-path /data/benchmarks \
    --interleave-mm-strings \
    --host 0.0.0.0 --port 8099
```

From another shell, use a representative text+audio subset and let the
benchmark start and stop every configured stage profiler:

```bash
vllm bench serve --omni \
    --backend openai-chat-omni \
    --endpoint /v1/chat/completions \
    --model openbmb/MiniCPM-o-4_5 \
    --dataset-name daily-omni \
    --dataset-path liarliar/Daily-Omni \
    --daily-omni-video-dir /data/benchmarks/Daily-Omni/Videos \
    --daily-omni-pack-mode minicpm-interleave \
    --num-prompts 16 --max-concurrency 1 --profile
```

Archive the server log, `npu-smi info`, both repository commits, the exact
commands, and all three `/data/profiles/minicpmo45/stage*` directories. A
vLLM-Ascend change is justified only when these artifacts attribute material
time to a generic NPU operator, synchronization, transfer, or graph-runtime
path. If time remains in MiniCPM chunk policy or Code2Wav model code, make the
change in vLLM-Omni and rerun both performance and quality gates.

## GPU

### 1 x GPU (default — single command)

The default
[`vllm_omni/deploy/minicpmo_4_5.yaml`](../../vllm_omni/deploy/minicpmo_4_5.yaml)
co-locates Thinker, codec-only Talker, and Code2Wav on GPU 0. Their
`gpu_memory_utilization` budgets are 0.55, 0.15, and 0.15. The remaining
device memory is left available for runtime workspaces such as the HiFi-GAN
vocoder's cuDNN kernels. All stages admit
up to four sequences. Startup video profiling is bounded to 32 frames per
video. Use the two-GPU profile for production throughput.

#### Environment

- OS: Linux
- Python: 3.10+
- vLLM / vLLM-Omni: >= 0.21.0 (or current `main`)
- Optional Talker dep: `stepaudio2-minicpmo` (see Notes for why this is
  required and how to install it)

#### Command

```bash
vllm serve openbmb/MiniCPM-o-4_5 --omni \
    --trust-remote-code \
    --host 0.0.0.0 --port 8099
```

The deploy config is auto-loaded by the model registry — no
`--deploy-config` flag needed for this default single-GPU layout.

For the recommended two-GPU layout, add:

```bash
--deploy-config vllm_omni/deploy/minicpmo_4_5_2gpu.yaml
```

#### Performance comparison

Compare text-only and text+audio separately. Text-only isolates Thinker
generation; text+audio also schedules Talker and Code2Wav. The following full
Daily-Omni runs used the same two GPUs, 1197 samples, concurrency 10, and
identical `enable_thinking=false` / `use_tts_template=true` request settings.
The `origin/main` fused Talker ran eager because its graph capture copied an
unpinned CPU metadata tensor.

| Metric | `origin/main` two-stage | Three-stage batching |
| --- | ---: | ---: |
| Accuracy | 64.83% | 64.83% |
| Throughput | 0.62 req/s | 1.97 req/s |
| Mean E2EL | 16.17 s | 5.07 s |
| Mean serving TTFT | 0.92 s | 1.28 s |
| Mean audio TTFP | 16.17 s | 3.24 s |
| Mean audio RTF | 5.97 | 2.11 |
| Stage 0 mean TPOT / ITL | 8.27 / 8.27 ms | 40.08 / 40.11 ms |
| Stage 0 median TPOT / ITL | 7.23 / 7.24 ms | 7.43 / 7.53 ms |

The split pipeline improves throughput 3.19x and lowers audio TTFP by 80%.
Isolating the Thinker on GPU 0 also removes the prior single-GPU TPOT
regression: 40.08 ms is slightly better than the pre-rebase report (~44 ms).
Its median TPOT is effectively the same as main; the higher mean is queueing
tail latency because this profile bounds each stage to four sequences while
main's Thinker admits 16. Global TPOT/ITL remains zero when serving emits text
as one aggregated chunk, so the table reports Stage 0 metrics.

#### Chunk-latency tuning

The default Talker-to-Code2Wav window is 25 codec frames with 3 frames of
left context. The first window defaults to the same 25 frames, but can be made
smaller independently to reduce TTFP without forcing every later Code2Wav
invocation to use the less efficient size. Controlled sweeps can override all
three values without editing the deploy YAML:

```bash
VLLM_OMNI_MINICPMO45_INITIAL_CODEC_CHUNK_FRAMES=8 \
VLLM_OMNI_MINICPMO45_CODEC_CHUNK_FRAMES=25 \
VLLM_OMNI_MINICPMO45_CODEC_LEFT_CONTEXT_FRAMES=3 \
VLLM_OMNI_MINICPMO45_TOKEN2WAV_N_TIMESTEPS=10 \
vllm serve openbmb/MiniCPM-o-4_5 --omni \
    --trust-remote-code --host 0.0.0.0 --port 8099
```

Use `vllm bench serve --percentile-metrics audio_ttfp,audio_rtf,audio_chunk_rtf`
to compare runs. `audio_chunk_rtf` measures every chunk's delivery interval
against its playable duration; for the first chunk the interval is request
start to TTFP. Values below 1 keep pace with realtime. Saved results
include each request's raw `audio_ttfps` and `audio_chunk_rtfs` arrays as well
as the aggregate percentiles.
Smaller steady chunks can improve delivery cadence but increase Code2Wav
invocation and transfer overhead. Start by sweeping only the initial chunk
(for example 8, 12, 16, then 25 as the baseline), then sweep the steady chunk
only if per-chunk RTF remains above the target. Keep the default as the
baseline, change one variable at a time, and re-run the required audio-quality
evaluation before promoting a setting. Both chunk sizes must be positive and
left-context frames non-negative.

Code2Wav uses 10 diffusion steps by default. Once chunk geometry is settled,
sweep the step count separately (for example 8 and 6 against the 10-step
baseline). Fewer steps reduce vocoder work and may reduce audio quality, so a
candidate must pass both Seed-TTS metrics and the full two-percentage-point
quality gate before promotion. Step counts must be positive.

Video-MME is available as a native benchmark dataset. Download the licensed
videos according to the official Video-MME instructions, place the MP4 files
under one local directory using their `videoID` names, and run:

```bash
vllm bench serve --omni \
    --backend openai-chat-omni \
    --endpoint /v1/chat/completions \
    --model openbmb/MiniCPM-o-4_5 \
    --dataset-name video-mme \
    --dataset-path lmms-lab/Video-MME \
    --video-mme-video-dir /data/Video-MME/videos \
    --num-prompts 2700 \
    --max-concurrency 1 \
    --save-result
```

Start the server with `--allowed-local-media-path /data/Video-MME/videos`, or
add `--video-mme-inline-local-video` for small smoke runs. The saved JSON
contains overall accuracy plus official duration, domain, sub-category, and
task-type breakdowns.

Run Daily-Omni with the MiniCPM interleaving protocol and Seed-TTS with content
and speaker-similarity evaluation enabled:

```bash
vllm bench serve --omni \
    --backend openai-chat-omni \
    --endpoint /v1/chat/completions \
    --model openbmb/MiniCPM-o-4_5 \
    --dataset-name daily-omni \
    --dataset-path liarliar/Daily-Omni \
    --daily-omni-video-dir /data/benchmarks/Daily-Omni/Videos \
    --daily-omni-input-mode all \
    --daily-omni-pack-mode minicpm-interleave \
    --num-prompts 1197 --max-concurrency 1 \
    --percentile-metrics ttft,audio_ttfp,audio_rtf,audio_chunk_rtf \
    --save-result --result-filename daily-omni.json

vllm bench serve --omni \
    --backend openai-chat-omni \
    --endpoint /v1/chat/completions \
    --model openbmb/MiniCPM-o-4_5 \
    --dataset-name seed-tts \
    --dataset-path /data/benchmarks/seed-tts-eval \
    --seed-tts-wer-eval --seed-tts-sim-eval \
    --seed-tts-official-export-dir /data/results/baseline/seed-tts-en \
    --num-prompts 1000 --max-concurrency 1 \
    --percentile-metrics ttft,audio_ttfp,audio_rtf,audio_chunk_rtf \
    --save-result --result-filename seed-tts.json
```

Use the official dataset size present in the selected Seed-TTS split if it is
not 1000; the quality comparator requires the candidate to use the same count
as its baseline.

The in-process `--seed-tts-sim-eval` metric is a fast WavLM mean-pooling
proxy. It is useful while sweeping but is not the official Seed-TTS SIM
protocol. For promotion, run the official evaluator against the exported
`{utterance_id}.wav` files. `cal_wer.sh` and `cal_sim.sh` both write
`wav_res_ref_text.wer`, so preserve each report before running the next:

```bash
cd /opt/seed-tts-eval
export ARNOLD_WORKER_GPU=2

bash cal_wer.sh \
    /data/benchmarks/seed-tts-eval/en/meta.lst \
    /data/results/baseline/seed-tts-en en
cp /data/results/baseline/seed-tts-en/wav_res_ref_text.wer \
    /data/results/baseline/seed-tts-en-wer.txt

bash cal_sim.sh \
    /data/benchmarks/seed-tts-eval/en/meta.lst \
    /data/results/baseline/seed-tts-en \
    /models/wavlm_large_finetune.pth
cp /data/results/baseline/seed-tts-en/wav_res_ref_text.wer \
    /data/results/baseline/seed-tts-en-sim.txt

python -m vllm_omni.benchmarks.seed_tts_official_result \
    seed-tts.json \
    --wer-report /data/results/baseline/seed-tts-en-wer.txt \
    --sim-report /data/results/baseline/seed-tts-en-sim.txt \
    --output baseline-seed-tts-official.json
```

Repeat for the candidate using the same meta file and evaluated count. The
importer refuses reports without both a summary and per-utterance rows, labels
the exact protocol in JSON, and never overwrites the original serving result.

After producing baseline and candidate result files for each required suite,
enforce the two-percentage-point rule with the fail-closed comparator:

```bash
python -m vllm_omni.benchmarks.quality_gate \
    baseline-daily-omni.json candidate-daily-omni.json \
    --require-metric daily_omni_accuracy_incl_http_fail \
    --require-evaluated-count daily_omni_evaluated=1197 \
    --max-regression-pp 2

python -m vllm_omni.benchmarks.quality_gate \
    baseline-video-mme.json candidate-video-mme.json \
    --require-metric video_mme_accuracy_incl_http_fail \
    --require-evaluated-count video_mme_evaluated=2700 \
    --max-regression-pp 2

python -m vllm_omni.benchmarks.quality_gate \
    baseline-seed-tts-official.json candidate-seed-tts-official.json \
    --require-metric seed_tts_content_error_mean \
    --require-metric seed_tts_sim_mean \
    --require-evaluated-count seed_tts_content_evaluated=1000 \
    --require-evaluated-count seed_tts_sim_evaluated=1000 \
    --require-seed-tts-official \
    --max-regression-pp 2
```

The comparator also requires identical evaluated counts, preventing a faster
candidate from passing by dropping failed or difficult requests.

For each performance sweep, save three distinct baseline files and three
distinct candidate files. Then make the latency/throughput decision
machine-checkable. This example promotes a steady-chunk candidate only if the
median p99 chunk RTF improves by at least 1%, while median TTFP and overall RTF
regress by no more than 2%:

```bash
python -m vllm_omni.benchmarks.performance_gate \
    --baseline baseline-1.json --baseline baseline-2.json --baseline baseline-3.json \
    --candidate candidate-1.json --candidate candidate-2.json --candidate candidate-3.json \
    --target-metric p99_audio_chunk_rtf \
    --guard-metric p99_audio_ttfp_ms \
    --guard-metric mean_audio_rtf \
    --min-improvement-percent 1 \
    --max-guard-regression-percent 2 \
    --report-json performance-promotion.json
```

The gate rejects duplicate paths, fewer than three runs, request failures,
different completed counts, and missing/zero/non-finite metrics. For the
concurrency sweep, use `--target-metric request_throughput
--higher-is-better request_throughput`; keep TTFT, TTFP, and RTF as guards.
Run the quality gates above as a separate mandatory promotion condition.

#### Verification

**Quick smoke test (text-only output)**:

```bash
curl http://localhost:8099/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "openbmb/MiniCPM-o-4_5",
        "messages": [{"role": "user", "content": "Briefly introduce yourself."}],
        "modalities": ["text"]
    }'
```

**Text + speech in one response** (the headline 4.5 feature). The model
bridge conditions the Talker from the generated assistant span, so the
generic serving layer does not inject MiniCPM-specific template defaults.
`use_tts_template=true` remains supported when explicitly requested:

```bash
curl http://localhost:8099/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "openbmb/MiniCPM-o-4_5",
        "messages": [{"role": "user", "content": "Say hello, then introduce vLLM in one sentence."}],
        "modalities": ["text", "audio"],
        "chat_template_kwargs": {"use_tts_template": true}
    }'
```

When using the OpenAI Python SDK, the same flag can also be sent as
`extra_body={"chat_template_kwargs": {"use_tts_template": True}}`
because the client merges `extra_body` into the request root.

Response carries text in one choice's `message.content` and base64 WAV
in another choice's `message.audio.data` (24 kHz mono, see Notes). With
`modalities: ["text", "audio"]` you typically get two `choices` entries
(one text, one audio).

**Streaming text + speech** (use `--stream`):

```bash
python examples/online_serving/minicpmo/openai_chat_completion_client_for_multimodal_generation.py \
    --query-type text \
    --prompt "Say hello, then introduce vLLM in one sentence." \
    --port 8099 \
    --stream
```

The client prints text deltas as they arrive and saves streamed audio chunks
to WAV files.

**Gradio demo (text + image + audio + video UI)**:

```bash
bash examples/online_serving/minicpmo/run_gradio_demo.sh
# or run the python entry point directly:
python examples/online_serving/minicpmo/gradio_demo.py \
    --minicpmo45-api-base http://localhost:8099/v1 \
    --minicpmo45-model openbmb/MiniCPM-o-4_5 \
    --port 7862
```

Open `http://<host>:7862` and try a text prompt with the **"Generate
speech output (TTS)"** checkbox on / off.

#### Notes

- Memory budget: Thinker, Talker, and Code2Wav reserve 0.55, 0.15, and
  0.15 of GPU 0. The larger Thinker share protects its multimodal KV cache,
  while the unreserved memory remains available to runtime kernels; all three
  model processes still share one CUDA device.
- `--trust-remote-code` is required — the HF repo ships a custom
  `MiniCPMO` config / model class.
- Stage 0 Thinker and Stage 1 Talker enable vLLM CUDA Graphs. Stage 2 remains
  eager because its request-owned Flow/HiFT caches and variable chunk/cache
  shapes are not yet exposed through a static exact-shape graph wrapper.
- All default stages use `max_num_seqs: 4` to reduce cross-process GPU
  contention. Talker AR
  state and Code2Wav caches are request-owned; Code2Wav batches only
  exact-shape-compatible chunks and does not fall back to serial decode.
- `limit_mm_per_prompt.video.num_frames: 32` bounds startup dummy profiling,
  not runtime media decoding. Use `media_io_kwargs.video.num_frames` when a
  matching URL/file video sampling limit is required.
- `StageRequestStats.batch_size` is a request-scoped placeholder, not the
  scheduler's execution batch.
- Single-GPU co-location trades throughput for hardware density: Stage 0/1
  CUDA Graph replay and eager Stage 2 vocoder kernels compete across three
  CUDA contexts. Use the 8x4090 config or a custom multi-GPU mapping for
  throughput-sensitive serving.

### 8 x RTX 4090 24GB (consumer-GPU layout)

Use
[`vllm_omni/deploy/minicpmo_4_5_8x4090.yaml`](../../vllm_omni/deploy/minicpmo_4_5_8x4090.yaml)
on an 8x RTX 4090 host. Thinker uses 4-way TP across GPUs 0–3
(`~85 %` mem each ≈ 20.4 GiB/card), Talker uses GPU 4, and Code2Wav
uses GPU 5. GPUs 6–7 are left free.

#### Command

```bash
vllm serve openbmb/MiniCPM-o-4_5 --omni \
    --deploy-config vllm_omni/deploy/minicpmo_4_5_8x4090.yaml \
    --trust-remote-code \
    --host 0.0.0.0 --port 8099
```

#### Notes

- `max_model_len` is capped at 4096 in this layout — 8192 still OOMs on
  4090s. Raise it if your cards have more headroom (e.g. 4090 D /
  custom 32 GB SKUs), but verify with a long-prompt run before
  promoting.
- All other knobs match the single-GPU section; the only difference is
  the per-card memory pressure on the thinker shards.

## Notes (applies to all layouts)

- **Code2Wav dependency**: Stage 2 loads `Token2wav` from the
  MiniCPM-o-flavored
  vocoder (PyPI package `stepaudio2-minicpmo` — NOT the upstream
  `stepfun-ai/Step-Audio2`, whose `Token2wav.__init__` signature
  rejects `n_timesteps`). Install via the published extra:

  ```bash
  pip install stepaudio2-minicpmo
  ```

  A missing dep raises `ImportError` at first request with the same
  install hint instead of silently emitting empty audio.

- **TTS conditioning**: the MiniCPM stage bridge can condition speech from
  the generated assistant span without changing shared serving code.
  `chat_template_kwargs.use_tts_template=true` remains supported when an
  explicit `<|tts_bos|>` boundary is desired. For **curl**, put
  `chat_template_kwargs` at the request root; the OpenAI Python SDK may use
  `extra_body` because it flattens those fields into the root.

- **Reference voice**: request audio is carried on the first codec chunk.
  Code2Wav owns the temporary prompt WAV and prompt-feature cache, and removes
  both when the stream ends.

- **Talker sampling**: codec-token sampling reads the checkpoint `tts_config`
  and defaults to deterministic seed 42. Stage-1 deploy sampling parameters
  control only vLLM's binary continue/stop token.

- **Output audio**: 24 kHz mono WAV inside the OpenAI-style
  `message.audio.data` (base64). The Gradio demo's WAV player decodes
  this automatically.

- **Routing**: MiniCPM-o 4.5 and 2.6 both ship `architectures=
  ["MiniCPMO"]` in HF config; routing is disambiguated by
  `hf_config.version == "4.5"` via the
  `hf_config_predicate` on the 4.5 pipeline. A 2.6 checkpoint loaded
  with this recipe's `--deploy-config` will be rejected at startup
  rather than silently misrouted.

- **Async chunking**: enabled in all deploy configs. Talker sends
  25-code chunks with three-code left context to Code2Wav through
  `SharedMemoryConnector`; terminal chunks flush held lookahead state.
- **Response choices**: text and audio are separate choices. SDK clients
  should select the choice whose `message.audio.data` is populated rather
  than assuming `choices[0]` contains audio.
