import argparse
import asyncio
import os
from typing import Any

from vllm.benchmarks.serve import main_async

# Import patch to register Omni datasets and backends
# This monkey-patches vllm.benchmarks.datasets.get_samples before it's used
# Must be imported before any vllm.benchmarks module usage
import vllm_omni.benchmarks.patch.patch  # noqa: F401
from vllm_omni.benchmarks.patch.patch import (
    maybe_enable_stage_metrics,
    set_print_stage,
    should_request_stage_metrics,
)


def main(args: argparse.Namespace) -> dict[str, Any]:
    seed_tts_sim_eval = getattr(args, "seed_tts_sim_eval", False)
    official_export_dir = getattr(args, "seed_tts_official_export_dir", None)
    if getattr(args, "seed_tts_wer_eval", False) or seed_tts_sim_eval:
        os.environ["SEED_TTS_WER_EVAL"] = "1"
    if seed_tts_sim_eval:
        os.environ["SEED_TTS_SIM_EVAL"] = "1"
    if official_export_dir:
        os.environ["SEED_TTS_OFFICIAL_EXPORT_DIR"] = str(official_export_dir)
    if getattr(args, "seed_tts_wer_save_items", False):
        os.environ["SEED_TTS_WER_SAVE_ITEMS"] = "1"
    if getattr(args, "daily_omni_save_eval_items", False):
        os.environ["DAILY_OMNI_SAVE_EVAL_ITEMS"] = "1"
    if getattr(args, "videomme_save_eval_items", False) or getattr(
        args, "video_mme_save_eval_items", False
    ):
        os.environ["VIDEOMME_SAVE_EVAL_ITEMS"] = "1"
    set_print_stage(getattr(args, "print_stage", False))
    args.extra_body = maybe_enable_stage_metrics(
        getattr(args, "extra_body", None),
        enabled=should_request_stage_metrics(args),
    )
    return asyncio.run(main_async(args))
