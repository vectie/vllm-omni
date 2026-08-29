from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[2]


def test_minicpmo45_runtime_has_no_evaluator_conditioned_shortcuts():
    """Submission runtime must not recognize or rewrite evaluation requests."""
    runtime_files = (
        _REPO_ROOT / "vllm_omni/entrypoints/openai/serving_chat.py",
        _REPO_ROOT
        / "vllm_omni/model_executor/stage_input_processors/minicpmo_4_5_omni.py",
    )
    forbidden = (
        "minicpmo45_tts_teacher_forcing",
        "_prepare_minicpmo45_teacher_forcing",
        "_apply_minicpmo45_teacher_forcing_sampling",
        "completed_spans",
    )

    for path in runtime_files:
        source = path.read_text(encoding="utf-8")
        for marker in forbidden:
            assert marker not in source, f"evaluation-conditioned shortcut {marker!r} found in {path}"


def test_minicpmo45_runtime_does_not_import_benchmark_data_modules():
    runtime_roots = (
        _REPO_ROOT / "vllm_omni/entrypoints/openai",
        _REPO_ROOT / "vllm_omni/engine",
        _REPO_ROOT / "vllm_omni/model_executor",
        _REPO_ROOT / "vllm_omni/platforms",
        _REPO_ROOT / "vllm_omni/worker",
    )
    forbidden_imports = (
        "vllm_omni.benchmarks",
        "daily_omni_dataset",
        "seed_tts_dataset",
        "videomme_dataset",
    )

    for root in runtime_roots:
        for path in root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            for marker in forbidden_imports:
                assert marker not in source, f"benchmark dependency {marker!r} found in runtime file {path}"


def test_output_changing_score_experiments_are_not_source_defaults():
    source = (_REPO_ROOT / "vllm_omni/config/stage_config.py").read_text(
        encoding="utf-8"
    )
    disabled_defaults = (
        '_MINICPMO45_SINGLE_CHIP_CFM1_DEFAULT_ENV, "0"',
        '_MINICPMO45_SINGLE_CHIP_CFM2_DEFAULT_ENV, "0"',
        '_MINICPMO45_SINGLE_CHIP_RTF_FIRST47_DEFAULT_ENV, "0"',
        '_MINICPMO45_SINGLE_CHIP_RTF_TERMINAL600_DEFAULT_ENV, "0"',
    )
    for marker in disabled_defaults:
        assert marker in source, f"output-changing default is not disabled: {marker}"
