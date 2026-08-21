from vllm_omni.entrypoints.cli.main import _needs_ascend_benchmark_fast_exit


def test_ascend_benchmark_fast_exit_is_narrowly_scoped() -> None:
    ascend_env = {"ASCEND_HOME_PATH": "/usr/local/Ascend"}

    assert _needs_ascend_benchmark_fast_exit(
        ["vllm", "bench", "serve", "--omni"],
        ascend_env,
    )
    assert not _needs_ascend_benchmark_fast_exit(
        ["vllm", "serve", "model", "--omni"],
        ascend_env,
    )
    assert not _needs_ascend_benchmark_fast_exit(
        ["vllm", "bench", "serve", "--omni"],
        {},
    )


def test_ascend_benchmark_fast_exit_can_be_disabled() -> None:
    assert not _needs_ascend_benchmark_fast_exit(
        ["vllm", "bench", "serve", "--omni"],
        {
            "ASCEND_TOOLKIT_HOME": "/usr/local/Ascend",
            "VLLM_OMNI_DISABLE_ASCEND_BENCH_FAST_EXIT": "true",
        },
    )
