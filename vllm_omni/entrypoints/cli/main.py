"""
CLI entry point for vLLM-Omni that intercepts vLLM commands.
"""

import importlib.metadata
import os
import sys
from collections.abc import Mapping, Sequence


def _needs_ascend_benchmark_fast_exit(
    argv: Sequence[str],
    environ: Mapping[str, str],
) -> bool:
    """Return whether a completed benchmark should skip native teardown.

    The Ascend A3 challenge image can abort in the torch-npu/ACL process
    destructors after a short-lived CLI has already completed successfully.
    Restrict the workaround to the benchmark subprocess; the long-running
    server and all exception paths retain their normal lifecycle.
    """
    return (
        len(argv) > 1
        and argv[1] == "bench"
        and bool(environ.get("ASCEND_HOME_PATH") or environ.get("ASCEND_TOOLKIT_HOME"))
        and environ.get("VLLM_OMNI_DISABLE_ASCEND_BENCH_FAST_EXIT", "0").lower()
        not in {"1", "true", "yes"}
    )


def _ascend_benchmark_fast_exit_if_needed() -> None:
    if not _needs_ascend_benchmark_fast_exit(sys.argv, os.environ):
        return
    # Benchmark result files are written by the dispatcher. Flush terminal
    # output explicitly because os._exit intentionally bypasses Python atexit
    # handlers as well as the broken native destructors.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def main():
    """Main CLI entry point that intercepts vLLM commands."""
    # Check if --omni flag is present
    if "--omni" not in sys.argv:
        from vllm.entrypoints.cli.main import main as vllm_main

        vllm_main()
        return
    else:
        # Force colored logging even when piped (e.g. `| tee`).
        # Must be set before any vLLM import because the logger
        # formatter is configured at import time via _use_color().
        if "VLLM_LOGGING_COLOR" not in os.environ:
            os.environ["VLLM_LOGGING_COLOR"] = "1"

        from vllm.entrypoints.serve.utils.api_utils import VLLM_SUBCMD_PARSER_EPILOG, cli_env_setup

        import vllm_omni.entrypoints.cli.benchmark.main
        import vllm_omni.entrypoints.cli.serve
        from vllm_omni.utils.tracking_parser import TrackingArgumentParser

        CMD_MODULES = [
            vllm_omni.entrypoints.cli.serve,
            vllm_omni.entrypoints.cli.benchmark.main,
        ]

        cli_env_setup()

        from vllm_omni.entrypoints.cli.serve import _ensure_vllm_platform

        _ensure_vllm_platform()

        parser = TrackingArgumentParser(
            description="vLLM OMNI CLI",
            epilog=VLLM_SUBCMD_PARSER_EPILOG.format(subcmd="[subcommand]"),
        )
        try:
            _omni_version = importlib.metadata.version("vllm_omni")
        except importlib.metadata.PackageNotFoundError:
            try:
                from vllm_omni.version import __version__ as _omni_version  # type: ignore
            except Exception:
                _omni_version = "dev"
        parser.add_argument(
            "-v",
            "--version",
            action="version",
            version=_omni_version,
        )
        subparsers = parser.add_subparsers(required=False, dest="subparser")
        cmds = {}
        for cmd_module in CMD_MODULES:
            new_cmds = cmd_module.cmd_init()
            for cmd in new_cmds:
                cmd.subparser_init(subparsers).set_defaults(dispatch_function=cmd.cmd)
                cmds[cmd.name] = cmd
        args = parser.parse_args()
        if args.subparser in cmds:
            cmds[args.subparser].validate(args)

        if hasattr(args, "dispatch_function"):
            args.dispatch_function(args)
            _ascend_benchmark_fast_exit_if_needed()
        else:
            parser.print_help()


if __name__ == "__main__":
    main()
