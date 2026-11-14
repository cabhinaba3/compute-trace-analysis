"""Run a Python script or module under ``jax.profiler.trace`` and write an XProf log directory.

    python -m cta.profile.jax_runner --out logdir [--no-perfetto] -- script.py [args...]
    python -m cta.profile.jax_runner --out logdir -- -m package.module [args...]

The log directory holds ``plugins/profile/<run>/<host>.xplane.pb`` plus, by default, a Chrome-format trace
(``create_perfetto_trace=True``) that ``cta`` reads without extra dependencies. Mark steps in your code with
``jax.profiler.StepTraceAnnotation("train", step_num=i)``; without markers CTA treats each jitted-module execution
as a step.
"""  # noqa: E501

from __future__ import annotations

import argparse
import pathlib
import runpy
import sys


def main(argv: list[str] | None = None) -> int:
    """Runs the target script or module under ``jax.profiler.trace``.

    The log directory is written even if the target raises an
    ``Exception`` or calls ``sys.exit``.

    Args:
        argv: Command-line arguments; None means ``sys.argv[1:]``.

    Returns:
        The target's exit status: its ``SystemExit`` code, 1 if it raised
        (or exited with a non-integer code), else 0.
    """
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--out", required=True, help="log directory to write")
    parser.add_argument(
        "--no-perfetto",
        action="store_true",
        help="write only the xplane.pb (needs xprof to analyze)",
    )
    parser.add_argument(
        "target",
        nargs=argparse.REMAINDER,
        help="script.py [args] or -m module [args]",
    )
    args = parser.parse_args(argv)
    target = args.target[1:] if args.target[:1] == ["--"] else args.target
    if not target:
        parser.error("nothing to run")

    import jax  # lazy: only needed when profiling

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    status = 0
    with jax.profiler.trace(
        str(out), create_perfetto_trace=not args.no_perfetto
    ):
        try:
            if target[0] == "-m":
                sys.argv = [target[1], *target[2:]]
                runpy.run_module(target[1], run_name="__main__", alter_sys=True)
            else:
                sys.argv = target
                runpy.run_path(target[0], run_name="__main__")
        except SystemExit as exc:
            status = exc.code if isinstance(exc.code, int) else 1
        except Exception as exc:  # keep the trace collected up to the crash
            print(
                f"cta profile: target raised {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            status = 1
    print(f"wrote {out}", file=sys.stderr)
    return status


if __name__ == "__main__":
    sys.exit(main())
