"""Run a Python script or module under ``jax.profiler.trace`` and write an XProf log directory.

    python -m cta.profile.jax_runner --out logdir [--no-perfetto] -- script.py [args...]
    python -m cta.profile.jax_runner --out logdir -- -m package.module [args...]

The log directory holds ``plugins/profile/<run>/<host>.xplane.pb`` plus, by default, a Chrome-format trace
(``create_perfetto_trace=True``) that ``cta`` reads without extra dependencies. Mark steps in your code with
``jax.profiler.StepTraceAnnotation("train", step_num=i)``; without markers CTA treats each jitted-module execution
as a step.
"""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="log directory to write")
    ap.add_argument("--no-perfetto", action="store_true", help="write only the xplane.pb (needs xprof to analyze)")
    ap.add_argument("target", nargs=argparse.REMAINDER, help="script.py [args] or -m module [args]")
    ns = ap.parse_args(argv)
    target = ns.target[1:] if ns.target[:1] == ["--"] else ns.target
    if not target:
        ap.error("nothing to run")

    import jax  # lazy: only needed when profiling

    out = Path(ns.out)
    out.mkdir(parents=True, exist_ok=True)
    status = 0
    with jax.profiler.trace(str(out), create_perfetto_trace=not ns.no_perfetto):
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
            print(f"cta profile: target raised {type(exc).__name__}: {exc}", file=sys.stderr)
            status = 1
    print(f"wrote {out}", file=sys.stderr)
    return status


if __name__ == "__main__":
    sys.exit(main())
