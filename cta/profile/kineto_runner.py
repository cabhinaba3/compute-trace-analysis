"""Run a Python script or module under ``torch.profiler`` and write a Chrome/Kineto trace (``.json`` / ``.json.gz``).

    python -m cta.profile.kineto_runner --out trace.json.gz [--step-on optimizer|none] -- script.py [args...]
    torchrun --nproc_per_node=2 -m cta.profile.kineto_runner --out dir/trace_rank{rank}.json.gz -- train.py   # 1 trace/rank
    python -m cta.profile.kineto_runner --out trace.json.gz -- -m package.module [args...]

With ``--step-on optimizer`` every ``optimizer.step()`` advances the profiler, producing ``ProfilerStep#N`` ranges that
CTA uses for warmup slicing and per-step statistics. Without step markers the whole run is one step.
"""  # noqa: E501

from __future__ import annotations

import argparse
import inspect
import os
import pathlib
import runpy
import sys


def main(argv: list[str] | None = None) -> int:
    """Runs the target script or module under ``torch.profiler``.

    The trace is written even if the target raises an ``Exception`` or
    calls ``sys.exit``.

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
    parser.add_argument(
        "--out",
        required=True,
        help="trace file to write (.json or .json.gz); {rank} expands to $RANK",
    )
    parser.add_argument(
        "--step-on", choices=("optimizer", "none"), default="optimizer"
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

    import torch  # lazy: only needed when profiling
    from torch import profiler
    from torch.optim import optimizer

    activities = [profiler.ProfilerActivity.CPU]
    if torch.cuda.is_available():
        activities.append(profiler.ProfilerActivity.CUDA)

    # torch only emits ProfilerStep#N ranges when a schedule is given, so use
    # one that records every step forever.
    kwargs: dict[str, object] = {}
    if args.step_on == "optimizer":
        kwargs["schedule"] = lambda _step: profiler.ProfilerAction.RECORD
        # Keep events across steps (torch >= 2.2).
        if "acc_events" in inspect.signature(profiler.profile).parameters:
            kwargs["acc_events"] = True

    handle = None
    status = 0
    with profiler.profile(activities=activities, **kwargs) as prof:  # type: ignore[arg-type]
        if args.step_on == "optimizer":
            handle = optimizer.register_optimizer_step_post_hook(
                lambda *_a, **_k: prof.step()
            )
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
        finally:
            if handle is not None:
                handle.remove()
    out = pathlib.Path(args.out.replace("{rank}", os.environ.get("RANK", "0")))
    out.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(out))
    print(f"wrote {out}", file=sys.stderr)
    return status


if __name__ == "__main__":
    sys.exit(main())
