r"""Runs a script or module under ``torch.profiler`` and writes a Kineto trace.

The trace is a Chrome-format ``.json`` or ``.json.gz`` file.

    python -m cta.profile.kineto_runner --out trace.json.gz \
        [--step-on optimizer|none] -- script.py [args...]
    torchrun --nproc_per_node=2 -m cta.profile.kineto_runner \
        --out dir/trace_rank{rank}.json.gz -- train.py   # one trace per rank
    python -m cta.profile.kineto_runner --out trace.json.gz \
        -- -m package.module [args...]

With ``--step-on optimizer`` every ``optimizer.step()`` advances the
profiler, producing ``ProfilerStep#N`` ranges that CTA uses for warm-up
slicing and per-step statistics. Without step markers the whole run is one
step.
"""

from __future__ import annotations

import argparse
import inspect
import os
import pathlib
import sys

from cta.profile import runners


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
    with profiler.profile(activities=activities, **kwargs) as prof:  # type: ignore[arg-type]
        if args.step_on == "optimizer":
            handle = optimizer.register_optimizer_step_post_hook(
                lambda *_a, **_k: prof.step()
            )
        try:
            status = runners.execute_target(target)
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
