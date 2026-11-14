"""Synthetic 3D-parallel training graphs (GOAL text, GOAL binary, Chakra)."""

from __future__ import annotations

import dataclasses
import pathlib

from cta import _native as native


@dataclasses.dataclass(frozen=True, slots=True)
class TrainingSpec:
    """One training job on ``dp * pp * tp`` ranks.

    The schedule is GPipe-style: all microbatch forwards, then all backwards in
    reverse order. Rank ``r`` sits at data-parallel index ``d``, pipeline stage
    ``p`` and tensor-parallel index ``t`` with ``r = (d * pp + p) * tp + t``.
    Each layer has a tensor-parallel all-reduce; every ``moe_every``-th layer
    also has two all-to-alls inside an expert group of ``ep_size`` consecutive
    data-parallel ranks. Gradients are all-reduced across the data-parallel
    group in ``dp_buckets`` buckets that overlap the last microbatch's
    backward, and an optimizer node ends each step.

    Attributes:
        dp: Data-parallel size.
        pp: Pipeline stages.
        tp: Tensor-parallel size.
        microbatches: Microbatches per step.
        layers_per_stage: Layers on each pipeline stage.
        steps: Training steps.
        moe_every: 0 for no MoE layers, else every N-th layer is one.
        ep_size: Expert-parallel group size; must divide ``dp``.
        dp_buckets: Gradient buckets per step (at most ``layers_per_stage``).
        layer_fwd_ns: Forward compute time of a layer.
        layer_bwd_ns: Backward compute time of a layer.
        optimizer_ns: Optimizer compute time per step.
        expert_ns: Expert compute time of an MoE layer.
        tp_bytes: Tensor-parallel all-reduce size.
        pp_bytes: Pipeline activation (and gradient) message size.
        dp_bytes: Total gradient bytes per step, split over the buckets.
        moe_bytes: MoE all-to-all size.
    """

    dp: int = 2
    pp: int = 2
    tp: int = 2
    microbatches: int = 4
    layers_per_stage: int = 4
    steps: int = 2
    moe_every: int = 0
    ep_size: int = 1
    dp_buckets: int = 2
    layer_fwd_ns: int = 1_000_000
    layer_bwd_ns: int = 2_000_000
    optimizer_ns: int = 500_000
    expert_ns: int = 800_000
    tp_bytes: int = 8 << 20
    pp_bytes: int = 16 << 20
    dp_bytes: int = 64 << 20
    moe_bytes: int = 8 << 20

    @property
    def ranks(self) -> int:
        """Number of ranks: ``dp * pp * tp``."""
        return self.dp * self.pp * self.tp

    def validate(self) -> None:
        """Checks that the spec can be generated.

        Raises:
            ValueError: A size is below 1, ``ep_size`` does not divide ``dp``,
                ``dp_buckets`` exceeds ``layers_per_stage``, or the job has
                too many steps for 31-bit message tags.
        """
        self.to_native().validate()

    def to_native(self) -> native.TrainingSpec:
        """Returns the C++ spec with the same values."""
        spec = native.TrainingSpec()
        for field in dataclasses.fields(self):
            setattr(spec, field.name, getattr(self, field.name))
        return spec


def write_goal_training(
    path: pathlib.Path, spec: TrainingSpec | None = None
) -> int:
    """Writes the schedule as a `.goal` text file.

    Collectives are spelled out as send/recv (ring all-reduce and pairwise
    all-to-all), so they create cross-rank edges.

    Args:
        path: Output file; its parent directory is created if missing.
        spec: The training job; None uses the default ``TrainingSpec()``.

    Returns:
        The number of nodes written.

    Raises:
        ValueError: The spec cannot be generated (for example ``ep_size`` does
            not divide ``dp``).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    return native.write_goal_training(
        str(path), (spec or TrainingSpec()).to_native()
    )


def write_goal_training_bin(
    path: pathlib.Path, spec: TrainingSpec | None = None
) -> int:
    """Writes the schedule of ``write_goal_training`` as a `.bin` file.

    Args:
        path: Output file; its parent directory is created if missing.
        spec: The training job; None uses the default ``TrainingSpec()``.

    Returns:
        The number of nodes written.

    Raises:
        ValueError: The spec cannot be generated.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    return native.write_goal_training_bin(
        str(path), (spec or TrainingSpec()).to_native()
    )


def write_chakra_training(
    out_dir: pathlib.Path,
    spec: TrainingSpec | None = None,
    *,
    threads: int = 0,
) -> int:
    """Writes one Chakra `.et` shard per rank.

    A collective is one ``COMM_COLL`` node; only pipeline traffic is
    send/recv, as in real Chakra traces.

    Args:
        out_dir: Directory for ``trace.rank<N>.et``; created if missing.
        spec: The training job; None uses the default ``TrainingSpec()``.
        threads: Worker threads; 0 uses the CPUs available to the process.

    Returns:
        The total number of nodes written over all ranks.

    Raises:
        ValueError: The spec cannot be generated.
    """
    return native.write_chakra_training(
        str(out_dir), (spec or TrainingSpec()).to_native(), threads
    )
