"""Declarative plot specs: a kind plus its options, from YAML or the CLI."""

from __future__ import annotations

from collections.abc import Mapping
import dataclasses
import pathlib
from typing import Any

import yaml

from cta.viz import registry


@dataclasses.dataclass(slots=True)
class PlotSpec:
    """A declarative plot: a kind, its options and an optional output name.

    Attributes:
        kind: Plot kind name or alias, as typed.
        options: The kind's options, flat as in YAML.
        name: Output file stem; None uses the kind.
    """

    kind: str
    options: dict[str, Any] = dataclasses.field(default_factory=dict)
    name: str | None = None

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> PlotSpec:
        """Builds a spec from one YAML entry and validates its options.

        Args:
            raw: A mapping with ``kind`` (default ``"custom"``), an optional
                ``name`` and the kind's options as the other keys.

        Returns:
            The plot spec.

        Raises:
            ValueError: If ``raw`` is not a mapping, the kind is unknown or an
                option is invalid.
        """
        if not isinstance(raw, Mapping):
            raise ValueError(
                f"A plot spec must be a mapping, got {type(raw).__name__}"
            )
        body = dict(raw)
        kind = str(body.pop("kind", "custom"))
        name = body.pop("name", None)
        spec = cls(kind=kind, options=body, name=str(name) if name else None)
        # Validate now, not at render time.
        registry.resolve_options(registry.get_kind(kind), spec.options)
        return spec

    @property
    def stem(self) -> str:
        """The output file stem: ``name``, else the kind as typed.

        Using the kind as typed means ``--kind tornado`` still writes
        ``tornado.<ext>``.
        """
        return self.name or self.kind


def load_specs(path: str | pathlib.Path) -> list[PlotSpec]:
    """Reads plot specs from a YAML file.

    Args:
        path: YAML file holding a list of specs or a mapping with a
            ``plots:`` list.

    Returns:
        The validated plot specs, in file order.

    Raises:
        ValueError: If the file holds no non-empty list of specs or a spec is
            invalid.
    """
    document = yaml.safe_load(pathlib.Path(path).read_text())
    raw_specs = (
        document.get("plots") if isinstance(document, dict) else document
    )
    if not isinstance(raw_specs, list) or not raw_specs:
        raise ValueError(
            f"{path}: expected a non-empty list of plot specs "
            "(or a 'plots:' list)"
        )
    return [PlotSpec.from_dict(raw_spec) for raw_spec in raw_specs]
