"""Dataclasses for grid runner: Template and GridItem (one row of the grid)."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from .client import SimulationSettings


@dataclass(frozen=True)
class Template:
    """Alpha expression template with a single {f} placeholder for a data-field id.

    Decay and truncation are template-level — different templates want different defaults
    (e.g. a momentum signal vs a mean-reversion signal).
    """

    name: str
    expression: str  # contains "{f}"
    decay: int = 4
    truncation: float = 0.08

    def render(self, field_id: str) -> str:
        return self.expression.format(f=field_id)


@dataclass(frozen=True)
class GridItem:
    """One concrete simulation cell: a rendered expression + full settings.

    Field id is kept separately so we can record it in the DB without re-parsing the expression.
    """

    template_name: str
    field_id: str
    expression: str
    settings: SimulationSettings

    def expression_hash(self) -> str:
        return hashlib.sha1(self.expression.encode("utf-8")).hexdigest()

    def dedup_key(self) -> tuple[str, str, str, int, int, str, float]:
        """Tuple uniquely identifying this sim — matches the DB UNIQUE index."""
        s = self.settings
        return (
            self.expression_hash(),
            s.region,
            s.universe,
            s.delay,
            s.decay,
            s.neutralization,
            s.truncation,
        )
