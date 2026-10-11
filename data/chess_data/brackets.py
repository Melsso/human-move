from __future__ import annotations

from dataclasses import dataclass

# bucket names are the average rating of the training games
DEFAULT_MAX_ELO_GAP = 200


@dataclass(frozen=True)
class EloBracket:
    name: str
    min_elo: int
    max_elo: int
    max_elo_gap: int = DEFAULT_MAX_ELO_GAP


DEFAULT_BRACKETS: tuple[EloBracket, ...] = (
    EloBracket("1000", 900, 1100),
    EloBracket("2000", 1900, 2100),
)


def brackets_by_name(names: list[str] | None) -> list[EloBracket]:
    if names is None:
        return list(DEFAULT_BRACKETS)

    by_name = {b.name: b for b in DEFAULT_BRACKETS}
    unknown = [n for n in names if n not in by_name]
    if unknown:
        raise ValueError(
            f"unknown bucket name(s) {unknown}, available: {sorted(by_name)}"
        )
    return [b for b in DEFAULT_BRACKETS if b.name in names]
