from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EloBracket:
    name: str
    min_elo: int
    max_elo: int
    max_elo_gap: int = 200


DEFAULT_BRACKETS: tuple[EloBracket, ...] = (
    EloBracket("1000", 1000, 1300),
    EloBracket("1500", 1600, 1900),
    EloBracket("2000", 2200, 2500),
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
