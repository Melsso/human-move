from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class MatchResult:
    wins: int
    draws: int
    losses: int

    @property
    def games(self) -> int:
        return self.wins + self.draws + self.losses

    @property
    def score(self) -> float:
        if self.games == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.games


def elo_diff_from_score(score: float) -> float:
    score = max(0.001, min(0.999, score))
    return 400 * math.log10(score / (1 - score))


def wilson_score_interval(p_hat: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half_width = (z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2))) / denom
    return max(0.0, center - half_width), min(1.0, center + half_width)


@dataclass(frozen=True)
class EloEstimate:
    point: float
    low: float
    high: float

    def __str__(self) -> str:
        return f"{self.point:+.0f} [{self.low:+.0f}, {self.high:+.0f}]"


def elo_estimate(result: MatchResult, z: float = 1.96) -> EloEstimate:
    low_score, high_score = wilson_score_interval(result.score, result.games, z)
    return EloEstimate(
        point=elo_diff_from_score(result.score),
        low=elo_diff_from_score(low_score),
        high=elo_diff_from_score(high_score),
    )
