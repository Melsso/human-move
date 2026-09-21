import math

import pytest
from chess_eval.stats import (
    EloEstimate,
    MatchResult,
    elo_diff_from_score,
    elo_estimate,
    wilson_score_interval,
)


def test_match_result_score_all_wins() -> None:
    result = MatchResult(wins=10, draws=0, losses=0)
    assert result.score == 1.0
    assert result.games == 10


def test_match_result_score_all_losses() -> None:
    result = MatchResult(wins=0, draws=0, losses=10)
    assert result.score == 0.0


def test_match_result_score_draws_count_half() -> None:
    result = MatchResult(wins=0, draws=10, losses=0)
    assert result.score == 0.5


def test_match_result_score_mixed() -> None:
    result = MatchResult(wins=5, draws=2, losses=3)
    assert result.score == pytest.approx(0.6)


def test_match_result_zero_games_does_not_divide_by_zero() -> None:
    result = MatchResult(wins=0, draws=0, losses=0)
    assert result.games == 0
    assert result.score == 0.0


def test_elo_diff_from_score_at_50_percent_is_zero() -> None:
    assert elo_diff_from_score(0.5) == pytest.approx(0.0, abs=1e-9)


def test_elo_diff_from_score_is_positive_above_50_percent() -> None:
    assert elo_diff_from_score(0.75) > 0


def test_elo_diff_from_score_is_negative_below_50_percent() -> None:
    assert elo_diff_from_score(0.25) < 0


def test_elo_diff_from_score_is_symmetric() -> None:
    assert elo_diff_from_score(0.75) == pytest.approx(-elo_diff_from_score(0.25))


def test_elo_diff_from_score_known_value() -> None:
    assert elo_diff_from_score(0.64) == pytest.approx(100, abs=2)


def test_elo_diff_from_score_does_not_raise_at_exactly_0_or_1() -> None:
    low = elo_diff_from_score(0.0)
    high = elo_diff_from_score(1.0)
    assert math.isfinite(low)
    assert math.isfinite(high)
    assert low < 0
    assert high > 0


def test_wilson_interval_contains_point_estimate() -> None:
    low, high = wilson_score_interval(0.6, 100)
    assert low <= 0.6 <= high


def test_wilson_interval_narrows_with_more_games() -> None:
    low_small, high_small = wilson_score_interval(0.6, 20)
    low_large, high_large = wilson_score_interval(0.6, 2000)
    assert (high_large - low_large) < (high_small - low_small)


def test_wilson_interval_at_zero_games_is_maximally_wide() -> None:
    low, high = wilson_score_interval(0.5, 0)
    assert low == 0.0
    assert high == 1.0


def test_wilson_interval_stays_within_0_and_1() -> None:
    low, high = wilson_score_interval(0.99, 5)
    assert 0.0 <= low
    assert high <= 1.0


def test_elo_estimate_returns_point_within_interval() -> None:
    result = MatchResult(wins=60, draws=10, losses=30)
    estimate = elo_estimate(result)
    assert estimate.low <= estimate.point <= estimate.high


def test_elo_estimate_interval_widens_with_fewer_games() -> None:
    small_sample = elo_estimate(MatchResult(wins=6, draws=1, losses=3))
    large_sample = elo_estimate(MatchResult(wins=600, draws=100, losses=300))
    assert (large_sample.high - large_sample.low) < (
        small_sample.high - small_sample.low
    )


def test_elo_estimate_str_format() -> None:
    estimate = EloEstimate(point=42.0, low=-10.0, high=94.0)
    assert str(estimate) == "+42 [-10, +94]"
