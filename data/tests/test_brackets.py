from itertools import pairwise

import pytest
from chess_data.brackets import DEFAULT_BRACKETS, EloBracket, brackets_by_name


def test_brackets_by_name_preserves_default_order_regardless_of_input_order() -> None:
    result = brackets_by_name(["2000", "1000"])
    assert [b.name for b in result] == ["1000", "2000"]


def test_brackets_by_name_none_returns_all_defaults() -> None:
    assert brackets_by_name(None) == list(DEFAULT_BRACKETS)


def test_brackets_by_name_single_name_returns_just_that_one() -> None:
    result = brackets_by_name(["1500"])
    assert [b.name for b in result] == ["1500"]


def test_brackets_by_name_raises_on_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown bucket"):
        brackets_by_name(["not_a_real_bucket"])


def test_brackets_by_name_raises_listing_all_unknown_names_at_once() -> None:
    with pytest.raises(ValueError) as exc_info:
        brackets_by_name(["1000", "bogus1", "bogus2"])
    assert "bogus1" in str(exc_info.value)
    assert "bogus2" in str(exc_info.value)


def test_default_brackets_names_are_unique() -> None:
    names = [b.name for b in DEFAULT_BRACKETS]
    assert len(names) == len(set(names))


def test_default_brackets_ranges_are_non_overlapping() -> None:
    sorted_brackets = sorted(DEFAULT_BRACKETS, key=lambda b: b.min_elo)
    for a, b in pairwise(sorted_brackets):
        assert a.max_elo < b.min_elo, f"{a} and {b} overlap"


def test_elo_bracket_is_frozen() -> None:
    bracket = EloBracket("1000", 900, 1100)
    with pytest.raises(AttributeError):
        bracket.min_elo = 800


def test_elo_bracket_equality_is_by_value() -> None:
    a = EloBracket("1000", 900, 1100)
    b = EloBracket("1000", 900, 1100)
    c = EloBracket("1000", 900, 1101)
    assert a == b
    assert a != c
