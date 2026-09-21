import io

import chess.pgn
from chess_data.brackets import EloBracket
from chess_data.filters import (
    MultiBucketGameBuilder,
    game_average_elo,
    should_keep_game,
)

RATED_GOOD_GAME = """\
[Event "Rated Blitz game"]
[White "playerA"]
[Black "playerB"]
[WhiteElo "1050"]
[BlackElo "980"]
[Termination "Normal"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 1-0
"""

CASUAL_GAME = """\
[Event "Casual Blitz game"]
[White "playerA"]
[Black "playerB"]
[WhiteElo "1050"]
[BlackElo "980"]
[Termination "Normal"]

1. e4 e5 1-0
"""

BOT_GAME = """\
[Event "Rated Blitz game"]
[White "playerA"]
[Black "BotBlack"]
[WhiteElo "1050"]
[BlackElo "1500"]
[BlackTitle "BOT"]
[Termination "Normal"]

1. e4 e5 1-0
"""

MISMATCHED_ELO_GAME = """\
[Event "Rated Blitz game"]
[White "playerA"]
[Black "playerB"]
[WhiteElo "1050"]
[BlackElo "2200"]
[Termination "Normal"]

1. e4 e5 1-0
"""

ABANDONED_GAME = """\
[Event "Rated Blitz game"]
[White "playerA"]
[Black "playerB"]
[WhiteElo "1050"]
[BlackElo "980"]
[Termination "Abandoned"]

1. e4 1-0
"""


def _parse(pgn_text: str) -> chess.pgn.Game:
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    assert game is not None
    return game


def _parse_headers_only(pgn_text: str) -> chess.pgn.Headers:
    headers = chess.pgn.read_headers(io.StringIO(pgn_text))
    assert headers is not None
    return headers


def test_average_elo() -> None:
    game = _parse(RATED_GOOD_GAME)
    assert game_average_elo(game.headers) == 1015.0


def test_keeps_rated_game_in_range() -> None:
    game = _parse(RATED_GOOD_GAME)
    assert should_keep_game(game.headers, min_elo=900, max_elo=1100) is True


def test_rejects_casual_game() -> None:
    game = _parse(CASUAL_GAME)
    assert should_keep_game(game.headers, min_elo=900, max_elo=1100) is False


def test_rejects_bot_game() -> None:
    game = _parse(BOT_GAME)
    assert should_keep_game(game.headers, min_elo=900, max_elo=1600) is False


def test_rejects_mismatched_elo_game() -> None:
    game = _parse(MISMATCHED_ELO_GAME)
    assert should_keep_game(game.headers, min_elo=900, max_elo=2300) is False


def test_rejects_abandoned_game() -> None:
    game = _parse(ABANDONED_GAME)
    assert should_keep_game(game.headers, min_elo=900, max_elo=1100) is False


def test_rejects_out_of_range_elo() -> None:
    game = _parse(RATED_GOOD_GAME)
    assert should_keep_game(game.headers, min_elo=1800, max_elo=2000) is False


def test_headers_only_and_full_parse_agree_on_every_sample_game() -> None:
    samples = [
        RATED_GOOD_GAME,
        CASUAL_GAME,
        BOT_GAME,
        MISMATCHED_ELO_GAME,
        ABANDONED_GAME,
    ]
    for pgn_text in samples:
        full_game = _parse(pgn_text)
        headers_only = _parse_headers_only(pgn_text)
        for min_elo, max_elo in [(900, 1100), (900, 1600), (900, 2300), (1800, 2000)]:
            full_result = should_keep_game(full_game.headers, min_elo, max_elo)
            fast_result = should_keep_game(headers_only, min_elo, max_elo)
            assert full_result == fast_result, (
                f"disagreement for min_elo={min_elo}, max_elo={max_elo}: "
                f"full={full_result}, headers_only={fast_result}"
            )


BRACKET_1000 = EloBracket("1000", 900, 1100)
BRACKET_1500 = EloBracket("1500", 1400, 1600)
BRACKET_2000 = EloBracket("2000", 1900, 2100)


def test_multi_bucket_game_builder_matches_the_correct_bracket() -> None:
    builder = MultiBucketGameBuilder([BRACKET_1000, BRACKET_1500, BRACKET_2000])
    game = chess.pgn.read_game(io.StringIO(RATED_GOOD_GAME), Visitor=lambda: builder)
    assert game is not None
    assert builder.matched_bracket == BRACKET_1000


def test_multi_bucket_game_builder_skips_when_no_bracket_matches() -> None:
    builder = MultiBucketGameBuilder([BRACKET_1500, BRACKET_2000])
    game = chess.pgn.read_game(io.StringIO(RATED_GOOD_GAME), Visitor=lambda: builder)
    assert game is not None
    assert list(game.mainline_moves()) == []
    assert builder.matched_bracket is None


def test_multi_bucket_game_builder_keeps_movetext_for_matched_bracket() -> None:
    builder = MultiBucketGameBuilder([BRACKET_1000, BRACKET_1500, BRACKET_2000])
    game = chess.pgn.read_game(io.StringIO(RATED_GOOD_GAME), Visitor=lambda: builder)
    assert game is not None
    moves = list(game.mainline_moves())
    assert len(moves) == 6
    assert builder.matched_bracket == BRACKET_1000


def test_multi_bucket_game_builder_rejects_bot_and_casual_regardless_of_brackets() -> (
    None
):
    for pgn_text in [BOT_GAME, CASUAL_GAME, ABANDONED_GAME]:
        builder = MultiBucketGameBuilder([BRACKET_1000, BRACKET_1500, BRACKET_2000])
        game = chess.pgn.read_game(io.StringIO(pgn_text), Visitor=lambda b=builder: b)
        assert game is not None
        assert builder.matched_bracket is None
        assert list(game.mainline_moves()) == []
