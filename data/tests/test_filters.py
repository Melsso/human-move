import io

import chess.pgn
from chess_data.filters import game_average_elo, should_keep_game

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


def test_average_elo() -> None:
    game = _parse(RATED_GOOD_GAME)
    assert game_average_elo(game) == 1015.0


def test_keeps_rated_game_in_range() -> None:
    game = _parse(RATED_GOOD_GAME)
    assert should_keep_game(game, min_elo=900, max_elo=1100) is True


def test_rejects_casual_game() -> None:
    game = _parse(CASUAL_GAME)
    assert should_keep_game(game, min_elo=900, max_elo=1100) is False


def test_rejects_bot_game() -> None:
    game = _parse(BOT_GAME)
    assert should_keep_game(game, min_elo=900, max_elo=1600) is False


def test_rejects_mismatched_elo_game() -> None:
    game = _parse(MISMATCHED_ELO_GAME)
    assert should_keep_game(game, min_elo=900, max_elo=2300) is False


def test_rejects_abandoned_game() -> None:
    game = _parse(ABANDONED_GAME)
    assert should_keep_game(game, min_elo=900, max_elo=1100) is False


def test_rejects_out_of_range_elo() -> None:
    game = _parse(RATED_GOOD_GAME)
    assert should_keep_game(game, min_elo=1800, max_elo=2000) is False
