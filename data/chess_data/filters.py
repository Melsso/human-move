"""
Decides which games from a raw Lichess PGN dump are worth training on.

Lichess dumps contain everything: bullet games, bot games, casual (unrated)
games, aborted games, variant games (Chess960, King of the Hill, ...). Most
of that is noise for "learn what a human at rating X plays" -- bot games in
particular would teach the model to imitate an engine, not a human, which
defeats the entire point.
"""

from __future__ import annotations

import chess.pgn


def game_average_elo(game: chess.pgn.Game) -> float | None:
    try:
        white_elo = int(game.headers.get("WhiteElo", ""))
        black_elo = int(game.headers.get("BlackElo", ""))
    except ValueError:
        return None
    return (white_elo + black_elo) / 2


def should_keep_game(
    game: chess.pgn.Game,
    min_elo: int,
    max_elo: int,
    max_elo_gap: int = 200,
) -> bool:
    headers = game.headers

    if headers.get("Variant", "Standard") != "Standard":
        return False

    event = headers.get("Event", "")
    if "Rated" not in event:
        return False

    if headers.get("WhiteTitle") == "BOT" or headers.get("BlackTitle") == "BOT":
        return False

    termination = headers.get("Termination", "")
    if termination in ("Abandoned", "Rules infraction"):
        return False

    try:
        white_elo = int(headers.get("WhiteElo", ""))
        black_elo = int(headers.get("BlackElo", ""))
    except ValueError:
        return False

    if abs(white_elo - black_elo) > max_elo_gap:
        return False

    avg = (white_elo + black_elo) / 2
    return min_elo <= avg <= max_elo
