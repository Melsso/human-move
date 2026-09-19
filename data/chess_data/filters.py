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


def game_average_elo(headers: chess.pgn.Headers) -> float | None:
    try:
        white_elo = int(headers.get("WhiteElo", ""))
        black_elo = int(headers.get("BlackElo", ""))
    except ValueError:
        return None
    return (white_elo + black_elo) / 2


def should_keep_game(
    headers: chess.pgn.Headers,
    min_elo: int,
    max_elo: int,
    max_elo_gap: int = 200,
) -> bool:
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


class FilteringGameBuilder(chess.pgn.GameBuilder):
    """
    A chess.pgn.GameBuilder that decides whether to keep a game as soon as
    its headers are parsed, instead of after the whole game (headers +
    movetext) has been parsed.

    chess.pgn.read_game() calls Visitor.end_headers() right after the header
    block is consumed, before it starts parsing any moves. If end_headers()
    returns chess.pgn.SKIP, the parser takes its fast path for the rest of
    the game: it scans the movetext with a cheap regex just to find the game
    boundary (for correctly skipping over "}"/"{" in comments), without
    tokenizing moves, running SAN parsing, or pushing anything onto a board.
    That's exactly the expensive part we want to skip for games we're going
    to throw away anyway.

    If the game passes the filter, end_headers() returns normally and
    chess.pgn.GameBuilder takes over from there, building the full game tree
    exactly as it would with the default visitor.

    This has to happen inside a single read_game() call (rather than calling
    chess.pgn.read_headers() and then, if wanted, chess.pgn.read_game() on
    the same handle) because read_headers() already consumes and discards
    the movetext of the game whose headers it just returned, and getting
    back to reparse it would require seeking backwards -- which a streamed
    .zst decompressor generally can't do.
    """

    def __init__(self, min_elo: int, max_elo: int, max_elo_gap: int = 200) -> None:
        super().__init__()
        self.min_elo = min_elo
        self.max_elo = max_elo
        self.max_elo_gap = max_elo_gap

    def end_headers(self) -> chess.pgn.SkipType | None:
        if not should_keep_game(
            self.game.headers, self.min_elo, self.max_elo, self.max_elo_gap
        ):
            return chess.pgn.SKIP
        return super().end_headers()
