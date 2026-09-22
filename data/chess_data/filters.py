from __future__ import annotations

import chess.pgn

from chess_data.brackets import EloBracket


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
    max_elo_gap: int = 100,
) -> bool:
    if headers.get("Variant", "Standard") != "Standard":
        return False

    event = headers.get("Event", "")
    if "Rated Rapid" not in event:
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


class MultiBucketGameBuilder(chess.pgn.GameBuilder):
    def __init__(self, brackets: list[EloBracket]) -> None:
        super().__init__()
        self.brackets = brackets
        self.matched_bracket: EloBracket | None = None

    def end_headers(self) -> chess.pgn.SkipType | None:
        for bracket in self.brackets:
            if should_keep_game(
                self.game.headers, bracket.min_elo, bracket.max_elo, bracket.max_elo_gap
            ):
                self.matched_bracket = bracket
                return super().end_headers()
        return chess.pgn.SKIP
