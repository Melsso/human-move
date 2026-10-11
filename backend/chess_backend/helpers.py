import chess
from fastapi import HTTPException


def build_board(fen: str, history: list[str]) -> chess.Board:
    try:
        target = chess.Board(fen)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"invalid FEN: {e}") from e
    if not history:
        return target

    board = chess.Board()
    try:
        for uci in history:
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves:
                raise ValueError(f"illegal move {uci}")
            board.push(move)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"invalid history: {e}") from e

    if board.epd() != target.epd():
        raise HTTPException(status_code=400, detail="history does not match fen")
    return board
