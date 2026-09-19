"""
Run with: uv run --package chess-backend uvicorn chess_backend.main:app --reload
(or `make serve`, which does exactly that).

Serves the API under /api/* AND the static frontend at / from the same
process on the same port -- deliberately avoiding a separate frontend dev
server, so there's no CORS configuration to get wrong for a local-only
project like this.
"""

from __future__ import annotations

from pathlib import Path

import chess
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles

from chess_backend.inference import ModelRegistry, TierNotFoundError, compute_model_move
from chess_backend.schemas import MoveCandidate, MoveRequest, MoveResponse, TierInfo

app = FastAPI(title="chess-ai backend")
registry = ModelRegistry()

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/tiers", response_model=list[TierInfo])
def get_tiers() -> list[TierInfo]:
    tiers = registry.discover_tiers()
    result = []
    for name in tiers:
        meta = registry.get_meta(name)
        result.append(TierInfo(tier=name, **meta))
    return result


@app.post("/api/move", response_model=MoveResponse)
def make_move(req: MoveRequest) -> MoveResponse:
    try:
        board = chess.Board(req.fen)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"invalid FEN: {e}") from e

    try:
        human_move = chess.Move.from_uci(req.move_uci)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"invalid UCI move: {e}") from e

    if human_move not in board.legal_moves:
        raise HTTPException(status_code=400, detail="illegal move for this position")

    human_move_san = board.san(human_move)
    board.push(human_move)
    fen_after_human = board.fen()

    if board.is_game_over():
        return MoveResponse(
            human_move_san=human_move_san,
            fen_after_human=fen_after_human,
            model_move_uci=None,
            model_move_san=None,
            fen_after_model=fen_after_human,
            game_over=True,
            result=board.result(),
            top_candidates=[],
        )

    try:
        model = registry.get_model(req.tier)
    except TierNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e

    model_move, candidates = compute_model_move(
        model, board, temperature=req.temperature
    )
    model_move_san = board.san(model_move)
    board.push(model_move)
    fen_after_model = board.fen()

    return MoveResponse(
        human_move_san=human_move_san,
        fen_after_human=fen_after_human,
        model_move_uci=model_move.uci(),
        model_move_san=model_move_san,
        fen_after_model=fen_after_model,
        game_over=board.is_game_over(),
        result=board.result() if board.is_game_over() else None,
        top_candidates=[MoveCandidate(**c) for c in candidates],
    )


if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
