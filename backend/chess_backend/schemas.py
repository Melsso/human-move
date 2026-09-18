from __future__ import annotations

from pydantic import BaseModel


class TierInfo(BaseModel):
    tier: str
    epoch: int
    val_top1: float
    val_top3: float


class MoveCandidate(BaseModel):
    uci: str
    san: str
    prob: float


class MoveRequest(BaseModel):
    fen: str
    move_uci: str
    tier: str
    temperature: float = 0.0


class MoveResponse(BaseModel):
    human_move_san: str
    fen_after_human: str
    model_move_uci: str | None
    model_move_san: str | None
    fen_after_model: str
    game_over: bool
    result: str | None
    top_candidates: list[MoveCandidate]
