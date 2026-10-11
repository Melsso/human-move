# Changelog

All notable changes to this project are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.1.0] - 2026-10-11

First release.

### Added
- Three playable tiers (~1185, ~1600 and ~2010 measured on a Stockfish UCI_Elo scale, named 1000, 1500 and 2000). Tier 1000 is policy-only with temperature. Tiers 1500 and 2000 re-rank the policy’s top moves with a policy-guided alpha-beta search on the model trained on ~2000-rated games, with depth 3 and depth 6 respectively.
- Shared package with board and move encoding, masked inference, tier-aware move selection and a hand-written search (iterative deepening, transposition table, killer/history ordering, null-move pruning, late-move reductions, check extensions, quiescence with delta pruning, repetition detection and a static evaluation).
- Data pipeline: chunked, single-pass conversion of Lichess PGN dumps (`.pgn` / `.pgn.zst`) into rating-bucketed training tensors. Only rated Rapid, standard-variant, human games are kept.
- Training: residual policy network, game-level train/validation split, chunked and resumable training, self-describing checkpoints that reject an outdated encoding.
- Backend: FastAPI service with `/api/tiers`, `/api/move`, `/api/eval` and a static board UI. `/api/move` accepts an optional move history and an optional human move (so the model can move first).
- Board UI with move list and review navigation, candidate-move display, optional Stockfish eval bar and a game-over dialog.
- Evaluation: Stockfish Elo calibration and an inter-tier round robin, both with Wilson confidence intervals and per-game seeding.
- Board UI: play as either colour, promotion picker, take back, resign, last-move and legal-move highlights, fully vendored assets.” and “CHECKPOINTS_DIR and make fetch-weights so serving doesn’t depend on the training layout.
- `uv` workspace, `make` targets, strict mypy, ruff and test suites for every package.
- GitHub Actions: CI and tag-based GitHub releases.
- MIT license.