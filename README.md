# human-move

A from-scratch chess engine with three human-like playing strengths. Policy networks are trained on rated Rapid games from Lichess, bucketed by rating. At play time each tier adds a temperature and, for the two stronger tiers, a small policy-guided alpha-beta search. Fully local: no external services.

Inspired by [Maia Chess](https://www.maiachess.com/) (McIlroy-Young et al., KDD 2020), which trains networks to predict human moves at specific ratings. This is an independent implementation trained on a much smaller, Rapid-only dataset, with a hand-written search layer and a tier system added on top.

<!-- demo GIF goes here -->

## Results

Strength is measured against Stockfish at fixed `UCI_Elo` levels (100 games per level, 1 s per Stockfish move, random 8-ply openings, colours alternated) and in an inter-tier round robin (100 games per pairing). Each tier is tested at levels around its strength, and its headline figure is the mean implied Elo of the levels where it scored between 35% and 65%. Intervals are 95% Wilson intervals, about ±70 Elo at this game count.

| Tier | Measured strength | How it plays |
|---|---|---|
| 1000 | ~1185 | Policy only, temperature 0.3 |
| 1500 | ~1600 | Depth-3 search on the 2000-bucket model |
| 2000 | ~2010 | Same model, depth-6 search, search weight 4 |

| Tier | vs Stockfish | W / D / L | Score | Implied Elo (95% CI) |
|---|---|---|---|---|
| 1000 | 1320 | 23 / 15 / 62 | 30.5% | 1177 (1104 to 1250) |
| 1000 | 1500 | 8 / 13 / 79 | 14.5% | 1192 (1096 to 1287) |
| 1500 | 1320 | 77 / 8 / 15 | 81.0% | 1572 (1486 to 1658) |
| 1500 | 1500 | 51 / 12 / 37 | 57.0% | 1549 (1481 to 1617) |
| 1500 | 1700 | 32 / 20 / 48 | 42.0% | 1644 (1575 to 1712) |
| 2000 | 1700 | 65 / 9 / 26 | 69.5% | 1843 (1770 to 1916) |
| 2000 | 1900 | 49 / 17 / 34 | 57.5% | 1953 (1884 to 2021) |
| 2000 | 2100 | 37 / 17 / 46 | 45.5% | 2069 (2001 to 2137) |

Round robin (score of the first tier): 1000 vs 1500 scored 4%, 1500 vs 2000 scored 12.5%, and 1000 vs 2000 scored 1%. Raw data is in `eval/results/`.

These numbers are on Stockfish's `UCI_Elo` scale. They are not Lichess ratings and should be read as a relative benchmark. Tier names are nominal. Stockfish's `UCI_Elo` floor is 1320, so tier 1000 can only be benchmarked from above.

## Layout

```
human-move/
├── shared/     # encodings, masked inference, search, tier config, move selection
├── data/       # Lichess PGN -> rating-bucketed training tensors
├── training/   # policy network + training loop
├── backend/    # FastAPI inference API + static board UI (backend/static/)
├── eval/       # Stockfish calibration + inter-tier round robin
└── scripts/    # train_all_chunks.sh, vendor_static.sh
```

`shared` is the single source of truth for board and move encoding and for how a tier picks a move. Training, serving and evaluation all use it, so they can't drift apart. It has no torch dependency.

## Tiers

Defined in `shared/chess_shared/tiers.py`:

| Tier | Checkpoint | Temperature | Search |
|---|---|---|---|
| 1000 | bucket 1000 | 0.3 | none |
| 1500 | bucket 2000 | 0 | depth 3, weight 1 |
| 2000 | bucket 2000 | 0 | depth 6, weight 4 |

Bucket names are the average rating of their training games (`1000`: 900 to 1100, `2000`: 1900 to 2100). Tier names are the target playing strength. There is no separate model per tier: high-rated Rapid data was too sparse and noisy to train several models, so tiers 1500 and 2000 reuse the model trained on ~2000-rated games and differ only in search. The search re-scores the network's top 6 moves as `log(policy_prob) + weight * eval_in_pawns` (details in [`shared/README.md`](shared/README.md)).

## Training data

Only rated Rapid, standard-variant games between human players are used (no bots, no abandoned or rules-infraction games, both players within 200 Elo).
The models were trained on roughly half of one monthly Lichess dump, which is small compared with the original Maia work. Lichess game data is [CC0](https://database.lichess.org/).

## Quickstart

A `uv` workspace with one lockfile.

```bash
uv sync --all-packages
make fetch-weights                  # downloads the two trained checkpoints
CHECKPOINTS_DIR=weights make serve
```

Open http://127.0.0.1:8000, pick an opponent and play. The optional eval bar needs a system Stockfish (`brew install stockfish` or `apt install stockfish`).

Use `make <target>` instead of raw `uv` commands. Every target syncs with `--all-packages` first, because a bare `uv sync` uninstalls the workspace members. If you hit `ModuleNotFoundError: No module named 'chess_X'`, run `make reset`. After editing any `pyproject.toml`, run `uv lock`.

## Training your own

```bash
make count-games SOURCE=data/downloads/lichess_db_standard_rated_2024-06.pgn.zst
make prepare SOURCE=data/downloads/lichess_db_standard_rated_2024-06.pgn.zst CHUNK_SIZE=700000
make train
make serve
make eval-round-robin TIERS=1000,1500,2000 GAMES=100
make eval-stockfish TIER=1500 STOCKFISH_ELOS=1320,1500,1700 GAMES=100
make check
```

- `make prepare` reads the source once and builds every bucket in chunks
  (`data/processed/bucket_<name>_<chunk>.npz`).
- `make train` trains each bucket through its chunks in order, resuming each from the previous one. It writes `training/checkpoints/bucket_<name>_<chunk>/`. Chunks that already have a `best.pt` are skipped.
- `make serve` uses the highest chunk per bucket from `training/checkpoints`, or from `CHECKPOINTS_DIR` if set.

### Makefile variables

| Variable | Used by | Default | Meaning |
|---|---|---|---|
| `SOURCE` | `count-games`, `prepare` | required | `.pgn` / `.pgn.zst` dump |
| `CHUNK_SIZE` | `prepare` | required | Games scanned per chunk |
| `BUCKETS` | `prepare`, `train` | all | e.g. `1000,2000` |
| `MAX_CHUNKS` | `prepare` | all | Stop after N chunks |
| `EPOCHS` | `train` | `10` | Epochs per chunk |
| `TIER` | `eval-stockfish` | required | Tier to benchmark |
| `TIERS` | `eval-round-robin` | required | Comma-separated tiers |
| `STOCKFISH_ELOS` | `eval-stockfish` | package default | `UCI_Elo` levels |
| `GAMES` | eval targets | package default | Games per level or pairing |
| `WORKERS` | eval targets | CPU count | Parallel worker processes |
| `WEIGHTS_TAG` | `fetch-weights` | `v0.1.0` | Release to download weights from |

`make check` runs the format check, ruff, strict mypy and all tests. `make test`, `make lint`, `make format` and `make typecheck` run them individually.

## Package docs

[`shared`](shared/README.md), [`data`](data/README.md),
[`training`](training/README.md), [`backend`](backend/README.md),
[`eval`](eval/README.md).

## Credits

Approach inspired by Maia Chess. Board UI uses chess.js, chessboard.js and jQuery (MIT) and the Wikipedia piece set (CC BY-SA), all vendored in `backend/static/vendor/`.

## License

MIT, see [LICENSE](./LICENSE). Training data comes from the [Lichess open database](https://database.lichess.org/) (CC0).