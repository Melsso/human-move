# chess-eval

Evaluates the rating tiers in two ways:

- Stockfish Elo calibration measures absolute strength.
- An inter-tier round robin measures relative ordering, with no Stockfish
  needed.

Both use the backend's `ModelRegistry` and `Player`, so a tier is evaluated
with exactly the model, search depth, weight and temperature the app uses.
Tiers are defined in `shared/chess_shared/tiers.py`: `1000`, `1500`, `2000`.

## Setup

Stockfish calibration needs a system Stockfish binary. Round robin doesn't.

```bash
brew install stockfish
apt install stockfish
dnf install stockfish
```

`STOCKFISH_PATH` overrides the lookup.

## Commands

```bash
make eval-stockfish TIER=1500 [STOCKFISH_ELOS=1320,1500,1700] [GAMES=100] [WORKERS=6]
make eval-round-robin TIERS=1000,1500,2000 [GAMES=100] [WORKERS=6]

uv run --package chess-eval pytest eval/tests/ -v
```

Results go to `eval/results/`.

### `chess_eval.stockfish` flags

| Flag | Default | Meaning |
|---|---|---|
| `tier` (positional) | — | Tier to benchmark |
| `--stockfish-elos` | `1320,1500,1700,1900,2100` | `UCI_Elo` levels to test against |
| `--games` | `100` | Games per level |
| `--move-time-ms` | `300` | Stockfish think time per move |
| `--threads` | `1` | Stockfish threads per game |
| `--workers` | CPU count | Parallel games (separate processes) |
| `--out` | none | Save results as JSON |

### `chess_eval.round_robin` flags

| Flag | Default | Meaning |
|---|---|---|
| `tiers` (positional) | — | Comma-separated tiers, e.g. `1000,1500,2000` |
| `--games` | `500` | Games per pairing |
| `--workers` | CPU count | Parallel games (separate processes) |
| `--out` | none | Save results as JSON |

## Design notes

- `chess_eval.stats` is pure Elo math with no chess dependencies.
- Every Elo number carries a Wilson-interval confidence bracket. At 100 games
  the bracket is typically ±150–250 Elo, so read the bracket and not just the
  point estimate.
- `UCI_Elo` is Stockfish's own weakening scale, not a human Lichess rating.
  Treat it as a relative benchmark.
- An illegal move from a model is a hard failure. A Stockfish communication
  error only excludes that game, and it is reported.
- Both scripts run games in spawned worker processes. Each worker is pinned to
  one torch thread and loads its tier's `Player` once. Searching tiers are
  slow (depth 6 can take seconds per move), so size `--games` and `--workers`
  accordingly. `--workers 1` in round robin runs in-process.
- Games are deterministic. The random opening and any temperature sampling are
  seeded per game, so rerunning a command reproduces the same games. The
  Stockfish JSON also records the tier config that was used.
- Unknown tier names fail immediately with the list of valid tiers.