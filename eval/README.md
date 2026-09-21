# chess-eval

Evaluates trained rating-bucket models two complementary ways: Stockfish
Elo calibration (absolute strength) and inter-bucket round robin
(relative ordering, no Stockfish needed). See the root `README.md` for
where this fits alongside the held-out validation accuracy
`chess_training.train` already reports during training.

## Setup: Stockfish

Requires a **system-installed** Stockfish binary, not a Python package —
there's a `stockfish` package on PyPI, but it only wraps the UCI
protocol (same role `python-chess`'s `chess.engine` already plays here);
it doesn't bundle the actual engine.

```bash
brew install stockfish        # macOS
apt install stockfish         # Debian/Ubuntu
dnf install stockfish         # Fedora
```

Round-robin evaluation needs no external engine at all — your own
trained models just play each other directly.

## Commands

```bash
# Stockfish calibration -- absolute strength
uv run --package chess-eval python -m chess_eval.stockfish 1000 \
    --stockfish-elos 1320,1500,1700,1900,2100 --games 100 \
    --out eval/results/1000_stockfish.json
# or:
make eval-stockfish TIER=1000

# inter-bucket round robin -- relative ordering, no Stockfish needed
uv run --package chess-eval python -m chess_eval.round_robin 1000,1500,2000 \
    --games 40 --out eval/results/round_robin.json
# or:
make eval-round-robin TIERS=1000,1500,2000

# tests (only the Stockfish-playing parts need a real engine; everything
# else -- stats, seeding, round robin against synthetic checkpoints --
# runs without one)
uv run --package chess-eval pytest eval/tests/ -v
```

Tier names are whatever `chess_backend.inference.ModelRegistry` resolves
them to — the same grouped, most-trained-chunk-per-bucket names the
backend/frontend already use (see `backend/README.md`).

### `chess_eval.stockfish` flags

| Flag | Default | Meaning |
|---|---|---|
| `tier` (positional) | — | Tier to benchmark, e.g. `1000` |
| `--stockfish-elos` | `1320,1500,1700,1900,2100` | Comma-separated `UCI_Elo` levels to test against |
| `--games` | `100` | Games played per level |
| `--move-time-ms` | `50` | Stockfish's think time per move |
| `--threads` | `1` | Stockfish threads per game |
| `--workers` | CPU count | Parallel games (separate OS processes) |
| `--out` | none | Save results as JSON |

### `chess_eval.round_robin` flags

| Flag | Default | Meaning |
|---|---|---|
| `tiers` (positional) | — | Comma-separated tier names, e.g. `1000,1500,2000` |
| `--games` | `40` | Games played per pairing |
| `--out` | none | Save results as JSON |

## Design notes worth knowing

- **`chess_eval.stats` has zero chess/engine dependencies** — pure Elo
  math, fully unit tested without Stockfish or a trained checkpoint
  anywhere in sight. If you only read one file to understand the
  methodology, read this one.
- **Every Elo number comes with a confidence interval**, via a Wilson
  score interval on the win/draw/loss proportion, converted to Elo
  through the same logistic transform as the point estimate (not a
  separate, symmetric Elo-space standard error, which would misrepresent
  the asymmetry near strong/weak results). At 100 games the interval is
  commonly ±150–250 Elo wide — a 30–40 point gap between two runs is very
  likely just noise, not a real change. Read the bracket, not just the
  point estimate.
- **`UCI_Elo` is Stockfish's own internal weakening estimate, not
  verified against real human Lichess ratings.** Treat calibration
  results as a reproducible, relative benchmark — great for comparing
  your own buckets against each other or tracking whether a retrain
  helped — not as a literal claim about human-equivalent strength.
- **An illegal move from your model is a hard failure, on purpose.**
  `legal_move_mask` should make this impossible; if it happens, that's a
  real bug worth seeing immediately, not something to quietly exclude
  from the score. A genuine engine communication error (crashed process,
  broken pipe) is a different failure mode — those games are excluded
  from scoring and reported, but don't abort the rest of the run.
- **Stockfish games run in parallel worker processes; round-robin games
  run single-process.** Each Stockfish game needs its own engine
  subprocess and pays real wall-clock time per move, so parallelizing
  across `multiprocessing.Pool` workers matters; round-robin games are
  just two small neural-net forward passes per ply with no subprocess
  round-trip, fast enough serially that the added complexity wouldn't
  pay for itself.
- **Both scripts reload games deterministically** (a fixed seed per
  game), so rerunning the exact same command reproduces the exact same
  games — useful for isolating whether a result changed because of an
  actual model change versus random variance.