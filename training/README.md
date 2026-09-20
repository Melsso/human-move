# chess-training

Model definition, training loop, terminal play script. Trains a separate
policy network per rating bucket on the `.npz` files `chess_data.prepare`
produces — see the root `README.md`'s "Processing a full Lichess dump"
section for the actual end-to-end workflow (`make prepare` → `make
train`). This package's own CLIs below are the single-chunk building
blocks underneath those `make` targets; use them directly for finer
control (custom hyperparameters, resuming from a specific checkpoint,
etc).

## What's here

- **`model.py`** — `MaiaPolicyNet`: a 6-block, 64-filter residual CNN,
  matching the architecture from the original Maia Chess paper
  (McIlroy-Young et al., 2020), adapted to `chess_shared`'s simpler
  4096-way move space instead of Leela Chess Zero's 1858-way encoding.
- **`dataset.py`** — `make_train_val_split(npz_path, val_fraction=0.05)`:
  loads a bucket `.npz`, shares the underlying array between train/val
  splits (no duplicate RAM use), and warns if the file couldn't actually
  be memory-mapped (`.npz` never can be, compressed or not — see the
  module docstring).
- **`train.py`** — the training loop, with MPS/CUDA/CPU auto-detection
  and `--resume-from` for chunked/incremental training.
- **`play.py`** — play a game against a checkpoint right in the terminal.

## Commands

```bash
# train one chunk from scratch
uv run --package chess-training python -m chess_training.train \
    ../data/processed/bucket_1000_1.npz \
    --out-dir checkpoints/bucket_1000_1 \
    --epochs 10 --batch-size 256 --lr 1e-3

# resume onto the next chunk -- architecture and epoch numbering are read
# from the checkpoint automatically, not re-specified
uv run --package chess-training python -m chess_training.train \
    ../data/processed/bucket_1000_2.npz \
    --out-dir checkpoints/bucket_1000_2 \
    --epochs 10 \
    --resume-from checkpoints/bucket_1000_1/best.pt

# play against a checkpoint in the terminal
uv run --package chess-training python -m chess_training.play \
    checkpoints/bucket_1000_2/best.pt

# tests
uv run --package chess-training pytest training/tests/ -v
```

Or, from the repo root, via the Makefile (which wraps the full
multi-bucket, multi-chunk workflow — see the root README):

```bash
make train                          # every bucket found under data/processed/
make train BUCKETS=1000,2000 EPOCHS=5
make play CHECKPOINT=training/checkpoints/bucket_1000_2/best.pt
make test
```

### `chess_training.train` flags

| Flag | Default | Meaning |
|---|---|---|
| `npz_path` (positional) | — | Bucket `.npz` from `chess_data.prepare` |
| `--out-dir` | required | Where to save checkpoints |
| `--epochs` | `10` | Epochs to train this run |
| `--batch-size` | `256` | Must be ≤ the train split size (drop_last=True), or the loop raises rather than silently training on zero batches |
| `--lr` | `1e-3` | Adam learning rate |
| `--val-fraction` | `0.05` | Fraction of positions held out for validation (index-based split, not game-based — see `dataset.py`'s docstring for the caveat) |
| `--num-blocks` | `6` | Residual blocks (ignored if `--resume-from` is set — read from the checkpoint instead) |
| `--num-filters` | `64` | Conv filters (same override behavior) |
| `--num-workers` | `2` | `DataLoader` worker processes |
| `--resume-from` | none | Checkpoint to continue training from; see below |

### `chess_training.play` flags

| Flag | Default | Meaning |
|---|---|---|
| `checkpoint` (positional) | — | Path to a `.pt` checkpoint |
| `--human-color` | `white` | Which side you play |
| `--temperature` | `0.0` | `0.0` = always the model's top move (how Maia actually plays); `>0.0` samples instead — more variety, but a weaker/less calibrated opponent |
| `--fen` | none | Start from a custom position instead of the starting position |

## Design notes worth knowing

- **Checkpoints are self-describing.** Every `.pt` file stores
  `num_blocks`, `num_filters`, `epoch`, `val_top1`, `val_top3` alongside
  the weights, so anything loading a checkpoint (`play.py`, the backend,
  `--resume-from`) rebuilds the exact right architecture automatically —
  you never need to remember or pass matching flags by hand.
- **`--resume-from` is cumulative, not a fresh start.** Architecture and
  epoch count both come from the checkpoint, so `epoch_6.pt` after
  resuming from an `epoch_5.pt` checkpoint really is epoch 6 of training,
  not epoch 1 of a new run — checkpoint filenames across chunks never
  collide even if you reuse the same `--out-dir`.
- **`best.pt` tracks the best epoch *within this run only*.** Resuming
  onto a new chunk resets the "best so far" tracking, since a different
  chunk's validation set isn't directly comparable to the previous one's.
- **Watch for overfitting between chunks/epochs**: `train_loss` will keep
  dropping as long as you keep training, but `val_loss` bottoming out and
  then climbing again is the normal sign the model's started memorizing
  rather than generalizing — that's exactly what `best.pt`'s
  save-on-improvement logic exists to catch.