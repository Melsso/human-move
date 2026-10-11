# chess-training

Model definition and training loop. It trains one policy network per rating bucket on the `.npz` files `chess_data.prepare` produces. The end-to-end workflow is `make prepare` then `make train` (see the root README). The CLI below is the single-chunk building block underneath `make train`.

## Modules

- **`model.py`** — `MaiaPolicyNet`, a 6-block, 64-filter residual CNN over `NUM_PLANES` input planes and `NUM_MOVES` outputs. Also:
  - `load_checkpoint(path)` loads a checkpoint and rejects one whose `in_planes` or `num_moves` differ from the current encoding.
  - `model_from_checkpoint(checkpoint)` rebuilds the network.
  - The training resume logic and the backend both use these two.
- **`dataset.py`** — `make_train_val_split` loads a bucket `.npz` fully into RAM and splits train and validation **by game id**, so positions from one game never land on both sides. Both loaders share the same tensors.
- **`train.py`** — the training loop with MPS/CUDA/CPU auto-detection and `--resume-from` for chunked training.

## Commands

```bash
uv run --package chess-training python -m chess_training.train \
    data/processed/bucket_1000_1.npz \
    --out-dir training/checkpoints/bucket_1000_1 \
    --epochs 10

uv run --package chess-training python -m chess_training.train \
    data/processed/bucket_1000_2.npz \
    --out-dir training/checkpoints/bucket_1000_2 \
    --epochs 10 \
    --resume-from training/checkpoints/bucket_1000_1/best.pt

uv run --package chess-training pytest training/tests/ -v
```

From the repo root, `make train [BUCKETS=1000,1500] [EPOCHS=10]` trains every chunk of every bucket in order, resuming each chunk from the previous one and skipping chunks that already have a `best.pt`.

### `chess_training.train` flags

| Flag | Default | Meaning |
|---|---|---|
| `npz_path` (positional) | — | Bucket `.npz` from `chess_data.prepare` |
| `--out-dir` | required | Where to save checkpoints |
| `--epochs` | `10` | Epochs to train this run |
| `--batch-size` | `1024` | Must not exceed the train split size (`drop_last=True`); otherwise the loop raises |
| `--lr` | `2e-3` | Adam learning rate |
| `--val-fraction` | `0.05` | Fraction of **games** held out for validation |
| `--num-blocks` | `6` | Residual blocks (read from the checkpoint when resuming) |
| `--num-filters` | `64` | Conv filters (read from the checkpoint when resuming) |
| `--resume-from` | none | Checkpoint to continue from |
| `--save-every-epoch` / `--no-save-every-epoch` | on | Write `epoch_N.pt` each epoch; `best.pt` and `last.pt` are always written |
| `--no-save-every-epoch` | set | avoids ~170 MB per epoch file. make train passes it by default. |

## Design notes

- Checkpoints are self-describing. They store the architecture, epoch, validation metrics, `in_planes` and `num_moves`. Loading rebuilds the right network and refuses checkpoints from an older encoding.
- `--resume-from` is cumulative. Epoch numbering continues from the checkpoint, so filenames never collide across chunks.
- `best.pt` tracks the best validation top-1 within the current run only, because each chunk has its own validation games.
- Which checkpoint each playable tier uses, and its search settings, is defined in `shared/chess_shared/tiers.py`, not here.