from pathlib import Path

import numpy as np
import pytest
import torch
from chess_shared import NUM_MOVES, NUM_PLANES
from chess_training.dataset import make_train_val_split
from chess_training.model import MaiaPolicyNet
from chess_training.train import evaluate, pick_device, train


def _write_synthetic_npz(path: Path, n: int, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    boards = rng.random((n, NUM_PLANES, 8, 8), dtype=np.float32)
    moves = rng.integers(0, NUM_MOVES, size=n, dtype=np.int64)
    np.savez_compressed(path, boards=boards, moves=moves)


def test_model_forward_shape():
    model = MaiaPolicyNet(NUM_PLANES, NUM_MOVES, num_blocks=2, num_filters=8)
    x = torch.randn(4, NUM_PLANES, 8, 8)
    out = model(x)
    assert out.shape == (4, NUM_MOVES)


def test_dataset_split_shapes(tmp_path: Path):
    npz_path = tmp_path / "synthetic.npz"
    _write_synthetic_npz(npz_path, n=200)

    train_ds, val_ds = make_train_val_split(str(npz_path), val_fraction=0.1, seed=0)
    assert len(train_ds) == 180
    assert len(val_ds) == 20

    board, move = train_ds[0]
    assert board.shape == (NUM_PLANES, 8, 8)
    assert move.dtype == torch.long


def test_dataset_split_is_disjoint(tmp_path: Path):
    npz_path = tmp_path / "synthetic.npz"
    _write_synthetic_npz(npz_path, n=100)

    train_ds, val_ds = make_train_val_split(str(npz_path), val_fraction=0.2, seed=0)
    train_idx = set(train_ds._indices.tolist())
    val_idx = set(val_ds._indices.tolist())
    assert train_idx.isdisjoint(val_idx)
    assert len(train_idx) + len(val_idx) == 100


def test_evaluate_runs_and_returns_sane_metrics(tmp_path: Path):
    npz_path = tmp_path / "synthetic.npz"
    _write_synthetic_npz(npz_path, n=64)
    _, val_ds = make_train_val_split(str(npz_path), val_fraction=0.5, seed=0)

    from torch.utils.data import DataLoader

    loader = DataLoader(val_ds, batch_size=8)
    model = MaiaPolicyNet(NUM_PLANES, NUM_MOVES, num_blocks=1, num_filters=8)
    device = torch.device("cpu")

    loss, top1, top3 = evaluate(model, loader, device)
    assert loss > 0
    assert 0.0 <= top1 <= 1.0
    assert 0.0 <= top3 <= 1.0
    assert top3 >= top1  # top-3 accuracy can never be lower than top-1


def test_full_training_loop_runs_and_produces_checkpoints(tmp_path: Path):
    """
    Not testing that the model learns anything meaningful (random labels,
    two epochs -- there's no signal to learn) -- testing that the training
    loop itself runs end to end without crashing and produces the expected
    checkpoint files, on a tiny model so this stays fast.
    """
    npz_path = tmp_path / "synthetic.npz"
    _write_synthetic_npz(npz_path, n=64)
    out_dir = tmp_path / "checkpoints"

    train(
        npz_path=npz_path,
        out_dir=out_dir,
        epochs=2,
        batch_size=8,
        lr=1e-3,
        val_fraction=0.25,
        num_blocks=1,
        num_filters=8,
        num_workers=0,
    )

    assert (out_dir / "epoch_1.pt").exists()
    assert (out_dir / "epoch_2.pt").exists()
    assert (out_dir / "best.pt").exists()

    checkpoint = torch.load(out_dir / "best.pt", weights_only=True)
    assert "model_state_dict" in checkpoint
    assert "val_top1" in checkpoint

    model = MaiaPolicyNet(NUM_PLANES, NUM_MOVES, num_blocks=1, num_filters=8)
    model.load_state_dict(checkpoint["model_state_dict"])


def test_train_val_split_shares_underlying_arrays_not_duplicated(tmp_path: Path):
    """
    Regression test for a real memory bug: train/val datasets used to each
    call np.load() independently, loading two full separate in-RAM copies
    of the same array. They should now share one loaded array.
    """
    npz_path = tmp_path / "synthetic.npz"
    _write_synthetic_npz(npz_path, n=50)

    train_ds, val_ds = make_train_val_split(str(npz_path), val_fraction=0.2, seed=0)
    assert train_ds._boards is val_ds._boards
    assert train_ds._moves is val_ds._moves


def test_compressed_npz_triggers_mmap_fallback_warning(tmp_path: Path):
    """
    Regression test: chess_data.prepare writes compressed .npz files, and
    numpy silently ignores mmap_mode for compressed archives -- the array
    gets fully loaded into RAM instead. We should surface that with a
    warning rather than silently pretending it's memory-mapped.
    """
    npz_path = tmp_path / "synthetic.npz"
    _write_synthetic_npz(npz_path, n=20)  # _write_synthetic_npz uses savez_compressed

    with pytest.warns(UserWarning, match="could not actually be memory-mapped"):
        make_train_val_split(str(npz_path), val_fraction=0.2, seed=0)


def test_model_can_overfit_a_tiny_learnable_dataset(tmp_path: Path):
    """
    The previous test only proves the loop runs without crashing (random
    labels give it nothing to learn). This test proves the wiring is
    actually correct: give the model a tiny dataset where the move label
    is a deterministic function of the board tensor, and confirm training
    accuracy climbs sharply. If backprop, the loss function, or the
    optimizer step were wired wrong, this would stay near chance level
    (1/NUM_MOVES) no matter how many epochs we run.
    """
    n = 32
    rng = np.random.default_rng(0)
    boards = rng.random((n, NUM_PLANES, 8, 8), dtype=np.float32)
    moves = (np.round(boards.sum(axis=(1, 2, 3)) * 1000).astype(np.int64)) % NUM_MOVES

    npz_path = tmp_path / "learnable.npz"
    np.savez_compressed(npz_path, boards=boards, moves=moves)
    out_dir = tmp_path / "checkpoints"

    train(
        npz_path=npz_path,
        out_dir=out_dir,
        epochs=30,
        batch_size=8,
        lr=3e-3,
        val_fraction=0.25,
        num_blocks=2,
        num_filters=16,
        num_workers=0,
    )

    checkpoint = torch.load(out_dir / "epoch_30.pt", weights_only=True)
    model = MaiaPolicyNet(NUM_PLANES, NUM_MOVES, num_blocks=2, num_filters=16)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(boards))
        preds = logits.argmax(dim=1).numpy()
    train_acc = (preds == moves).mean()
    assert train_acc > 0.5, (
        f"expected the model to overfit this tiny dataset, got acc={train_acc}"
    )


def test_train_raises_on_batch_size_larger_than_train_split(tmp_path: Path):
    """
    Regression test for a real bug caught while writing this test suite:
    batch_size > train split size, combined with drop_last=True, used to
    silently produce zero batches per epoch -- the loop would "train" for
    N epochs doing nothing, printing a misleading train_loss=0.0000 the
    whole time. Now it should fail loudly instead.
    """
    npz_path = tmp_path / "tiny.npz"
    _write_synthetic_npz(npz_path, n=8)

    with pytest.raises(ValueError, match="zero batches"):
        train(
            npz_path=npz_path,
            out_dir=tmp_path / "checkpoints",
            epochs=1,
            batch_size=32,  # larger than the ~6-position train split
            lr=1e-3,
            val_fraction=0.25,
            num_blocks=1,
            num_filters=8,
            num_workers=0,
        )


def test_pick_device_returns_valid_device():
    device = pick_device()
    assert device.type in ("mps", "cuda", "cpu")
