from pathlib import Path

import numpy as np
import pytest
import torch
from chess_shared import NUM_MOVES, NUM_PLANES
from chess_training.dataset import make_train_val_split
from chess_training.model import MaiaPolicyNet
from chess_training.train import evaluate, pick_device, train


def _write_synthetic_npz(
    path: Path, n: int, seed: int = 0, game_ids: np.ndarray | None = None
) -> None:
    rng = np.random.default_rng(seed)
    boards = rng.random((n, NUM_PLANES, 8, 8), dtype=np.float32)
    moves = rng.integers(0, NUM_MOVES, size=n, dtype=np.int64)
    if game_ids is None:
        game_ids = np.arange(n, dtype=np.int64)
    np.savez_compressed(path, boards=boards, moves=moves, game_ids=game_ids)


def test_model_forward_shape():
    model = MaiaPolicyNet(NUM_PLANES, NUM_MOVES, num_blocks=2, num_filters=8)
    x = torch.randn(4, NUM_PLANES, 8, 8)
    out = model(x)
    assert out.shape == (4, NUM_MOVES)


def test_dataset_split_shapes(tmp_path: Path):
    npz_path = tmp_path / "synthetic.npz"
    _write_synthetic_npz(npz_path, n=200)

    train_loader, val_loader = make_train_val_split(
        str(npz_path), batch_size=16, val_fraction=0.1, seed=0
    )
    assert train_loader.num_samples == 180
    assert val_loader.num_samples == 20

    boards, moves = next(iter(train_loader))
    assert boards.shape == (16, NUM_PLANES, 8, 8)
    assert moves.shape == (16,)
    assert moves.dtype == torch.long


def test_dataset_split_is_disjoint(tmp_path: Path):
    npz_path = tmp_path / "synthetic.npz"
    _write_synthetic_npz(npz_path, n=100)

    train_loader, val_loader = make_train_val_split(
        str(npz_path), batch_size=8, val_fraction=0.2, seed=0
    )
    train_idx = set(train_loader.indices.tolist())
    val_idx = set(val_loader.indices.tolist())
    assert train_idx.isdisjoint(val_idx)
    assert len(train_idx) + len(val_idx) == 100


def test_split_is_grouped_by_game_not_by_position(tmp_path: Path):
    game_sizes = [3, 1, 5, 2, 4, 6, 1, 2, 3, 5]
    n = sum(game_sizes)
    game_ids = np.concatenate(
        [np.full(size, gid, dtype=np.int64) for gid, size in enumerate(game_sizes)]
    )
    npz_path = tmp_path / "grouped.npz"
    _write_synthetic_npz(npz_path, n=n, game_ids=game_ids)

    train_loader, val_loader = make_train_val_split(
        str(npz_path), batch_size=4, val_fraction=0.3, seed=0
    )

    train_games = set(game_ids[train_loader.indices.numpy()].tolist())
    val_games = set(game_ids[val_loader.indices.numpy()].tolist())
    assert train_games.isdisjoint(val_games)
    assert len(train_loader.indices) + len(val_loader.indices) == n
    assert len(val_games) >= 1
    assert len(train_games) >= 1


def test_make_train_val_split_requires_game_ids(tmp_path: Path):
    rng = np.random.default_rng(0)
    n = 20
    boards = rng.random((n, NUM_PLANES, 8, 8), dtype=np.float32)
    moves = rng.integers(0, NUM_MOVES, size=n, dtype=np.int64)
    npz_path = tmp_path / "legacy_no_game_ids.npz"
    np.savez_compressed(npz_path, boards=boards, moves=moves)

    with pytest.raises(KeyError, match="game_ids"):
        make_train_val_split(str(npz_path), batch_size=4, val_fraction=0.2, seed=0)


def test_loader_len_respects_drop_last(tmp_path: Path):
    npz_path = tmp_path / "synthetic.npz"
    _write_synthetic_npz(npz_path, n=100)

    train_loader, val_loader = make_train_val_split(
        str(npz_path), batch_size=32, val_fraction=0.2, seed=0
    )
    assert len(train_loader) == 2
    assert len(list(train_loader)) == 2
    assert len(val_loader) == 1
    ((val_boards, _),) = list(val_loader)
    assert val_boards.shape[0] == 20


def test_val_loader_covers_every_val_sample_once(tmp_path: Path):
    npz_path = tmp_path / "synthetic.npz"
    _write_synthetic_npz(npz_path, n=50)

    _, val_loader = make_train_val_split(
        str(npz_path), batch_size=4, val_fraction=0.3, seed=0
    )
    seen = sum(moves.shape[0] for _, moves in val_loader)
    assert seen == val_loader.num_samples == 15


def test_train_loader_shuffles_between_epochs_but_is_a_permutation(tmp_path: Path):
    npz_path = tmp_path / "synthetic.npz"
    n = 64
    rng = np.random.default_rng(0)
    boards = rng.random((n, NUM_PLANES, 8, 8), dtype=np.float32)
    moves = np.arange(n, dtype=np.int64)
    game_ids = np.arange(n, dtype=np.int64)
    np.savez_compressed(npz_path, boards=boards, moves=moves, game_ids=game_ids)

    train_loader, _ = make_train_val_split(
        str(npz_path), batch_size=8, val_fraction=0.25, seed=0
    )

    epoch1 = torch.cat([m for _, m in train_loader]).tolist()
    epoch2 = torch.cat([m for _, m in train_loader]).tolist()

    assert sorted(epoch1) == sorted(epoch2) == sorted(train_loader.indices.tolist())
    assert epoch1 != epoch2


def test_compact_boards_stores_float16(tmp_path: Path):
    npz_path = tmp_path / "synthetic.npz"
    _write_synthetic_npz(npz_path, n=20)

    train_loader, _ = make_train_val_split(
        str(npz_path), batch_size=4, val_fraction=0.2, seed=0, compact_boards=True
    )
    assert train_loader.boards.dtype == torch.float16


def test_evaluate_runs_and_returns_sane_metrics(tmp_path: Path):
    npz_path = tmp_path / "synthetic.npz"
    _write_synthetic_npz(npz_path, n=64)
    _, val_loader = make_train_val_split(
        str(npz_path), batch_size=8, val_fraction=0.5, seed=0
    )

    model = MaiaPolicyNet(NUM_PLANES, NUM_MOVES, num_blocks=1, num_filters=8)
    device = torch.device("cpu")

    loss, top1, top3 = evaluate(
        model, val_loader, device, amp=False, channels_last=False
    )
    assert loss > 0
    assert 0.0 <= top1 <= 1.0
    assert 0.0 <= top3 <= 1.0
    assert top3 >= top1


def test_full_training_loop_runs_and_produces_checkpoints(tmp_path: Path):
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
    )

    assert (out_dir / "epoch_1.pt").exists()
    assert (out_dir / "epoch_2.pt").exists()
    assert (out_dir / "best.pt").exists()
    assert (out_dir / "last.pt").exists()

    checkpoint = torch.load(out_dir / "best.pt", weights_only=True)
    assert "model_state_dict" in checkpoint
    assert "val_top1" in checkpoint

    last = torch.load(out_dir / "last.pt", weights_only=True)
    assert last["epoch"] == 2

    model = MaiaPolicyNet(NUM_PLANES, NUM_MOVES, num_blocks=1, num_filters=8)
    model.load_state_dict(checkpoint["model_state_dict"])


def test_checkpoint_records_encoding_dimensions(tmp_path: Path):
    npz_path = tmp_path / "synthetic.npz"
    _write_synthetic_npz(npz_path, n=16)
    out_dir = tmp_path / "checkpoints"

    train(
        npz_path=npz_path,
        out_dir=out_dir,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        val_fraction=0.25,
        num_blocks=1,
        num_filters=8,
    )

    checkpoint = torch.load(out_dir / "last.pt", weights_only=True)
    assert checkpoint["in_planes"] == NUM_PLANES
    assert checkpoint["num_moves"] == NUM_MOVES


def test_resume_rejects_checkpoint_with_mismatched_in_planes(tmp_path: Path):
    npz_path = tmp_path / "synthetic.npz"
    _write_synthetic_npz(npz_path, n=16)
    chunk1_dir = tmp_path / "chunk1"
    train(
        npz_path=npz_path,
        out_dir=chunk1_dir,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        val_fraction=0.25,
        num_blocks=1,
        num_filters=8,
    )

    stale = torch.load(chunk1_dir / "last.pt", weights_only=True)
    stale["in_planes"] = NUM_PLANES + 1
    stale_path = chunk1_dir / "stale_planes.pt"
    torch.save(stale, stale_path)

    with pytest.raises(ValueError, match="in_planes"):
        train(
            npz_path=npz_path,
            out_dir=tmp_path / "chunk2",
            epochs=1,
            batch_size=4,
            lr=1e-3,
            val_fraction=0.25,
            num_blocks=1,
            num_filters=8,
            resume_from=stale_path,
        )


def test_resume_rejects_checkpoint_with_mismatched_num_moves(tmp_path: Path):
    npz_path = tmp_path / "synthetic.npz"
    _write_synthetic_npz(npz_path, n=16)
    chunk1_dir = tmp_path / "chunk1"
    train(
        npz_path=npz_path,
        out_dir=chunk1_dir,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        val_fraction=0.25,
        num_blocks=1,
        num_filters=8,
    )

    stale = torch.load(chunk1_dir / "last.pt", weights_only=True)
    stale["num_moves"] = NUM_MOVES - 1
    stale_path = chunk1_dir / "stale_moves.pt"
    torch.save(stale, stale_path)

    with pytest.raises(ValueError, match="num_moves"):
        train(
            npz_path=npz_path,
            out_dir=tmp_path / "chunk2",
            epochs=1,
            batch_size=4,
            lr=1e-3,
            val_fraction=0.25,
            num_blocks=1,
            num_filters=8,
            resume_from=stale_path,
        )


def test_resume_from_legacy_checkpoint_without_encoding_fields_still_works(
    tmp_path: Path,
):
    npz_path = tmp_path / "synthetic.npz"
    _write_synthetic_npz(npz_path, n=16)
    chunk1_dir = tmp_path / "chunk1"
    train(
        npz_path=npz_path,
        out_dir=chunk1_dir,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        val_fraction=0.25,
        num_blocks=1,
        num_filters=8,
    )

    legacy = torch.load(chunk1_dir / "last.pt", weights_only=True)
    del legacy["in_planes"]
    del legacy["num_moves"]
    legacy_path = chunk1_dir / "legacy.pt"
    torch.save(legacy, legacy_path)

    chunk2_dir = tmp_path / "chunk2"
    train(
        npz_path=npz_path,
        out_dir=chunk2_dir,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        val_fraction=0.25,
        num_blocks=1,
        num_filters=8,
        resume_from=legacy_path,
    )
    assert (chunk2_dir / "epoch_2.pt").exists()


def test_save_every_epoch_false_skips_epoch_files_but_keeps_best_and_last(
    tmp_path: Path,
):
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
        save_every_epoch=False,
    )

    assert not (out_dir / "epoch_1.pt").exists()
    assert not (out_dir / "epoch_2.pt").exists()
    assert (out_dir / "best.pt").exists()
    assert (out_dir / "last.pt").exists()


def test_channels_last_training_runs_and_checkpoint_loads_into_plain_model(
    tmp_path: Path,
):
    npz_path = tmp_path / "synthetic.npz"
    _write_synthetic_npz(npz_path, n=64)
    out_dir = tmp_path / "checkpoints"

    train(
        npz_path=npz_path,
        out_dir=out_dir,
        epochs=1,
        batch_size=8,
        lr=1e-3,
        val_fraction=0.25,
        num_blocks=1,
        num_filters=8,
        channels_last=True,
    )

    checkpoint = torch.load(out_dir / "last.pt", weights_only=True)
    model = MaiaPolicyNet(NUM_PLANES, NUM_MOVES, num_blocks=1, num_filters=8)
    model.load_state_dict(checkpoint["model_state_dict"])


def test_train_val_split_shares_underlying_arrays_not_duplicated(tmp_path: Path):
    npz_path = tmp_path / "synthetic.npz"
    _write_synthetic_npz(npz_path, n=50)

    train_loader, val_loader = make_train_val_split(
        str(npz_path), batch_size=8, val_fraction=0.2, seed=0
    )
    assert train_loader.boards is val_loader.boards
    assert train_loader.moves is val_loader.moves


def test_compressed_npz_triggers_mmap_fallback_warning(tmp_path: Path):
    npz_path = tmp_path / "synthetic.npz"
    _write_synthetic_npz(npz_path, n=20)

    with pytest.warns(UserWarning, match="can never be memory-mapped"):
        make_train_val_split(str(npz_path), batch_size=4, val_fraction=0.2, seed=0)


def test_resume_from_picks_up_architecture_from_checkpoint_not_args(tmp_path: Path):
    torch.manual_seed(0)
    n = 16
    rng = np.random.default_rng(0)
    boards = rng.random((n, NUM_PLANES, 8, 8), dtype=np.float32)
    moves = rng.integers(0, NUM_MOVES, size=n, dtype=np.int64)
    game_ids = np.arange(n, dtype=np.int64)
    npz_path = tmp_path / "data.npz"
    np.savez_compressed(npz_path, boards=boards, moves=moves, game_ids=game_ids)

    chunk1_dir = tmp_path / "chunk1"
    train(
        npz_path=npz_path,
        out_dir=chunk1_dir,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        val_fraction=0.25,
        num_blocks=3,
        num_filters=12,
    )
    chunk1_checkpoint = torch.load(chunk1_dir / "epoch_1.pt", weights_only=True)
    assert chunk1_checkpoint["num_blocks"] == 3
    assert chunk1_checkpoint["num_filters"] == 12

    chunk2_dir = tmp_path / "chunk2"
    train(
        npz_path=npz_path,
        out_dir=chunk2_dir,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        val_fraction=0.25,
        num_blocks=999,
        num_filters=999,
        resume_from=chunk1_dir / "epoch_1.pt",
    )
    chunk2_checkpoint = torch.load(chunk2_dir / "epoch_2.pt", weights_only=True)
    assert chunk2_checkpoint["num_blocks"] == 3
    assert chunk2_checkpoint["num_filters"] == 12


def test_resume_from_continues_epoch_numbering_cumulatively(tmp_path: Path):
    torch.manual_seed(0)
    n = 16
    rng = np.random.default_rng(0)
    boards = rng.random((n, NUM_PLANES, 8, 8), dtype=np.float32)
    moves = rng.integers(0, NUM_MOVES, size=n, dtype=np.int64)
    game_ids = np.arange(n, dtype=np.int64)
    npz_path = tmp_path / "data.npz"
    np.savez_compressed(npz_path, boards=boards, moves=moves, game_ids=game_ids)

    chunk1_dir = tmp_path / "chunk1"
    train(
        npz_path=npz_path,
        out_dir=chunk1_dir,
        epochs=5,
        batch_size=4,
        lr=1e-3,
        val_fraction=0.25,
        num_blocks=2,
        num_filters=8,
    )
    assert (chunk1_dir / "epoch_5.pt").exists()

    chunk2_dir = tmp_path / "chunk2"
    train(
        npz_path=npz_path,
        out_dir=chunk2_dir,
        epochs=3,
        batch_size=4,
        lr=1e-3,
        val_fraction=0.25,
        num_blocks=2,
        num_filters=8,
        resume_from=chunk1_dir / "epoch_5.pt",
    )
    assert not (chunk2_dir / "epoch_1.pt").exists()
    assert (chunk2_dir / "epoch_6.pt").exists()
    assert (chunk2_dir / "epoch_7.pt").exists()
    assert (chunk2_dir / "epoch_8.pt").exists()


def test_resume_from_last_pt_works(tmp_path: Path):
    torch.manual_seed(0)
    npz_path = tmp_path / "data.npz"
    _write_synthetic_npz(npz_path, n=16)

    chunk1_dir = tmp_path / "chunk1"
    train(
        npz_path=npz_path,
        out_dir=chunk1_dir,
        epochs=2,
        batch_size=4,
        lr=1e-3,
        val_fraction=0.25,
        num_blocks=1,
        num_filters=8,
        save_every_epoch=False,
    )

    chunk2_dir = tmp_path / "chunk2"
    train(
        npz_path=npz_path,
        out_dir=chunk2_dir,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        val_fraction=0.25,
        num_blocks=1,
        num_filters=8,
        resume_from=chunk1_dir / "last.pt",
    )
    assert (chunk2_dir / "epoch_3.pt").exists()


def test_resume_from_actually_warm_starts_not_reinitializes(tmp_path: Path):
    torch.manual_seed(0)
    n = 32
    rng = np.random.default_rng(0)
    boards = rng.random((n, NUM_PLANES, 8, 8), dtype=np.float32)
    moves = (np.round(boards.sum(axis=(1, 2, 3)) * 1000).astype(np.int64)) % NUM_MOVES
    game_ids = np.arange(n, dtype=np.int64)
    npz_path = tmp_path / "learnable.npz"
    np.savez_compressed(npz_path, boards=boards, moves=moves, game_ids=game_ids)

    chunk1_dir = tmp_path / "chunk1"
    train(
        npz_path=npz_path,
        out_dir=chunk1_dir,
        epochs=30,
        batch_size=8,
        lr=3e-3,
        val_fraction=0.25,
        num_blocks=2,
        num_filters=16,
    )

    chunk2_dir = tmp_path / "chunk2"
    train(
        npz_path=npz_path,
        out_dir=chunk2_dir,
        epochs=1,
        batch_size=8,
        lr=3e-3,
        val_fraction=0.25,
        num_blocks=2,
        num_filters=16,
        resume_from=chunk1_dir / "epoch_30.pt",
    )

    chunk2_checkpoint = torch.load(chunk2_dir / "epoch_31.pt", weights_only=True)
    model = MaiaPolicyNet(NUM_PLANES, NUM_MOVES, num_blocks=2, num_filters=16)
    model.load_state_dict(chunk2_checkpoint["model_state_dict"])
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(boards)).argmax(dim=1).numpy()
    train_acc = (preds == moves).mean()
    assert train_acc > 0.5, (
        f"expected chunk 2 to still reflect chunk 1's learning after just 1 more "
        f"epoch, got acc={train_acc} -- looks like resume_from reinitialized "
        "instead of warm-starting"
    )


def test_model_can_overfit_a_tiny_learnable_dataset(tmp_path: Path):
    torch.manual_seed(0)
    n = 32
    rng = np.random.default_rng(0)
    boards = rng.random((n, NUM_PLANES, 8, 8), dtype=np.float32)
    moves = (np.round(boards.sum(axis=(1, 2, 3)) * 1000).astype(np.int64)) % NUM_MOVES
    game_ids = np.arange(n, dtype=np.int64)

    npz_path = tmp_path / "learnable.npz"
    np.savez_compressed(npz_path, boards=boards, moves=moves, game_ids=game_ids)
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
    npz_path = tmp_path / "tiny.npz"
    _write_synthetic_npz(npz_path, n=8)

    with pytest.raises(ValueError, match="zero batches"):
        train(
            npz_path=npz_path,
            out_dir=tmp_path / "checkpoints",
            epochs=1,
            batch_size=32,
            lr=1e-3,
            val_fraction=0.25,
            num_blocks=1,
            num_filters=8,
        )


def test_pick_device_returns_valid_device():
    device = pick_device()
    assert device.type in ("mps", "cuda", "cpu")
