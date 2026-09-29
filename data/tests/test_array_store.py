from pathlib import Path

import numpy as np
from chess_data.array_store import GrowableArrayStore


def test_growable_array_store_grows_and_preserves_data(tmp_path: Path) -> None:
    store = GrowableArrayStore(
        row_shape=(3,), dtype=np.float32, initial_capacity=2, tmp_dir=tmp_path
    )
    rows = [np.array([i, i + 1, i + 2], dtype=np.float32) for i in range(10)]
    for row in rows:
        store.append(row)

    assert store.count == 10
    result = store.finalized()
    assert result.shape == (10, 3)
    for i, row in enumerate(rows):
        assert np.array_equal(result[i], row)

    store.cleanup()


def test_growable_array_store_cleanup_removes_backing_file(tmp_path: Path) -> None:
    store = GrowableArrayStore(
        row_shape=(2,), dtype=np.float32, initial_capacity=2, tmp_dir=tmp_path
    )
    store.append(np.array([1.0, 2.0], dtype=np.float32))
    backing_path = store._path
    assert backing_path.exists()

    store.cleanup()
    assert not backing_path.exists()
