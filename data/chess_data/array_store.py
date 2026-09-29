from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import numpy.typing as npt


class GrowableArrayStore:
    def __init__(
        self,
        row_shape: tuple[int, ...],
        dtype: npt.DTypeLike,
        initial_capacity: int,
        tmp_dir: Path,
    ) -> None:
        self._row_shape = row_shape
        self._dtype = dtype
        self._tmp_dir = tmp_dir
        self._capacity = max(initial_capacity, 1)
        self._count = 0
        self._path = self._new_backing_path()
        self._array = np.memmap(
            self._path, dtype=dtype, mode="w+", shape=(self._capacity, *row_shape)
        )

    def _new_backing_path(self) -> Path:
        fd, name = tempfile.mkstemp(dir=self._tmp_dir, suffix=".dat")
        os.close(fd)
        return Path(name)

    @property
    def count(self) -> int:
        return self._count

    def append(self, row: object) -> None:
        if self._count >= self._capacity:
            self._grow()
        self._array[self._count] = row
        self._count += 1

    def _grow(self) -> None:
        new_capacity = self._capacity * 2
        new_path = self._new_backing_path()
        new_array = np.memmap(
            new_path,
            dtype=self._dtype,
            mode="w+",
            shape=(new_capacity, *self._row_shape),
        )
        new_array[: self._count] = self._array[: self._count]
        new_array.flush()

        old_path = self._path
        self._array = new_array
        self._path = new_path
        self._capacity = new_capacity
        old_path.unlink(missing_ok=True)

    def finalized(self) -> np.ndarray:
        self._array.flush()
        return self._array[: self._count]

    def cleanup(self) -> None:
        del self._array
        self._path.unlink(missing_ok=True)
