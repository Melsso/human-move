from __future__ import annotations

import io
from pathlib import Path
from typing import TextIO

import zstandard as zstd


def open_pgn_stream(path: Path) -> TextIO:
    if path.suffix == ".zst":
        dctx = zstd.ZstdDecompressor()
        stream_reader = dctx.stream_reader(path.open("rb"))
        return io.TextIOWrapper(stream_reader, encoding="utf-8", errors="replace")
    return open(path, encoding="utf-8", errors="replace")
