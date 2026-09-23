from __future__ import annotations

import os
import shutil
from pathlib import Path


def find_stockfish() -> Path:
    env_path = os.environ.get("STOCKFISH_PATH")
    if env_path:
        path = Path(env_path)
        if not path.is_file():
            raise RuntimeError(
                f"STOCKFISH_PATH is set to '{path}', but that's not a file."
            )
        if not os.access(path, os.X_OK):
            raise RuntimeError(
                f"STOCKFISH_PATH is set to '{path}', but it's not executable "
                f"-- try: chmod +x {path}"
            )
        return path

    which_path = shutil.which("stockfish")
    if which_path is None:
        raise RuntimeError(
            "stockfish not found. Install it 'brew install stockfish' (macOS), "
            "'apt install stockfish' (Debian/Ubuntu), 'dnf install stockfish' (Fedora) -- "
        )
    return Path(which_path)
