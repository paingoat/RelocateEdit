"""Put a vendored repo on `sys.path` only when its model is loaded."""

from __future__ import annotations

import sys
from pathlib import Path

from relocate_edit.config import ROOT

THIRD_PARTY = ROOT / "third_party"


def vendor_path(name: str) -> Path:
    path = THIRD_PARTY / name
    if not path.is_dir():
        raise FileNotFoundError(f"Vendored code is missing: {path}")
    return path


def ensure_on_path(path: Path) -> None:
    entry = str(path)
    if entry not in sys.path:
        sys.path.insert(0, entry)
