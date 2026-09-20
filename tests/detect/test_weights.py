from __future__ import annotations

from pathlib import Path

import pytest

from detect.errors import ModelFileNotFoundError
from detect.weights import first_existing


def test_first_existing_returns_first_real_file(tmp_path: Path) -> None:
    missing = tmp_path / "gone.pt"
    present = tmp_path / "armor.pt"
    present.write_bytes(b"x")
    assert first_existing((missing, present)) == present


def test_first_existing_raises_when_none_exist(tmp_path: Path) -> None:
    missing = tmp_path / "gone.engine"
    with pytest.raises(ModelFileNotFoundError):
        first_existing((missing,))
