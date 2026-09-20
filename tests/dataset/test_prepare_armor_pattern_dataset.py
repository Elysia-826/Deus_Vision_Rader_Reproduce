from pathlib import Path

import pytest

from prepare_armor_pattern_dataset import (
    CLASS_NAMES,
    DatasetLayoutError,
    Sample,
    flags_for,
    xany_stem,
)


def test_flags_one_hot() -> None:
    flags = flags_for("BS")
    assert list(flags) == list(CLASS_NAMES)
    assert flags["BS"] is True
    assert sum(flags.values()) == 1


def test_unknown_class_raises() -> None:
    with pytest.raises(DatasetLayoutError):
        flags_for("G1")


def test_xany_stem_prefixes_class() -> None:
    sample = Sample(cls="R3", image=Path("R3/14096_001.jpg"), stem="14096_001")
    assert xany_stem(sample) == "R3__14096_001"
