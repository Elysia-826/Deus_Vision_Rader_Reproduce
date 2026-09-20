from __future__ import annotations

import pytest

from locate.motion import point_on_loop
from locate.types import FieldXY

SQUARE = (
    FieldXY(0.0, 0.0),
    FieldXY(4.0, 0.0),
    FieldXY(4.0, 4.0),
    FieldXY(0.0, 4.0),
)


def test_t0_is_first_waypoint() -> None:
    p = point_on_loop(SQUARE, 0.0, speed_mps=2.0)
    assert p.x == pytest.approx(0.0)
    assert p.y == pytest.approx(0.0)


def test_halfway_first_edge() -> None:
    p = point_on_loop(SQUARE, 1.0, speed_mps=2.0)
    assert p.x == pytest.approx(2.0)
    assert p.y == pytest.approx(0.0)


def test_wraps_after_full_lap() -> None:
    # perimeter 16 m, 2 m/s → 8 s per lap
    p = point_on_loop(SQUARE, 8.0, speed_mps=2.0)
    assert p.x == pytest.approx(0.0)
    assert p.y == pytest.approx(0.0)


def test_second_edge() -> None:
    p = point_on_loop(SQUARE, 3.0, speed_mps=2.0)
    assert p.x == pytest.approx(4.0)
    assert p.y == pytest.approx(2.0)
