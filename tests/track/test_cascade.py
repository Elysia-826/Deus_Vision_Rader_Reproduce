from __future__ import annotations

import pytest

from track.cascade import CascadeMatchTracker
from track.types import FieldObservation, Role, Team, TrackPhase


def _obs(
    bot_id: int,
    x: float,
    y: float,
    team: Team | None,
    role: Role | None,
) -> FieldObservation:
    return FieldObservation(
        bot_id=bot_id,
        x=x,
        y=y,
        team=team,
        role=role,
        pattern_conf=0.9 if role is not None else 0.0,
    )


def _confirm(tracker: CascadeMatchTracker) -> None:
    tracker.step(
        (
            _obs(1, 0.0, 0.0, Team.RED, Role.THREE),
            _obs(2, 5.0, 0.0, Team.BLUE, Role.ONE),
        ),
        dt_s=1.0,
    )
    tracker.step(
        (
            _obs(1, 1.5, 0.0, Team.RED, Role.THREE),
            _obs(2, 3.5, 0.0, Team.BLUE, Role.ONE),
        ),
        dt_s=1.0,
    )


def test_pattern_gap_keeps_r3_on_the_measurement() -> None:
    tracker = CascadeMatchTracker()
    _confirm(tracker)
    missed = tracker.step((_obs(1, 3.0, 0.0, None, None), _obs(2, 2.0, 0.0, Team.BLUE, Role.ONE)), dt_s=1.0)
    again = tracker.step((_obs(1, 4.5, 0.0, None, None), _obs(2, 0.5, 0.0, Team.BLUE, Role.ONE)), dt_s=1.0)
    r3 = {item.label: item for item in missed}["R3"]
    r3_again = {item.label: item for item in again}["R3"]
    assert r3.label == "R3"
    assert r3.x == pytest.approx(3.0, abs=0.25)
    assert r3_again.x == pytest.approx(4.5, abs=0.25)
    assert r3.phase is TrackPhase.CONFIRMED


def test_one_frame_crossed_labels_do_not_swap_slots() -> None:
    tracker = CascadeMatchTracker()
    _confirm(tracker)
    swapped = tracker.step(
        (
            _obs(1, 3.0, 0.0, Team.BLUE, Role.ONE),
            _obs(2, 2.0, 0.0, Team.RED, Role.THREE),
        ),
        dt_s=1.0,
    )
    by_label = {item.label: item for item in swapped}
    assert by_label["R3"].bot_id == 1
    assert by_label["R3"].x == pytest.approx(3.0, abs=0.35)
    assert by_label["B1"].bot_id == 2
    assert by_label["B1"].x == pytest.approx(2.0, abs=0.35)


def test_first_agree_frame_is_published() -> None:
    tracker = CascadeMatchTracker()
    out = tracker.step((_obs(1, 0.0, 0.0, Team.RED, Role.THREE),), dt_s=1.0)
    assert len(out) == 1
    assert out[0].label == "R3"
    assert out[0].phase is TrackPhase.TENTATIVE


def test_occlusion_coasts_with_last_velocity_instead_of_vanishing() -> None:
    tracker = CascadeMatchTracker()
    tracker.step((_obs(1, 0.0, 0.0, Team.RED, Role.THREE),), dt_s=1.0)
    tracker.step((_obs(1, 1.5, 0.0, Team.RED, Role.THREE),), dt_s=1.0)
    tracker.step((_obs(1, 3.0, 0.0, Team.RED, Role.THREE),), dt_s=1.0)
    coast = [tracker.step((), dt_s=1.0) for _ in range(3)]
    assert all(len(frame) == 1 and frame[0].label == "R3" for frame in coast)
    assert coast[0][0].phase is TrackPhase.LOST
    assert coast[0][0].x == pytest.approx(4.5, abs=0.45)
    assert coast[1][0].x == pytest.approx(6.0, abs=0.55)
    assert coast[2][0].x == pytest.approx(7.5, abs=0.7)
    assert coast[0][0].x > 3.3
