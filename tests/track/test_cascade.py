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
    *,
    armor_dead: bool = False,
) -> FieldObservation:
    return FieldObservation(
        bot_id=bot_id,
        x=x,
        y=y,
        team=team,
        role=role,
        pattern_conf=0.9 if role is not None else 0.0,
        armor_dead=armor_dead,
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


def test_off_field_extrapolation_stops_publishing() -> None:
    """外推漂出场地就停发：车不可能在场外，画出去/发裁判都是错的。"""
    tracker = CascadeMatchTracker()
    tracker.step((_obs(1, 14.0, 0.0, Team.RED, Role.THREE),), dt_s=1.0)
    tracker.step((_obs(1, 14.2, 0.0, Team.RED, Role.THREE),), dt_s=1.0)  # CONFIRMED，贴边
    assert len(tracker.step((), dt_s=1.0)) == 1  # LOST 外推 14.4 还在场内
    gone = tracker.step((), dt_s=1.0)  # 外推越过 +14.5
    assert all(item.label != "R3" for item in gone)


def test_off_field_observation_never_reaches_the_minimap() -> None:
    """观测侧 2m 宽容度放进来的远出界点，发布侧收紧到 ±0.5m。"""
    tracker = CascadeMatchTracker()
    out = tracker.step((_obs(1, 15.9, 0.0, Team.RED, Role.THREE),), dt_s=1.0)
    assert out == ()


def test_consecutive_all_dead_kills_the_slot() -> None:
    """连着 DEAD_FRAMES 帧装甲全 dead → 判死停发；复活后重新确认再上屏。"""
    tracker = CascadeMatchTracker()
    _confirm(tracker)
    for _ in range(5):
        out = tracker.step(
            (_obs(1, 1.0, 0.0, Team.RED, Role.THREE, armor_dead=True),), dt_s=1.0
        )
    assert tracker.step((_obs(1, 1.0, 0.0, Team.RED, Role.THREE, armor_dead=True),), dt_s=1.0) == ()
    # 复活：灯条重新点亮，重新确认（2 帧 AGREE）后恢复发布
    tracker.step((_obs(1, 1.0, 0.0, Team.RED, Role.THREE),), dt_s=1.0)
    revived = tracker.step((_obs(1, 1.1, 0.0, Team.RED, Role.THREE),), dt_s=1.0)
    assert any(item.label == "R3" for item in revived)


def test_dead_flicker_never_kills_the_slot() -> None:
    """dead 单帧误闪率 28%：dead/活 交替闪 12 帧也绝不能死。"""
    tracker = CascadeMatchTracker()
    _confirm(tracker)
    for index in range(12):
        dead = index % 2 == 0
        out = tracker.step(
            (_obs(1, 1.0 + 0.01 * index, 0.0, Team.RED, Role.THREE, armor_dead=dead),), dt_s=1.0
        )
        assert any(item.label == "R3" for item in out)
