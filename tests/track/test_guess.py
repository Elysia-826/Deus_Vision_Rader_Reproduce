from __future__ import annotations

from track.guess import apply_guess_to_lost, guess_field_xy, guess_points_for
from track.types import PublishedTrack, Role, Team, TrackPhase


def _track(
    label: str,
    x: float,
    y: float,
    *,
    vx: float = 0.0,
    vy: float = 0.0,
    phase: TrackPhase = TrackPhase.LOST,
) -> PublishedTrack:
    team = Team.RED if label.startswith("R") else Team.BLUE
    return PublishedTrack(label=label, team=team, x=x, y=y, phase=phase, bot_id=1, vx=vx, vy=vy)


def test_velocity_direction_picks_the_point_ahead() -> None:
    """步兵在原点朝 +x 走：候选里 (-5.5,0) 在身后，(0,±6) 侧向，中央高地顺路。"""
    guess = guess_field_xy(_track("B3", 0.0, 0.0, vx=2.0, vy=0.0))
    assert guess is not None
    assert (guess.x, guess.y) == (0.0, 0.0)


def test_zero_velocity_falls_back_to_nearest_point() -> None:
    """没速度就挑最近的战略点。"""
    guess = guess_field_xy(_track("B3", 0.4, 0.3, vx=0.0, vy=0.0))
    assert guess is not None
    assert (guess.x, guess.y) == (0.0, 0.0)


def test_unknown_role_is_left_alone() -> None:
    """Q（前哨）没有候选点：返回 None，apply 里原样保留。"""
    assert guess_field_xy(_track("BQ", 1.0, 1.0)) is None
    kept = apply_guess_to_lost((_track("BQ", 1.0, 1.0),))
    assert kept[0].x == 1.0 and kept[0].y == 1.0


def test_apply_only_touches_lost_slots() -> None:
    confirmed = _track("B1", 3.0, 3.0, phase=TrackPhase.CONFIRMED)
    lost_far = _track("R1", -12.0, 0.0, vx=-2.0, vy=0.0)  # 英雄朝对家基地走
    out = apply_guess_to_lost((confirmed, lost_far))
    assert out[0] == confirmed  # CONFIRMED 不动
    assert (out[1].x, out[1].y) == (-12.5, 0.0)  # LOST 换成猜点
    assert out[1].phase is TrackPhase.LOST  # phase 保留：空心点 + LOST 语义


def test_guess_points_table_has_every_role() -> None:
    for role in (Role.ONE, Role.TWO, Role.THREE, Role.FOUR, Role.SENTRY):
        assert guess_points_for(role), f"{role} 缺战略点表"
