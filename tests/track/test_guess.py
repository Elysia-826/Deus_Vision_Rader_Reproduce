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
    pos_uncertainty: float = 1.2,  # 默认超标，让 σ 门触发猜点
    lost_s: float = 0.0,
) -> PublishedTrack:
    team = Team.RED if label.startswith("R") else Team.BLUE
    return PublishedTrack(
        label=label, team=team, x=x, y=y, phase=phase, bot_id=1,
        vx=vx, vy=vy, pos_uncertainty=pos_uncertainty, lost_s=lost_s,
    )


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
    lost_far = _track("R1", -12.0, 0.0, vx=-2.0, vy=0.0, pos_uncertainty=1.2)
    out = apply_guess_to_lost((confirmed, lost_far))
    assert out[0] == confirmed
    assert (out[1].x, out[1].y) == (-11.0, -5.0)
    assert out[1].phase is TrackPhase.LOST


def test_low_uncertainty_does_not_trigger_guess() -> None:
    """短遮挡双门全没过：σ 0.3 < 0.8m 且丢失 0.2s < 0.5s → 保留外推坐标。"""
    lost_short = _track("R1", -12.0, 0.0, vx=-2.0, vy=0.0, pos_uncertainty=0.3, lost_s=0.2)
    out = apply_guess_to_lost((lost_short,))
    assert (out[0].x, out[0].y) == (-12.0, 0.0)


def test_duration_gate_triggers_even_with_small_uncertainty() -> None:
    """时长门（主）：丢失 ≥0.5s（裁判中断线，连击已清零）后即使 σ 还小也切猜点。

    实战帧率下 σ 要 0.73s 才过 0.8m，而旧版 COAST=5 帧 @10Hz 只有
    0.5s 寿命——纯 σ 门在实战永远开不了，猜点是死代码。
    """
    lost_long = _track("R1", -12.0, 0.0, vx=-2.0, vy=0.0, pos_uncertainty=0.3, lost_s=0.6)
    out = apply_guess_to_lost((lost_long,))
    assert (out[0].x, out[0].y) == (-11.0, -5.0)


def test_guess_rewrite_preserves_lost_s_and_metadata() -> None:
    """猜点接管后 lost_s/σ/速度原样保留——下游还有用。"""
    lost = _track("R1", -12.0, 0.0, vx=-2.0, vy=0.0, pos_uncertainty=1.2, lost_s=0.9)
    out = apply_guess_to_lost((lost,))
    assert out[0].lost_s == 0.9
    assert out[0].pos_uncertainty == 1.2
    assert out[0].vx == -2.0
    assert out[0].phase is TrackPhase.LOST


def test_guess_points_table_has_every_role() -> None:
    for role in (Role.ONE, Role.TWO, Role.THREE, Role.FOUR, Role.SENTRY):
        assert guess_points_for(role), f"{role} 缺战略点表"
