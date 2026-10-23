from __future__ import annotations

import math

import pytest

from track.cascade import ALLY_HINT_BONUS, ALLY_HINT_PENALTY, CascadeMatchTracker, _active_cost
from track.types import FieldObservation, PatternStatus, Role, Team, TrackPhase


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
    """连着 DEAD_FRAMES 帧装甲全 dead → R3 判死停发；复活后重新确认再上屏。

    只断言 R3：_confirm 里的 B1 同时在正常外推，COAST=8 帧 @dt=1s
    下它还会发布几帧——那是设计行为，不属于本测试关注点。
    """
    tracker = CascadeMatchTracker()
    _confirm(tracker)
    for _ in range(5):
        out = tracker.step(
            (_obs(1, 1.0, 0.0, Team.RED, Role.THREE, armor_dead=True),), dt_s=1.0
        )
    dead_frame = tracker.step(
        (_obs(1, 1.0, 0.0, Team.RED, Role.THREE, armor_dead=True),), dt_s=1.0
    )
    assert all(item.label != "R3" for item in dead_frame)
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


def test_ally_hint_redirects_misread_observation_to_own_slot() -> None:
    """哨兵盟友提示：盟友附近被误读成敌色的观测不进敌方槽。

    几何：敌我步兵贴脸（R3 在 2.2、己方 B3 在 2.5，正在交战），
    哨兵报己方车在 (2.35, 0)。己方车的灯条被误读成红色、图案 3。
    没有提示时观测进 R3（距离近 + 灯色同阵营）；有提示时位置先验
    把观测判给 B3——灯条误读不再污染敌方槽的坐标。

    注意 B3 开新槽的场景不可能发生：INACTIVE 槽开槽门要求灯色
    AGREE，误读观测永远开不出己方槽。提示只在槽间竞争中起效。
    旧版符号反了（拿 obs.team 比 slot.team）：灯条误读成红色时反而
    给红槽发奖金，本测试在那个版本下必然失败。
    """
    obs = _obs(9, 2.3, 0.1, Team.RED, Role.THREE)  # bot_id=9：新 BoT-SORT track
    hint = ((2.35, 0.0),)  # 哨兵说己方车在这里（距观测 0.11m）

    def _seed(tracker: CascadeMatchTracker) -> None:
        for _ in range(3):
            tracker.step(
                (
                    _obs(1, 2.2, 0.0, Team.RED, Role.THREE),
                    _obs(7, 2.5, 0.0, Team.BLUE, Role.THREE),
                ),
                dt_s=1.0,
            )

    # 对照组：没有提示 → 观测进 R3（灯色同阵营 + 距离近）
    plain = CascadeMatchTracker()
    _seed(plain)
    out_plain = plain.step((obs,), dt_s=1.0)
    r3_plain = {item.label: item for item in out_plain}["R3"]
    b3_plain = {item.label: item for item in out_plain}["B3"]
    assert r3_plain.x > 2.25, "无提示时观测被敌方槽 R3 吸走（旧行为）"
    assert b3_plain.x == pytest.approx(2.5, abs=0.05)

    # 实验组：own_team=BLUE + 提示 → 观测判给己方 B3，R3 不被污染
    tracker = CascadeMatchTracker(own_team=Team.BLUE)
    _seed(tracker)
    tracker.set_ally_hints(hint)
    out = tracker.step((obs,), dt_s=1.0)
    by_label = {item.label: item for item in out}
    r3 = by_label["R3"]
    b3 = by_label["B3"]
    assert r3.x < 2.25, f"R3 位置不该被盟友附近的误读观测污染: {r3.x}"
    assert b3.x < 2.5, "观测应判给己方 B3（位置先验压过灯色误读）"


def test_ally_hint_cost_moves_enemy_up_and_own_down() -> None:
    """符号单测：同一观测带提示后，敌方槽代价 +0.4、己方槽 −0.3。

    直接锁数值方向——regression 时符号再反，这里最先红。
    """
    tracker = CascadeMatchTracker(own_team=Team.BLUE)
    for _ in range(3):
        tracker.step(
            (
                _obs(1, 2.2, 0.0, Team.RED, Role.THREE),
                _obs(7, 2.4, 0.0, Team.BLUE, Role.THREE),
            ),
            dt_s=1.0,
        )
    r3_slot = next(s for s in tracker._slots if s.slot.label == "R3")
    b3_slot = next(s for s in tracker._slots if s.slot.label == "B3")
    obs = _obs(9, 2.3, 0.1, Team.RED, Role.THREE)
    hint = ((2.35, 0.0),)

    cost_r3_with = _active_cost(r3_slot, obs, PatternStatus.AGREE, hint, Team.BLUE)
    cost_r3_without = _active_cost(r3_slot, obs, PatternStatus.AGREE, (), Team.BLUE)
    cost_b3_with = _active_cost(b3_slot, obs, PatternStatus.AGREE, hint, Team.BLUE)
    cost_b3_without = _active_cost(b3_slot, obs, PatternStatus.AGREE, (), Team.BLUE)

    assert cost_r3_with == pytest.approx(cost_r3_without + ALLY_HINT_PENALTY)
    assert cost_b3_with == pytest.approx(cost_b3_without - ALLY_HINT_BONUS)


def test_ally_hint_requires_own_team_to_act() -> None:
    """没有 own_team 的提示无从判定敌我——提示必须失效（不崩、不改行为）。"""
    tracker = CascadeMatchTracker()
    for x in (2.0, 2.1, 2.2):
        tracker.step((_obs(1, x, 0.0, Team.RED, Role.THREE),), dt_s=1.0)
    tracker.set_ally_hints(((2.35, 0.0),))
    out = tracker.step((_obs(9, 2.3, 0.1, Team.RED, Role.THREE),), dt_s=1.0)
    by_label = {item.label: item for item in out}
    assert by_label["R3"].x > 2.25, "own_team 缺失时提示不生效，保持旧行为（观测进 R3）"


def test_lost_publishes_carry_growing_lost_s() -> None:
    """lost_s 随丢失帧数累积（每帧 +dt）；幽灵发布窗 0.1-1.1s@10Hz。

    双寿命：COAST=5 帧后进幽灵态（只发布不吸收），GHOST=12 帧注销。
    猜点的 0.5s 门在幽灵窗（帧 5-11）里真实可开——不像旧版 σ 门
    在 0.5s 吸收寿命里永远到不了 0.8m。
    """
    tracker = CascadeMatchTracker()
    tracker.step((_obs(1, 1.0, 0.0, Team.RED, Role.THREE),), dt_s=1.0)
    tracker.step((_obs(1, 1.1, 0.0, Team.RED, Role.THREE),), dt_s=1.0)
    lost_s_seen: list[float] = []
    for _ in range(11):  # miss_count 1..11，全部发布 LOST
        out = tracker.step((), dt_s=0.1)
        r3 = next(item for item in out if item.label == "R3")
        lost_s_seen.append(r3.lost_s)
    assert lost_s_seen == [pytest.approx(0.1 * n) for n in range(1, 12)]
    # 第 12 帧 miss_count=12 → 幽灵期满注销，不再发布
    assert all(item.label != "R3" for item in tracker.step((), dt_s=0.1))
    # 复活：观测回来，lost_s 归零（槽注销过，按状态机重开 TENTATIVE）
    revived = tracker.step((_obs(1, 1.2, 0.0, Team.RED, Role.THREE),), dt_s=0.1)
    r3_back = next(item for item in revived if item.label == "R3")
    assert r3_back.lost_s == 0.0
    assert r3_back.phase is TrackPhase.TENTATIVE


def test_ghost_never_absorbs_foreign_observations() -> None:
    """幽灵只发布不吸收：COAST 过线后，别的车走到幽灵 2.5m 门内也不许被偷。

    直接拉长吸收寿命的版本里，僵尸槽在 0.5-0.8s 把路过车的观测吸走，
    基准 9.6s → 5.6s。幽灵 REJECT 保住了匹配行为。
    """
    tracker = CascadeMatchTracker()
    tracker.step((_obs(1, 3.0, 0.0, Team.RED, Role.THREE),), dt_s=1.0)
    tracker.step((_obs(1, 3.1, 0.0, Team.RED, Role.THREE),), dt_s=1.0)
    # 丢 6 帧 → R3 进幽灵态（miss_count=6 ≥ COAST=5），卡尔曼停在 ~3.1
    for _ in range(6):
        tracker.step((), dt_s=0.1)
    # 另一辆车（蓝方英雄）路过幽灵位置：必须开自己的 B1 新槽，不能进幽灵
    out = tracker.step((_obs(5, 3.2, 0.0, Team.BLUE, Role.ONE),), dt_s=0.1)
    by_label = {item.label: item for item in out}
    r3 = by_label["R3"]
    assert r3.bot_id == 1, "幽灵 R3 不该吸收别人的观测"
    assert abs(r3.x - 3.1) < 0.2, f"幽灵位置不该被路过的车拖走: {r3.x}"
    assert r3.phase is TrackPhase.LOST
    assert "B1" in by_label, "路过的车必须开自己的新槽"


def test_ghost_reabsorbs_returning_robot_with_matching_identity() -> None:
    """本尊回归（灯条+图案都对上幽灵槽的身份）→ 幽灵复活为 CONFIRMED。

    幽灵占着自己的槽，本尊不可能开新槽——回归路径只有这一条：
    身份吻合的观测直接吸收，miss/lost_s 清零。
    """
    tracker = CascadeMatchTracker()
    tracker.step((_obs(1, 3.0, 0.0, Team.RED, Role.THREE),), dt_s=1.0)
    tracker.step((_obs(1, 3.1, 0.0, Team.RED, Role.THREE),), dt_s=1.0)
    for _ in range(6):  # 幽灵态（miss_count=6 ≥ COAST=5）
        tracker.step((), dt_s=0.1)
    out = tracker.step((_obs(9, 1.0, 0.0, Team.RED, Role.THREE),), dt_s=0.1)
    r3_items = [item for item in out if item.label == "R3"]
    assert len(r3_items) == 1
    assert r3_items[0].phase is TrackPhase.CONFIRMED
    assert r3_items[0].lost_s == 0.0
    assert r3_items[0].x < 1.5, "幽灵位置应被回归观测拉回真实位置"
