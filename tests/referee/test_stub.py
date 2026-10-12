from __future__ import annotations

import json

import pytest

from referee.commands import MapMark, mark_from_field
from referee.link import RefereeLink
from referee.stub import Judgement, StubReferee
from referee.transport import LoopbackPair
from track.types import SimRobot, Team

_E = 1  # 敌方工程槽（英雄=0，工程=1）


def _pair(own: Team) -> tuple[RefereeLink, StubReferee]:
    pair = LoopbackPair()
    link = RefereeLink(pair.radar, own_team=own)
    stub = StubReferee(pair.peer, own_team=own)
    return link, stub


def _engineer(x: float = 5.0, y: float = 3.0) -> tuple[SimRobot, ...]:
    return (SimRobot(name="Robot_Engineer_Red", team=Team.RED, x=x, y=y, z=0.2, label="R2"),)


def _marks(slot: int, x_m: float, y_m: float) -> tuple[MapMark | None, ...]:
    marks: list[MapMark | None] = [None] * 12
    marks[slot] = mark_from_field(x_m, y_m)
    return tuple(marks)


def test_rulebook_example() -> None:
    """规则手册 5.6.6 worked example，22 个 P 值逐点对照。

    5Hz 全准 → 前 5 发 P=1,3,6,10,15，第 3.0 秒（第 15 发）P=120；
    5Hz 全错 → 119.2,117.6,115.2,112,108；静默 1 秒 → 两次中断 103.2,97.6；
    复发两发全准 → 98.6,100.6。
    """
    link, stub = _pair(Team.BLUE)
    stub.set_truth(_engineer(5.0, 3.0))
    stub.step(now_s=0.0)  # 开赛，中断计时起点

    good = [round(0.2 * n, 3) for n in range(1, 16)]  # 0.2 … 3.0
    for t in good:
        assert link.send_map(_marks(_E, 5.0, 3.0), now_s=t) is True
        stub.step(now_s=t)
    assert stub.p_at(_E) == pytest.approx(120.0)
    for n in range(1, 6):
        assert stub.p_at(_E) == pytest.approx(120.0)  # 终点复述

    bad_ps = [119.2, 117.6, 115.2, 112.0, 108.0]
    for k, t in enumerate([3.2, 3.4, 3.6, 3.8, 4.0]):
        assert link.send_map(_marks(_E, 15.0, 3.0), now_s=t) is True  # d=10m，全错
        stub.step(now_s=t)
        assert stub.p_at(_E) == pytest.approx(bad_ps[k])

    stub.step(now_s=4.5)
    assert stub.p_at(_E) == pytest.approx(103.2)
    stub.step(now_s=5.0)
    assert stub.p_at(_E) == pytest.approx(97.6)

    link.send_map(_marks(_E, 5.0, 3.0), now_s=5.2)
    stub.step(now_s=5.2)
    assert stub.p_at(_E) == pytest.approx(98.6)
    link.send_map(_marks(_E, 5.0, 3.0), now_s=5.4)
    stub.step(now_s=5.4)
    assert stub.p_at(_E) == pytest.approx(100.6)

    assert stub.count_at(_E, Judgement.ACCURATE) == 17
    assert stub.count_at(_E, Judgement.ERROR) == 5
    assert stub.count_at(_E, Judgement.INTERRUPT) == 2


def test_silence_from_match_start_clamps_at_zero() -> None:
    link, stub = _pair(Team.BLUE)
    stub.step(now_s=0.0)
    stub.step(now_s=0.5)
    stub.step(now_s=1.0)
    assert stub.p_at(_E) == 0.0  # 从没发过包：中断连扣但 P 钳在 0
    assert stub.count_at(_E, Judgement.INTERRUPT) == 2


def test_zero_zero_mark_does_not_reset_interruption() -> None:
    """(0,0) 是「未发送」，不该让中断计时归零。"""
    link, stub = _pair(Team.BLUE)
    stub.set_truth(_engineer())
    stub.step(now_s=0.0)
    link.send_map((None,) * 12, now_s=0.2)
    stub.step(now_s=0.2)
    stub.step(now_s=0.6)  # 距开赛 0.6s ≥ 0.5s → 中断；若 (0,0) 重置了计时就不会触发
    assert stub.count_at(_E, Judgement.INTERRUPT) == 1


def test_marking_absent_robot_is_error() -> None:
    """场上没有这辆车还发坐标：服务器视角必然全错。"""
    link, stub = _pair(Team.BLUE)
    stub.set_truth(())  # 谁都不在
    stub.step(now_s=0.0)
    link.send_map(_marks(_E, 0.0, 0.0), now_s=0.2)
    stub.step(now_s=0.2)
    assert stub.count_at(_E, Judgement.ERROR) == 1
    assert stub.p_at(_E) == 0.0


def test_union_vulnerability_and_double_trigger() -> None:
    """两台同时易伤只计一次工会时间；双倍触发后 0x020E bit2 置位。"""
    link, stub = _pair(Team.BLUE)
    stub.set_truth(
        (
            SimRobot(name="Robot_Hero_Red", team=Team.RED, x=-1.0, y=1.0, z=0.4, label="R1"),
            SimRobot(name="Robot_Engineer_Red", team=Team.RED, x=5.0, y=3.0, z=0.2, label="R2"),
        )
    )
    stub.step(now_s=0.0)
    marks = _marks(0, -1.0, 1.0)[:1] + _marks(1, 5.0, 3.0)[1:]  # 两槽同时发
    for n in range(1, 16):
        t = round(0.2 * n, 3)
        assert link.send_map(marks, now_s=t)
        stub.step(now_s=t)
    assert stub.p_at(0) == pytest.approx(120.0)
    assert stub.p_at(1) == pytest.approx(120.0)
    for t in (3.2, 4.2, 5.2):  # 两台都在易伤：工会时间只走一份
        stub.step(now_s=t)
    summary = stub.summary()
    # 易伤从第 14 发（t=2.8，P=105）就开始：0.2+0.2+0.2+1.0+1.0 = 2.6s
    assert summary.union_vuln_s == pytest.approx(2.6)
    assert summary.vuln_s[0] == pytest.approx(2.6)
    assert summary.vuln_s[1] == pytest.approx(2.6)
    assert stub.radar_info_byte() & 0x04 == 0
    stub.trigger_double()  # 还没挣到机会：静默忽略
    assert stub.radar_info_byte() & 0x04 == 0

    # 持续标记 ~61s 挣到第一次双倍机会（P 顶到 150 钳住）
    t = 5.6
    while t < 66.0:
        assert link.send_map(marks, now_s=t)
        stub.step(now_s=t)
        t = round(t + 0.4, 3)
    assert stub.summary().double_chances_earned == 1
    stub.step(now_s=66.0)
    stub.trigger_double()
    assert stub.radar_info_byte() & 0x04 == 0x04
    stub.step(now_s=66.5)
    assert stub.summary().doubled_vuln_s == pytest.approx(0.5)


def test_log_frame_writes_parseable_records(tmp_path) -> None:
    link, stub = _pair(Team.BLUE)
    stub.set_truth(_engineer())
    stub.step(now_s=0.0)
    link.send_map(_marks(_E, 5.0, 3.0), now_s=0.2)
    stub.step(now_s=0.2)
    log = tmp_path / "referee_log.jsonl"
    stub.log_frame(log, frame=1, now_s=0.2)
    record = json.loads(log.read_text(encoding="utf-8"))
    assert record["frame"] == 1
    assert record["slots"]["E2"]["p"] == pytest.approx(1.0)
    assert record["slots"]["E2"]["sent"] is True
    assert record["slots"]["E2"]["judge"] == "accurate"
    assert "E1" not in record["slots"]  # 没真值也没发过的槽不写
