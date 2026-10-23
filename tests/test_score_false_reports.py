"""虚报/漏报计数口径：按身份判，不按最近邻。

提示词锁死的两条：
- 槽名错但距离近（哪怕贴着别的车）→ 必须算虚报；
- 同名且 <2.5 m → 不算虚报。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

import score_false_reports as sfr  # noqa: E402


def _gt(name: str, team: str, x: float, y: float, *, visible: bool = True) -> sfr.GtEntry:
    """走真值解析同一套 name→label 逻辑（Robot_Infantry_Red_01 → R3）。"""
    robot = sfr.sim_robot_from_mapping(
        {"name": name, "team": team, "x": x, "y": y, "z": 0.2}
    )
    return sfr.GtEntry(label=robot.label, x=robot.x, y=robot.y, visible=visible)


def _pred(slot: str, x: float, y: float, state: str = "confirmed") -> sfr.PredEntry:
    return sfr.PredEntry(slot=slot, x=x, y=y, state=state)


def test_wrong_slot_near_another_car_is_false_report() -> None:
    """位置贴着某台车但槽名错了 → 虚报（最近邻口径会把它算成命中）。"""
    hero_red = _gt("Robot_Hero_Red", "red", 0.0, 0.0)
    counts = sfr.count_frame((hero_red,), (_pred("R3", 0.3, 0.0),))
    assert counts.false_reports == 1
    assert counts.correct == 0
    assert counts.false_slots == ("R3",)


def test_wrong_team_same_role_is_false_report() -> None:
    """同兵种但队伍反了（红槽贴蓝车）→ 身份对不上，虚报。"""
    infantry_blue = _gt("Robot_Infantry_Blue_01", "blue", 0.0, 0.0)
    counts = sfr.count_frame((infantry_blue,), (_pred("R3", 0.2, 0.0),))
    assert counts.false_reports == 1


def test_same_name_within_gate_is_not_false_report() -> None:
    """同名（同队同兵种）且 <2.5 m → 不算虚报，误差入账。"""
    infantry_red = _gt("Robot_Infantry_Red_01", "red", 0.0, 0.0)
    counts = sfr.count_frame((infantry_red,), (_pred("R3", 2.4, 0.0),))
    assert counts.false_reports == 0
    assert counts.correct == 1
    assert counts.errors == (2.4,)


def test_same_name_beyond_gate_is_false_report() -> None:
    """同名但这台车不在 2.5 m 内 → 坐标说的不是这台车，虚报。"""
    infantry_red = _gt("Robot_Infantry_Red_01", "red", 0.0, 0.0)
    counts = sfr.count_frame((infantry_red,), (_pred("R3", 2.6, 0.0),))
    assert counts.false_reports == 1


def test_slot_without_matching_gt_is_false_report() -> None:
    """真值里没有这个兵种（如哨兵）→ 该槽发布即虚报。"""
    hero_red = _gt("Robot_Hero_Red", "red", 0.0, 0.0)
    counts = sfr.count_frame((hero_red,), (_pred("RS", 0.1, 0.0),))
    assert counts.false_reports == 1


def test_visible_gt_without_same_name_slot_is_miss() -> None:
    """可见真值没有任何同名槽发布 → 漏报。"""
    infantry_red = _gt("Robot_Infantry_Red_01", "red", 0.0, 0.0, visible=True)
    counts = sfr.count_frame((infantry_red,), ())
    assert counts.misses == 1
    assert counts.false_reports == 0


def test_invisible_gt_is_not_miss() -> None:
    """不可见真值（蓝相机投影过滤掉）不计漏报。"""
    infantry_red = _gt("Robot_Infantry_Red_01", "red", 0.0, 0.0, visible=False)
    counts = sfr.count_frame((infantry_red,), ())
    assert counts.misses == 0


def test_same_name_slot_covers_miss_even_if_false_position() -> None:
    """同名槽发了（哪怕坐标错到算虚报）→ 该真值不算漏报。"""
    infantry_red = _gt("Robot_Infantry_Red_01", "red", 0.0, 0.0, visible=True)
    hero_red = _gt("Robot_Hero_Red", "red", 5.0, 5.0, visible=True)
    counts = sfr.count_frame((infantry_red, hero_red), (_pred("R3", 3.0, 3.0),))
    assert counts.false_reports == 1
    assert counts.misses == 1  # 只有 R1 真值漏掉


def test_parse_and_count_end_to_end() -> None:
    """identity_log 文本 → 解析 → 计数全链路；可见性由注入函数决定。"""
    line = json.dumps(
        {
            "frame": 7,
            "gt": [
                {"name": "Robot_Infantry_Red_01", "team": "red", "x": 0.0, "y": 0.0, "z": 0.2},
                {"name": "Robot_Hero_Blue", "team": "blue", "x": 4.0, "y": 0.0, "z": 0.4},
            ],
            "pred": [
                {"slot": "R1", "x": 0.4, "y": 0.0, "state": "confirmed"},
                {"slot": "B1", "x": 4.1, "y": 0.1, "state": "confirmed"},
            ],
        }
    )
    frames = sfr.parse_frames([line], is_visible=lambda robot: True)
    report = sfr.count_frames(frames)
    # R1 贴着红步兵但槽名是英雄 → 虚报；B1 正确。
    assert report.false_reports == 1
    assert report.correct == 1
    # 红步兵没有 R3 同名槽 → 漏报；蓝英雄有 B1 → 不漏。
    assert report.misses == 1
    assert len(report.errors) == 1
    assert abs(report.errors[0] - (0.1**2 + 0.1**2) ** 0.5) < 1e-6
