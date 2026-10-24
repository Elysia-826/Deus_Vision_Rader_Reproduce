"""虚报/漏报计数口径：按身份判，不按最近邻。

提示词锁死的两条：
- 槽名错但距离近（哪怕贴着别的车）→ 必须算虚报；
- 同名且 <2.5 m → 不算虚报。

confirmed 虚报的互斥三类分解（会上裁判的虚报 = confirmed）：
- a 近距反队（同兵种）→ team_swap；
- b 同名在场但 ≥2.5 m → wrong_position；
- c 场上没有这个槽对应的车 → no_robot；
- 一条发布只能进一类。

confirmed 标记的裁判 d 分级（grade_confirmed，只数 confirmed）：
- 0.79 m → accurate；0.80 m → semi；1.60 m → error；
- 场上没有同名真值 → error（d 无定义）。

confirmed 发布的混淆表（stick_class，只数 confirmed）：
- 贴着反队同兵种 → 贴错，记「发布槽 → 实际槽」一对；
- 贴着同名且 <2.5 m → 贴对，不进混淆表；
- 2.5 m 内没有任何真值 → 附近没车，单独计数；
- 一条发布不能同时进两类。
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


def test_classify_team_swap_near_opposite_team_same_role() -> None:
    """近距反队（同兵种）必须算 a——同名真值同时在场（b 条件也成立）也只进 a。"""
    blue_hero = _gt("Robot_Hero_Blue", "blue", 0.0, 0.0)
    red_hero = _gt("Robot_Hero_Red", "red", 8.0, 0.0)
    kind = sfr.classify_false((blue_hero, red_hero), _pred("R1", 0.4, 0.0))
    assert kind == sfr.FALSE_TEAM_SWAP


def test_classify_wrong_position_same_name_beyond_gate() -> None:
    """同名真值在场但 ≥2.5 m 必须算 b。"""
    red_infantry = _gt("Robot_Infantry_Red_01", "red", 5.0, 0.0)
    kind = sfr.classify_false((red_infantry,), _pred("R3", 0.0, 0.0))
    assert kind == sfr.FALSE_WRONG_POSITION


def test_classify_no_robot_when_gt_missing() -> None:
    """场上没有这个槽对应的车必须算 c。"""
    red_hero = _gt("Robot_Hero_Red", "red", 3.0, 3.0)
    kind = sfr.classify_false((red_hero,), _pred("RS", 0.1, 0.0))
    assert kind == sfr.FALSE_NO_ROBOT


def test_one_publish_enters_exactly_one_class() -> None:
    """互斥：三类各一条时每条只进一类，虚报总数对得上。"""
    blue_hero = _gt("Robot_Hero_Blue", "blue", 0.0, 0.0)
    red_hero = _gt("Robot_Hero_Red", "red", 8.0, 0.0)
    red_infantry = _gt("Robot_Infantry_Red_01", "red", 5.0, 5.0)
    preds = (
        _pred("R1", 0.4, 0.0),  # a：贴着蓝英雄（红英雄在场但远——b 条件同时成立）
        _pred("R3", 0.0, 0.0),  # b：红步兵在场但 7 m 远
        _pred("RS", -3.0, 0.0),  # c：场上没有哨兵
    )
    counts = sfr.count_frame((blue_hero, red_hero, red_infantry), preds)
    kinds = [kind for kind, _state in counts.false_kind_state]
    assert counts.false_reports == 3
    assert sorted(kinds) == sorted(
        [sfr.FALSE_TEAM_SWAP, sfr.FALSE_WRONG_POSITION, sfr.FALSE_NO_ROBOT]
    )
    assert len(set(kinds)) == 3  # 没有一条进两类


def test_referee_bound_false_counts_confirmed_only() -> None:
    """会上裁判的虚报只数 confirmed；lost/tentative 默认不进 0x0305。"""
    frames = [
        ((_gt("Robot_Infantry_Red_01", "red", 5.0, 5.0),), (_pred("R4", 0.0, 0.0),)),
        ((_gt("Robot_Hero_Red", "red", 5.0, 5.0),), (_pred("B2", 0.0, 0.0, "lost"),)),
        ((_gt("Robot_Hero_Blue", "blue", 5.0, 5.0),), (_pred("B1", 0.0, 0.0, "tentative"),)),
    ]
    report = sfr.count_frames(frames)
    assert report.false_reports == 3
    assert report.referee_bound_false == 1
    assert dict(report.confirmed_false_kinds) == {sfr.FALSE_NO_ROBOT: 1}


def test_err_by_state_splits_correct_errors() -> None:
    """每态误差中位/最差只统计该态的正确发布。"""
    red_infantry = _gt("Robot_Infantry_Red_01", "red", 0.0, 0.0)
    frames = [
        (
            (red_infantry,),
            (
                _pred("R3", 1.0, 0.0, "confirmed"),
                _pred("R3", 2.0, 0.0, "lost"),
            ),
        )
    ]
    report = sfr.count_frames(frames)
    stats = {state: (median, maximum, n) for state, median, maximum, n in report.err_by_state}
    assert stats["confirmed"] == (1.0, 1.0, 1)
    assert stats["lost"] == (2.0, 2.0, 1)


def test_grade_accurate_below_0_8m() -> None:
    """d = 0.79 m → 准确档。"""
    red_infantry = _gt("Robot_Infantry_Red_01", "red", 0.0, 0.0)
    grade, d = sfr.grade_confirmed((red_infantry,), _pred("R3", 0.79, 0.0))
    assert grade == sfr.GRADE_ACCURATE
    assert d is not None and abs(d - 0.79) < 1e-9


def test_grade_semi_at_0_8m() -> None:
    """d = 0.80 m → 半准确档（0.8 ≤ d < 1.6）。"""
    red_infantry = _gt("Robot_Infantry_Red_01", "red", 0.0, 0.0)
    grade, _d = sfr.grade_confirmed((red_infantry,), _pred("R3", 0.80, 0.0))
    assert grade == sfr.GRADE_SEMI


def test_grade_error_at_1_6m() -> None:
    """d = 1.60 m → 错误档（d ≥ 1.6）。"""
    red_infantry = _gt("Robot_Infantry_Red_01", "red", 0.0, 0.0)
    grade, _d = sfr.grade_confirmed((red_infantry,), _pred("R3", 1.60, 0.0))
    assert grade == sfr.GRADE_ERROR


def test_grade_error_without_same_name_gt() -> None:
    """场上没有同名真值 → 错误档，d 无定义。"""
    red_hero = _gt("Robot_Hero_Red", "red", 3.0, 3.0)
    grade, d = sfr.grade_confirmed((red_hero,), _pred("R3", 0.1, 0.0))
    assert grade == sfr.GRADE_ERROR
    assert d is None


def test_d_line_counts_confirmed_only_and_splits_errors() -> None:
    """准线只收 confirmed；错误档拆「同名 ≥1.6 m」和「没有这台车」。"""
    red_infantry = _gt("Robot_Infantry_Red_01", "red", 0.0, 0.0)
    red_hero = _gt("Robot_Hero_Red", "red", 4.0, 4.0)
    frames = [
        (
            (red_infantry, red_hero),
            (
                _pred("R3", 0.79, 0.0),  # accurate
                _pred("R3", 1.0, 0.0),  # semi
                _pred("R3", 3.0, 0.0),  # error：同名在场但 ≥1.6 m
                _pred("R4", 0.0, 0.0),  # error：场上没有步兵 4
                _pred("R3", 1.0, 0.0, "lost"),  # 非 confirmed，不进准线
            ),
        )
    ]
    report = sfr.count_frames(frames)
    assert report.confirmed_graded == 4
    assert dict(report.grade_counts) == {
        sfr.GRADE_ACCURATE: 1,
        sfr.GRADE_SEMI: 1,
        sfr.GRADE_ERROR: 2,
    }
    assert report.error_same_name == 1
    assert report.error_no_robot == 1
    assert report.grade_d_n == 3  # 无此车那条没有定义 d


def test_stick_wrong_opposite_team_same_role_records_pair() -> None:
    """贴着反队同兵种的车 → 贴错，记「发布槽 → 实际槽」一对。"""
    blue_hero = _gt("Robot_Hero_Blue", "blue", 0.0, 0.0)
    stick, actual = sfr.stick_class((blue_hero,), _pred("R1", 0.4, 0.0))
    assert stick == sfr.STICK_WRONG
    assert actual == "B1"
    report = sfr.count_frames((((blue_hero,), (_pred("R1", 0.4, 0.0),)),))
    assert report.stick_wrong == 1
    assert report.confusion_pairs == (("R1", "B1", 1),)
    assert dict(report.wrong_split) == {sfr.WRONG_SAME_ROLE_OPP_TEAM: 1}


def test_stick_right_same_name_within_gate() -> None:
    """贴着同名车且 <2.5 m → 贴对，不进混淆表。"""
    red_infantry = _gt("Robot_Infantry_Red_01", "red", 0.0, 0.0)
    stick, actual = sfr.stick_class((red_infantry,), _pred("R3", 2.0, 0.0))
    assert stick == sfr.STICK_RIGHT
    assert actual == "R3"
    report = sfr.count_frames((((red_infantry,), (_pred("R3", 2.0, 0.0),)),))
    assert report.stick_right == 1
    assert report.stick_wrong == 0
    assert report.confusion_pairs == ()


def test_stick_none_when_no_gt_within_gate() -> None:
    """2.5 m 内没有任何真值 → 附近没车，不进混淆对。"""
    red_hero = _gt("Robot_Hero_Red", "red", 3.0, 0.0)
    stick, actual = sfr.stick_class((red_hero,), _pred("R3", 0.0, 0.0))
    assert stick == sfr.STICK_NONE
    assert actual is None


def test_one_publish_enters_exactly_one_stick_class() -> None:
    """三类互斥：贴错/贴对/附近没车各一条 + 一条 lost 不进——总数对得上。"""
    blue_hero = _gt("Robot_Hero_Blue", "blue", 0.0, 0.0)
    red_infantry = _gt("Robot_Infantry_Red_01", "red", 6.0, 0.0)
    red_hero = _gt("Robot_Hero_Red", "red", 6.0, 4.0)
    preds = (
        _pred("R1", 0.4, 0.0),  # 贴错：R1 → B1
        _pred("R3", 6.2, 0.0),  # 贴对：贴红步兵
        _pred("RS", 0.0, 3.0),  # 附近没车：离任何真值都 ≥3 m
        _pred("R3", 0.1, 0.1, "lost"),  # 非 confirmed，不进混淆表
    )
    report = sfr.count_frames(((((blue_hero, red_infantry, red_hero)), preds),))
    assert report.confirmed_graded == 3
    assert report.stick_wrong == 1
    assert report.stick_right == 1
    assert report.stick_none == 1
    assert report.stick_wrong + report.stick_right + report.stick_none == report.confirmed_graded


def test_wrong_split_covers_three_kinds_exclusively() -> None:
    """贴错构成三类互斥：同兵种反队 / 同队异兵种 / 又反队又异兵种。"""
    frames = [
        (
            (
                _gt("Robot_Hero_Blue", "blue", 0.0, 0.0),
                _gt("Robot_Hero_Red", "red", 6.0, 0.0),
                _gt("Robot_Infantry_Blue_01", "blue", 6.0, 6.0),
            ),
            (
                _pred("R1", 0.4, 0.0),  # →B1 同兵种反队
                _pred("R3", 6.2, 0.0),  # →R1 同队异兵种
                _pred("R1", 6.2, 6.0),  # →B3 又反队又异兵种
            ),
        )
    ]
    report = sfr.count_frames(frames)
    assert report.stick_wrong == 3
    assert dict(report.wrong_split) == {
        sfr.WRONG_SAME_ROLE_OPP_TEAM: 1,
        sfr.WRONG_SAME_TEAM_DIFF_ROLE: 1,
        sfr.WRONG_BOTH_DIFF: 1,
    }
    # 并列时按「发布槽,实际槽」字典序稳定排序，保证可复现
    assert report.confusion_pairs == (
        ("R1", "B1", 1),
        ("R1", "B3", 1),
        ("R3", "R1", 1),
    )
