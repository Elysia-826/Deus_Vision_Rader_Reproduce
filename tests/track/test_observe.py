from __future__ import annotations

import pytest

from detect.types import BBox, Detection, FrameResult, LinkedRobot, PatternPred
from locate.field import FIELD_X_MAX, FIELD_X_MIN, FIELD_Y_MAX, FIELD_Y_MIN
from locate.homography import homography_from_pairs
from locate.types import FieldXY, Pixel
from track.observe import _appearance, _apply_own_team_prior, _on_field, observations_from_result
from track.types import Role, Team


def _homography():
    """任意一个可用的单应：像素四角 → 场地四角，够把底边映进场内。"""
    pixels = (Pixel(0, 0), Pixel(280, 0), Pixel(280, 150), Pixel(0, 150))
    fields = (
        FieldXY(FIELD_X_MIN, FIELD_Y_MAX),
        FieldXY(FIELD_X_MAX, FIELD_Y_MAX),
        FieldXY(FIELD_X_MAX, FIELD_Y_MIN),
        FieldXY(FIELD_X_MIN, FIELD_Y_MIN),
    )
    return homography_from_pairs(pixels, fields, ("tl", "tr", "br", "bl"))


def test_on_field_rejects_exploded_ray() -> None:
    assert _on_field(FieldXY(x=0.0, y=0.0))
    assert _on_field(FieldXY(x=15.9, y=9.4))
    assert not _on_field(FieldXY(x=-1101.0, y=-132.0))
    assert not _on_field(FieldXY(x=16.1, y=0.0))


def test_dead_pattern_borrows_red_lamp() -> None:
    car = Detection(label="robot", conf=0.9, box=BBox(0, 0, 80, 80))
    dead = Detection(
        label="dead",
        conf=0.8,
        box=BBox(10, 10, 20, 20),
        pattern=PatternPred(name="S", conf=0.82),
    )
    red = Detection(label="red", conf=0.7, box=BBox(12, 12, 22, 22))
    team, role, conf = _appearance(LinkedRobot(car=car, armors=(dead, red)))
    assert team is Team.RED
    assert role is Role.SENTRY
    assert conf > 0.8


def test_only_dead_armor_has_no_team() -> None:
    car = Detection(label="robot", conf=0.9, box=BBox(0, 0, 80, 80))
    dead = Detection(
        label="dead",
        conf=0.8,
        box=BBox(10, 10, 20, 20),
        pattern=PatternPred(name="S", conf=0.82),
    )
    team, role, conf = _appearance(LinkedRobot(car=car, armors=(dead,)))
    assert team is None
    assert role is Role.SENTRY
    assert conf > 0.8


def _car(box: tuple[int, int, int, int], armors: tuple[Detection, ...]) -> LinkedRobot:
    return LinkedRobot(
        car=Detection(label="robot", conf=0.9, box=BBox(*box)), armors=armors
    )


def _live_armor(label: str = "red") -> Detection:
    return Detection(label=label, conf=0.8, box=BBox(10, 10, 20, 20), pattern=PatternPred(name="3", conf=0.8))


def _dead_armor() -> Detection:
    return Detection(label="dead", conf=0.8, box=BBox(10, 10, 20, 20), pattern=PatternPred(name="3", conf=0.8))


def test_all_dead_armors_tag_the_observation_not_drop_it() -> None:
    """死亡只是标志位：全 dead 装甲的车照常产出观测，时序裁决在跟踪器。"""
    result = FrameResult(
        robots=(
            _car((0, 0, 280, 150), (_dead_armor(), _dead_armor())),
            _car((0, 0, 280, 150), (_dead_armor(), _live_armor())),  # 混合
            _car((0, 0, 280, 150), ()),  # 无装甲
        )
    )
    observations = observations_from_result(result, None, _homography(), None)
    assert len(observations) == 3  # 一个都不丢
    assert observations[0].armor_dead is True  # 全 dead
    assert observations[1].armor_dead is False  # 混合：不算死
    assert observations[2].armor_dead is False  # 无装甲：不判死


def test_no_armor_robot_still_observed() -> None:
    """没检出装甲的车不受死亡标志影响（和旧行为一致）。"""
    result = FrameResult(robots=(_car((0, 0, 280, 150), ()),))
    observations = observations_from_result(result, None, _homography(), None)
    assert len(observations) == 1


# ---------- 多装甲多数投票（max_det=3 之后一块车可带三块读数） ----------


def _armor(label: str, name: str, conf: float) -> Detection:
    return Detection(
        label=label, conf=0.8, box=BBox(10, 10, 20, 20), pattern=PatternPred(name=name, conf=conf)
    )


def test_pattern_majority_beats_single_high_conf_outlier() -> None:
    """两块说 3、一块 0.65 说 1 → 采信 3。旧的 max-conf 会采信 1。"""
    car = Detection(label="robot", conf=0.9, box=BBox(0, 0, 80, 80))
    robot = LinkedRobot(
        car=car,
        armors=(
            _armor("red", "3", 0.55),
            _armor("red", "3", 0.60),
            _armor("red", "1", 0.65),  # 单板高置信 outlier
        ),
    )
    team, role, conf = _appearance(robot)
    assert role is Role.THREE
    assert conf == pytest.approx(0.60)  # 获胜名字里的最高 conf
    assert team is Team.RED


def test_lamp_majority_beats_first_frame_outlier() -> None:
    """灯条颜色也多数票：第一块误读成红、后两块蓝 → 蓝。旧的取首块会跟错。"""
    car = Detection(label="robot", conf=0.9, box=BBox(0, 0, 80, 80))
    robot = LinkedRobot(
        car=car,
        armors=(
            _armor("red", "3", 0.8),   # 单板灯条误读
            _armor("blue", "3", 0.7),
            _armor("blue", "3", 0.75),
        ),
    )
    team, role, conf = _appearance(robot)
    assert team is Team.BLUE
    assert role is Role.THREE


def test_lamp_tie_falls_back_to_first_seen() -> None:
    """1红1蓝平票 → 回退首块（保持旧行为的确定性）。"""
    car = Detection(label="robot", conf=0.9, box=BBox(0, 0, 80, 80))
    robot = LinkedRobot(
        car=car,
        armors=(_armor("red", "3", 0.7), _armor("blue", "3", 0.7)),
    )
    team, role, _ = _appearance(robot)
    assert team is Team.RED


def test_pattern_tie_falls_back_to_top_conf() -> None:
    """两名字平票（1:1）→ 取置信度高的那个名字。"""
    car = Detection(label="robot", conf=0.9, box=BBox(0, 0, 80, 80))
    robot = LinkedRobot(
        car=car,
        armors=(_armor("red", "3", 0.50), _armor("red", "1", 0.90)),
    )
    _team, role, conf = _appearance(robot)
    assert role is Role.ONE
    assert conf == pytest.approx(0.90)


def test_dead_lamp_does_not_vote_color() -> None:
    """dead 不投颜色票——全 dead 板带图案时灯色只能来自活板（无活板则 None）。"""
    car = Detection(label="robot", conf=0.9, box=BBox(0, 0, 80, 80))
    robot = LinkedRobot(car=car, armors=(_dead_armor(), _dead_armor()))
    team, role, _ = _appearance(robot)
    assert team is None
    assert role is Role.THREE


# ---------- 本方颜色先验（雷达只看一边，知道自己是哪方） ----------


def test_own_team_prior_corrects_far_half_misread() -> None:
    """蓝方半场深处（x>3m）低置信的「红色」→ 修正为蓝（仿真 71% 灯条误读率）。"""
    assert _apply_own_team_prior(Team.RED, 0.5, x_m=5.0, own_team=Team.BLUE) is Team.BLUE
    assert _apply_own_team_prior(Team.BLUE, 0.4, x_m=-5.0, own_team=Team.RED) is Team.RED


def test_own_team_prior_respects_high_confidence() -> None:
    """图案置信度 ≥0.7：灯条可能真的对了（对方车开进来了），不覆盖。"""
    assert _apply_own_team_prior(Team.RED, 0.9, x_m=5.0, own_team=Team.BLUE) is Team.RED


def test_own_team_prior_inactive_in_midfield() -> None:
    """中线 ±3m 内不修正——那里敌方车本来就常见。"""
    assert _apply_own_team_prior(Team.RED, 0.4, x_m=1.0, own_team=Team.BLUE) is Team.RED
    assert _apply_own_team_prior(Team.RED, 0.4, x_m=-1.0, own_team=Team.BLUE) is Team.RED


def test_own_team_prior_noop_without_team_or_own() -> None:
    """无先验 / 灯条无色 / 阵营已一致 → 原样返回。"""
    assert _apply_own_team_prior(Team.RED, 0.4, x_m=5.0, own_team=None) is Team.RED
    assert _apply_own_team_prior(None, 0.4, x_m=5.0, own_team=Team.BLUE) is None
    assert _apply_own_team_prior(Team.BLUE, 0.4, x_m=5.0, own_team=Team.BLUE) is Team.BLUE
