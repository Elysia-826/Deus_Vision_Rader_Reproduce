from __future__ import annotations

from detect.types import BBox, Detection, FrameResult, LinkedRobot, PatternPred
from locate.field import FIELD_X_MAX, FIELD_X_MIN, FIELD_Y_MAX, FIELD_Y_MIN
from locate.homography import homography_from_pairs
from locate.types import FieldXY, Pixel
from track.observe import _appearance, _on_field, observations_from_result
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
