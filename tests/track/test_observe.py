from __future__ import annotations

from detect.types import BBox, Detection, LinkedRobot, PatternPred
from locate.types import FieldXY
from track.observe import _appearance, _on_field
from track.types import Role, Team


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
