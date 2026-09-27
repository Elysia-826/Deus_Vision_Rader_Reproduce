from __future__ import annotations

import pytest

from track.errors import TruthFieldError, UnmappedRobotName
from track.truth import sim_robot_from_mapping
from track.types import Team


def test_parse_keeps_name_team_and_z() -> None:
    robot = sim_robot_from_mapping(
        {"name": "Robot_3_Red", "team": "red", "x": 1, "y": 2, "z": 0.4}
    )
    assert robot.name == "Robot_3_Red"
    assert robot.team is Team.RED
    assert robot.x == pytest.approx(1.0)
    assert robot.y == pytest.approx(2.0)
    assert robot.z == pytest.approx(0.4)
    assert robot.label == "R3"


def test_missing_name_does_not_become_anonymous_xy() -> None:
    with pytest.raises(TruthFieldError) as caught:
        sim_robot_from_mapping({"team": "red", "x": 1, "y": 2, "z": 0.1})
    assert caught.value.key == "name"


def test_unmapped_name_includes_the_original() -> None:
    with pytest.raises(UnmappedRobotName) as caught:
        sim_robot_from_mapping(
            {"name": "Robot_Red_Alpha", "team": "red", "x": 1, "y": 2, "z": 0.1}
        )
    assert "Robot_Red_Alpha" in str(caught.value)
