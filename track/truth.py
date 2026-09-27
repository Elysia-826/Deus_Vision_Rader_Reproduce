"""把 live_meta 的 robots 收成带名字的真值。缺 name/team/z 不再退化成 FieldXY。"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import assert_never

from track.errors import TruthFieldError, UnmappedRobotName
from track.names import role_from_name
from track.types import SimRobot, Team

JsonScalar = str | int | float | bool | None


def parse_team(text: str) -> Team:
    match text:
        case "red":
            return Team.RED
        case "blue":
            return Team.BLUE
        case _:
            raise TruthFieldError(key="team", reason=text)


def sim_robot_from_mapping(item: Mapping[str, JsonScalar]) -> SimRobot:
    name = _require_str(item, "name")
    team = parse_team(_require_str(item, "team"))
    role = role_from_name(name)
    return SimRobot(
        name=name,
        team=team,
        x=_require_float(item, "x"),
        y=_require_float(item, "y"),
        z=_require_float(item, "z"),
        label=f"{'R' if team is Team.RED else 'B'}{role.value}",
    )


def parse_sim_robots(items: Sequence[Mapping[str, JsonScalar]]) -> tuple[SimRobot, ...]:
    return tuple(sim_robot_from_mapping(item) for item in items)


def _require_str(item: Mapping[str, JsonScalar], key: str) -> str:
    value = item.get(key)
    if not isinstance(value, str) or value == "":
        raise TruthFieldError(key=key, reason="missing or not a string")
    return value


def _require_float(item: Mapping[str, JsonScalar], key: str) -> float:
    value = item.get(key)
    match value:
        case bool():
            raise TruthFieldError(key=key, reason="bool is not a coordinate")
        case int() | float():
            return float(value)
        case str() | None:
            raise TruthFieldError(key=key, reason="missing or not a number")
        case unreachable:
            assert_never(unreachable)
