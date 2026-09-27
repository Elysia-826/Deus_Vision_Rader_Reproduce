"""固定兵种槽上的值对象。槽名创建后不变，当帧图案不是身份。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import assert_never


class Team(StrEnum):
    RED = "red"
    BLUE = "blue"


class Role(StrEnum):
    ONE = "1"
    TWO = "2"
    THREE = "3"
    FOUR = "4"
    SENTRY = "S"


class TrackPhase(StrEnum):
    INACTIVE = "inactive"
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    LOST = "lost"


class PatternStatus(StrEnum):
    ABSENT = "absent"
    AGREE = "agree"
    CONFLICT = "conflict"


@dataclass(frozen=True, slots=True)
class SlotId:
    """红/蓝 × 1/2/3/4/哨兵。前哨 Q 不进运动池。"""

    team: Team
    role: Role

    @property
    def label(self) -> str:
        return f"{team_prefix(self.team)}{self.role.value}"


def team_prefix(team: Team) -> str:
    match team:
        case Team.RED:
            return "R"
        case Team.BLUE:
            return "B"
        case unreachable:
            assert_never(unreachable)


def all_slots() -> tuple[SlotId, ...]:
    return tuple(SlotId(team, role) for team in Team for role in Role)


@dataclass(frozen=True, slots=True)
class FieldObservation:
    """一辆已定位的车。图案和颜色都可以空。"""

    bot_id: int | None
    x: float
    y: float
    team: Team | None
    role: Role | None
    pattern_conf: float


@dataclass(frozen=True, slots=True)
class PublishedTrack:
    """这一帧要画到小地图上的槽。LOST 仍发布，表示正在外推。"""

    label: str
    team: Team
    x: float
    y: float
    phase: TrackPhase
    bot_id: int | None


@dataclass(frozen=True, slots=True)
class SimRobot:
    """仿真真值。name 是身份，xy 只是位置。"""

    name: str
    team: Team
    x: float
    y: float
    z: float
    label: str
