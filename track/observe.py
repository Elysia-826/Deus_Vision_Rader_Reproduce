"""检测结果 → 场地观测。没有图案也保留车，身份不靠这一帧的分类字符串。"""

from __future__ import annotations

from typing import Final

from detect.types import FrameResult, LinkedRobot
from locate.homography import HomographyMap
from locate.ray_plane import foot_pixel, pixel_to_ground
from locate.camera import CameraPose
from locate.types import FieldXY, Pixel
from track.types import FieldObservation, Role, Team

_TEAM_BY_LABEL: Final[dict[str, Team]] = {"red": Team.RED, "blue": Team.BLUE}
_ROLE_BY_NAME: Final[dict[str, Role]] = {
    "1": Role.ONE,
    "2": Role.TWO,
    "3": Role.THREE,
    "4": Role.FOUR,
    "S": Role.SENTRY,
}


def observations_from_result(
    result: FrameResult,
    pose: CameraPose | None,
    homography: HomographyMap | None,
) -> tuple[FieldObservation, ...]:
    """底边像素走现有定位。单应优先，和 run_radar 原来的开关一致，这里不改标定。"""
    found: list[FieldObservation] = []
    for robot in result.robots:
        pixel = foot_pixel(robot.car.box)
        hit = _to_field(pixel, pose, homography)
        if hit is None:
            continue
        team, role, conf = _appearance(robot)
        found.append(
            FieldObservation(
                bot_id=robot.car.track_id,
                x=hit.x,
                y=hit.y,
                team=team,
                role=role,
                pattern_conf=conf,
            )
        )
    return tuple(found)


def _to_field(pixel: Pixel, pose: CameraPose | None, homography: HomographyMap | None) -> FieldXY | None:
    if homography is not None:
        return homography.pixel_to_field(pixel)
    if pose is None:
        return None
    return pixel_to_ground(pose, pixel)


def _appearance(robot: LinkedRobot) -> tuple[Team | None, Role | None, float]:
    best_conf = -1.0
    team: Team | None = None
    role: Role | None = None
    for armor in robot.armors:
        color = _TEAM_BY_LABEL.get(armor.label)
        if armor.pattern is None:
            if team is None and color is not None:
                team = color
            continue
        if armor.pattern.conf <= best_conf:
            continue
        best_conf = armor.pattern.conf
        team = color
        role = _ROLE_BY_NAME.get(armor.pattern.name)
    if best_conf < 0.0:
        return team, None, 0.0
    return team, role, best_conf
