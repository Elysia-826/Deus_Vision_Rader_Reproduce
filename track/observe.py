"""检测结果 → 场地观测。没有图案也保留车，身份不靠这一帧的分类字符串。"""

from __future__ import annotations

from typing import Final

from detect.types import FrameResult, LinkedRobot
from locate.camera import CameraPose
from locate.field import FIELD_X_MAX, FIELD_X_MIN, FIELD_Y_MAX, FIELD_Y_MIN
from locate.homography import HomographyMap
from locate.ray_mesh import FieldMesh
from locate.ray_plane import foot_pixel, pixel_to_ground
from locate.types import FieldXY, Pixel
from track.types import FieldObservation, Role, Team

_FIELD_SLACK_M: Final = 2.0

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
    mesh: FieldMesh | None = None,
    own_team: Team | None = None,
) -> tuple[FieldObservation, ...]:
    """底边像素乘现成定位。单应优先；否则打场地网格，没有网格退回 z=0。

    死亡只是附带信息：全 dead 装甲的车照常产出观测（armor_dead=True），
    由跟踪器做时序裁决（连续 DEAD_FRAMES 帧才真死）。逐帧丢弃会饿死
    跟踪器——实测仿真里 28% 的车帧装甲全误闪 dead。

    own_team：本方颜色先验。装甲 YOLO 远距灯条颜色 ~40% 误读（仿真里
    蓝灯条 71% 被读成红）。车在本方半场深处且图案置信度低时，灯条说
    「对方颜色」更可能是误读——位置先验修正为own_team。
    """
    found: list[FieldObservation] = []
    for robot in result.robots:
        pixel = foot_pixel(robot.car.box)
        hit = _to_field(pixel, pose, homography, mesh)
        if hit is None or not _on_field(hit):
            continue
        team, role, conf = _appearance(robot)
        team = _apply_own_team_prior(team, conf, hit.x, own_team)
        found.append(
            FieldObservation(
                bot_id=robot.car.track_id,
                x=hit.x,
                y=hit.y,
                team=team,
                role=role,
                pattern_conf=conf,
                armor_dead=bool(robot.armors) and all(armor.label == "dead" for armor in robot.armors),
            )
        )
    return tuple(found)


_OWN_HALF_M: Final[float] = 3.0  # 距中线超过 3m 才算「深处」
_PRIOR_MAX_CONF: Final[float] = 0.7  # 图案置信度低于此值时才允许先验修正


def _apply_own_team_prior(
    team: Team | None,
    conf: float,
    x_m: float,
    own_team: Team | None,
) -> Team | None:
    """位置先验修正：车在本方半场深处 + 灯条说「对方色」+ 图案低置信 → 修正为本方。

    误读统计（2026 仿真）：蓝灯条 71% 被读成红。一个蓝车在本方半场深处
    被读成「红色」→ 如果不修正，观测会去匹配红方槽（R1/R2/R3）→
    位置/身份链全错。先验是白送的：雷达站只看一边，知道自己是谁。
    """
    if own_team is None or team is None or team is own_team:
        return team  # 无先验 / 已一致 / 无颜色 → 不动
    if conf >= _PRIOR_MAX_CONF:
        return team  # 图案置信度高 → 灯条可能真的对了，不覆盖
    if own_team is Team.BLUE and x_m > _OWN_HALF_M:
        return own_team  # 蓝方半场深处的「红色」→ 修正为蓝
    if own_team is Team.RED and x_m < -_OWN_HALF_M:
        return own_team  # 红方半场深处的「蓝色」→ 修正为红
    return team


def _to_field(
    pixel: Pixel,
    pose: CameraPose | None,
    homography: HomographyMap | None,
    mesh: FieldMesh | None = None,
) -> FieldXY | None:
    if homography is not None:
        return homography.pixel_to_field(pixel)
    if pose is None:
        return None
    if mesh is not None:
        return mesh.pixel_to_field(pose, pixel)
    return pixel_to_ground(pose, pixel)


def _on_field(hit: FieldXY) -> bool:
    return (
        FIELD_X_MIN - _FIELD_SLACK_M <= hit.x <= FIELD_X_MAX + _FIELD_SLACK_M
        and FIELD_Y_MIN - _FIELD_SLACK_M <= hit.y <= FIELD_Y_MAX + _FIELD_SLACK_M
    )


def _appearance(robot: LinkedRobot) -> tuple[Team | None, Role | None, float]:
    best_conf = -1.0
    team: Team | None = None
    role: Role | None = None
    lamp_team: Team | None = None
    for armor in robot.armors:
        color = _TEAM_BY_LABEL.get(armor.label)
        if color is not None and lamp_team is None:
            lamp_team = color
        if armor.pattern is None:
            continue
        if armor.pattern.conf <= best_conf:
            continue
        best_conf = armor.pattern.conf
        team = color
        role = _ROLE_BY_NAME.get(armor.pattern.name)
    if team is None:
        team = lamp_team
    if best_conf < 0.0:
        return team, None, 0.0
    return team, role, best_conf
