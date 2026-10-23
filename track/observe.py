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
    """多装甲多数投票。max_det=3 后一辆车最多带三块装甲读数。

    同一辆车的三块板应当显示同一兵种/同色灯条；单板误读（如两块说 3、
    一块 0.65 说 1）被多数票压掉。逐板投票而不是取最高置信——
    「置信度最高的那块」恰好就是误读时没有任何机制能纠正它。

    灯条颜色：dead 不投票。多数胜；平票回退首块（保持旧行为）。
    兵种：同名多数胜；平票取平票组里置信度最高的名字。
    兵种置信度：获胜名字各板里的最高 conf（分类器对该名字的最佳读数）。
    """
    lamp_counts: dict[Team, int] = {}
    pattern_counts: dict[str, int] = {}
    pattern_best_conf: dict[str, float] = {}
    pattern_color: dict[str, Team | None] = {}
    first_lamp: Team | None = None
    for armor in robot.armors:
        color = _TEAM_BY_LABEL.get(armor.label)
        if color is not None:
            lamp_counts[color] = lamp_counts.get(color, 0) + 1
            if first_lamp is None:
                first_lamp = color
        if armor.pattern is None:
            continue
        name = armor.pattern.name
        pattern_counts[name] = pattern_counts.get(name, 0) + 1
        if armor.pattern.conf > pattern_best_conf.get(name, -1.0):
            pattern_best_conf[name] = armor.pattern.conf
            pattern_color[name] = color  # 跟着该名字的最高置信板走
    lamp_team = _majority_team(lamp_counts, first_lamp)
    if not pattern_counts:
        return lamp_team, None, 0.0
    best_name = _majority_name(pattern_counts, _top_conf_name(pattern_best_conf))
    if best_name is None:
        return lamp_team, None, 0.0
    conf = pattern_best_conf[best_name]
    # 队伍归属：灯色多数票优先（多数票就是为压单板误读而生）；
    # 全 dead 没有灯色票时才退回获胜图案板的颜色。
    team = lamp_team or pattern_color.get(best_name)
    role = _ROLE_BY_NAME.get(best_name)
    return team, role, conf


def _majority_team(counts: dict[Team, int], fallback: Team | None) -> Team | None:
    """灯色众数；空/平票回退 fallback（首块），保持旧行为的确定性。"""
    if not counts:
        return fallback
    top = max(counts.values())
    winners = [key for key, count in counts.items() if count == top]
    return winners[0] if len(winners) == 1 else fallback


def _majority_name(counts: dict[str, int], fallback: str | None) -> str | None:
    """兵种名众数；平票回退最高置信名（旧 max-conf 行为的等价物）。"""
    if not counts:
        return fallback
    top = max(counts.values())
    winners = [key for key, count in counts.items() if count == top]
    return winners[0] if len(winners) == 1 else fallback


def _top_conf_name(best_conf: dict[str, float]) -> str | None:
    if not best_conf:
        return None
    return max(best_conf, key=lambda name: best_conf[name])
