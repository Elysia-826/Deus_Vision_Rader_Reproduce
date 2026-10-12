"""港科大式盲区猜点：LOST 槽在外推断气后，按速度方向挑最像的战略点。

数学同 hkustenterprize/RM2025-Radar-Algorithm 的 guess_pts.py：
  score = w · cos(速度方向, 指向点方向) + (1-w) · exp(-d · 距离衰减)
外推（卡尔曼 dead-reckoning）只管半秒级遮挡；猜点管的是更长时失联——
车钻进遮挡区几秒后，外推位置已经飘了，而兵种行为先验（工程去补给、
步兵上高地）比漂移的外推更接近真值。

点位表是 RMUC2026 规则手册增益点的近似坐标（场地中心系，+x 蓝侧），
粗到米级即可——猜点只要把车拉回正确城区，不追求米级精度。
2027 官方地图发布后按新图重填 GUESS_POINTS。
"""

from __future__ import annotations

import math
from typing import Final

from locate.types import FieldXY
from track.types import PublishedTrack, Role, TrackPhase

_GUESS_WEIGHT: Final[float] = 0.003  # HKUST 默认：几乎只看方向，距离只做平手裁决
_DISTANCE_DECAY: Final[float] = 0.1  # exp(-0.1·d)：15m 外的点基本只剩方向分

# 各兵种的候选战略点（场地中心系，米）。按 RMUC2026 增益机制近似。
# TODO(2027 地图)：官方尺寸图发布后按图重填。
_GUESS_POINTS: Final[dict[str, tuple[tuple[float, float], ...]]] = {
    "1": ((-12.5, 0.0), (-5.5, 0.0), (0.0, 0.0)),  # 英雄：对家基地/前哨、中央高地
    "2": ((12.0, 5.0), (12.0, -5.0), (12.5, 0.0)),  # 工程：资源站/补给区一带
    "3": ((0.0, 0.0), (-5.5, 0.0), (0.0, 6.0), (0.0, -6.0)),  # 步兵：中央高地/前哨/隧道口
    "4": ((0.0, 0.0), (-5.5, 0.0), (0.0, 6.0), (0.0, -6.0)),
    "S": ((0.0, 0.0), (6.0, 3.0), (6.0, -3.0), (-6.0, 3.0), (-6.0, -3.0)),  # 哨兵：巡逻环
}


def guess_field_xy(track: PublishedTrack) -> FieldXY | None:
    """LOST 槽的落点建议。vx/vy 是卡尔曼速度；解不出（无点）返回 None。

    红蓝同表：点位按「该兵种会去哪」定义，不分队。速度为零时退化为取最近点。
    """
    points = _GUESS_POINTS.get(track.label[-1])
    if not points:
        return None
    speed = math.hypot(track.vx, track.vy)
    best: tuple[float, float] | None = None
    best_score = -1e18
    for point in points:
        dx, dy = point[0] - track.x, point[1] - track.y
        distance = math.hypot(dx, dy)
        if speed > 1e-6:
            cos_sim = (track.vx * dx + track.vy * dy) / (speed * distance + 1e-8)
        else:
            cos_sim = 0.0
        score = _GUESS_WEIGHT * cos_sim + (1.0 - _GUESS_WEIGHT) * math.exp(-distance * _DISTANCE_DECAY)
        if score > best_score:
            best_score = score
            best = point
    return None if best is None else FieldXY(x=best[0], y=best[1])


def apply_guess_to_lost(publishes: tuple[PublishedTrack, ...]) -> tuple[PublishedTrack, ...]:
    """把 LOST 槽的坐标替换成猜点。CONFIRMED/TENTATIVE 原样保留。

    run_radar 里挂在 RADAR_GUESS=1 开关后面，配合 RADAR_SEND_LOST=1 基准可 A/B：
    猜点只改坐标，phase 仍是 LOST（画空心点、按 LOST 语义发裁判）。
    """
    out: list[PublishedTrack] = []
    for item in publishes:
        if item.phase is not TrackPhase.LOST:
            out.append(item)
            continue
        guess = guess_field_xy(item)
        if guess is None:
            out.append(item)
            continue
        out.append(
            PublishedTrack(
                label=item.label,
                team=item.team,
                x=guess.x,
                y=guess.y,
                phase=item.phase,
                bot_id=item.bot_id,
                vx=item.vx,
                vy=item.vy,
            )
        )
    return tuple(out)


def guess_points_for(role: Role) -> tuple[tuple[float, float], ...]:
    """测试与联调用：看某兵种的候选点。"""
    return _GUESS_POINTS.get(role.value, ())
