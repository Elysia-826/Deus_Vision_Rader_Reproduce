"""港科大式盲区猜点：外推失效后才把坐标拉回战略城区。

数学同 hkustenterprize/RM2025-Radar-Algorithm 的 guess_pts.py：
  score = w · cos(速度方向, 指向点方向) + (1-w) · exp(-d · 距离衰减)

触发条件（丢失时长门，2026-10-22 改）：
  lost_s ≥ GUESS_TRIGGER_LOST_S（0.5s，裁判中断线）或 σ ≥ 0.8m。
  短遮挡（<0.5s）时外推比固定战略点准得多——实测「一进 LOST
  就猜」让易伤秒归零。0.5s 后连击已被中断判清零，外推误差随
  t² 发散，战略城区先验开始反超。
  协方差是副条件：转弯自适应噪声放大时 σ 可能提前过线，同样该切。
  （旧版只有 σ 门——按现有过程噪声 σ 要 0.73s 才过 0.8m，而
  COAST=5 帧 @10Hz 只有 0.5s 寿命，门永远开不了，形同虚设。）

点位表是 RM2027 场地特征的近似坐标（场地中心系，+x 蓝侧）。
2027 官方 CAD 精确坐标发布后按图重填 TODO。
"""

from __future__ import annotations

import math
from typing import Final

from locate.types import FieldXY
from track.types import PublishedTrack, Role, TrackPhase

_GUESS_WEIGHT: Final[float] = 0.003
_DISTANCE_DECAY: Final[float] = 0.1

# 触发门（主）：丢失时长。0.5s = 裁判中断线——连击已清零，
# 再发漂移外推只会吃 ERROR；战略点至少有机会落在准确圈内。
GUESS_TRIGGER_LOST_S: Final[float] = 0.5

# 触发门（副）：卡尔曼 σ 超过准确线 0.8m 时外推已不可信。
# 转弯自适应噪声（_BOOST_K=2）下可能比时长门先到。
GUESS_TRIGGER_UNCERTAINTY_M: Final[float] = 0.8

# RM2027 各兵种的候选战略点（场地中心系，米）。
# TODO(2027 CAD)：官方精确坐标发布后按图重填。
_GUESS_POINTS: Final[dict[str, tuple[tuple[float, float], ...]]] = {
    "1": ((3.0, 0.0), (-3.0, 0.0), (11.0, 5.0), (-11.0, -5.0)),
    "2": ((3.0, 0.0), (-3.0, 0.0), (0.0, 6.0), (0.0, -6.0)),
    "3": ((0.0, 0.0), (5.5, 0.0), (-5.5, 0.0), (0.0, 6.0), (0.0, -6.0)),
    "4": ((0.0, 0.0), (5.5, 0.0), (-5.5, 0.0), (0.0, 6.0), (0.0, -6.0)),
    "S": ((0.0, 0.0), (6.0, 3.0), (6.0, -3.0), (-6.0, 3.0), (-6.0, -3.0)),
}


def guess_field_xy(track: PublishedTrack) -> FieldXY | None:
    """LOST 槽的落点建议。vx/vy 是卡尔曼速度；解不出（无点）返回 None。"""
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
    """把外推失效的 LOST 槽坐标替换成猜点。

    丢失时长门：0.5s 内外推还在准确圈（实测中位 0.17m）→ 不覆盖。
    0.5s 后连击已被中断清零、外推误差随 t² 发散 → 战略城区先验接管。
    σ 门是副条件：转弯自适应噪声下 σ 可能提前过 0.8m。
    """
    out: list[PublishedTrack] = []
    for item in publishes:
        if item.phase is not TrackPhase.LOST:
            out.append(item)
            continue
        if (
            item.lost_s < GUESS_TRIGGER_LOST_S
            and item.pos_uncertainty < GUESS_TRIGGER_UNCERTAINTY_M
        ):
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
                pos_uncertainty=item.pos_uncertainty,
                lost_s=item.lost_s,
            )
        )
    return tuple(out)


def guess_points_for(role: Role) -> tuple[tuple[float, float], ...]:
    """测试与联调用：看某兵种的候选点。"""
    return _GUESS_POINTS.get(role.value, ())