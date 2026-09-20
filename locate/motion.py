"""场地平面上的折线巡逻。Blender 侧用同一公式搬车。"""

from __future__ import annotations

from locate.types import FieldXY

SPEED_MPS = 1.5


def point_on_loop(waypoints: tuple[FieldXY, ...], t: float, *, speed_mps: float = SPEED_MPS) -> FieldXY:
    """沿闭合折线匀速走。t 秒，speed 米/秒。"""
    if len(waypoints) < 2:
        raise ValueError("need at least 2 waypoints")  # noqa: GENERIC_ERR_OK
    if speed_mps <= 0.0:
        return waypoints[0]
    edges: list[tuple[FieldXY, FieldXY, float]] = []
    total = 0.0
    count = len(waypoints)
    for i, start in enumerate(waypoints):
        end = waypoints[(i + 1) % count]
        length = ((end.x - start.x) ** 2 + (end.y - start.y) ** 2) ** 0.5
        edges.append((start, end, length))
        total += length
    if total <= 0.0:
        return waypoints[0]
    dist = (speed_mps * t) % total
    walked = 0.0
    for start, end, length in edges:
        if dist <= walked + length or length <= 0.0:
            frac = 0.0 if length <= 0.0 else (dist - walked) / length
            return FieldXY(x=start.x + frac * (end.x - start.x), y=start.y + frac * (end.y - start.y))
        walked += length
    return waypoints[0]
