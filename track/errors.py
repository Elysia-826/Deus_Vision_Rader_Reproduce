"""跟踪与仿真真值的边界错误。缺字段或名字对不上兵种，都不能当成匿名点继续打分。"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class TruthFieldError(Exception):
    """live_meta 里某条机器人缺身份或坐标。"""

    key: str
    reason: str

    def __str__(self) -> str:
        return f"live_meta robots.{self.key}: {self.reason}"


@dataclass(frozen=True, slots=True)
class UnmappedRobotName(Exception):
    """名字里没有唯一兵种号。拒绝猜一张对照表。"""

    name: str

    def __str__(self) -> str:
        return f"robot name has no single role token, refusing to guess: {self.name}"
