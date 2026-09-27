"""用真值 name 数身份切换。按距离绑定，不用最近距离把对调算成两次命中。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from track.assign import min_cost_pairs
from track.types import PublishedTrack, SimRobot

SCORE_GATE_M: Final = 1.2
_REJECT: Final = 1.0e6


@dataclass(frozen=True, slots=True)
class Binding:
    label: str
    name: str
    distance_m: float


@dataclass(frozen=True, slots=True)
class ScoreFrame:
    gt: tuple[SimRobot, ...]
    pred: tuple[PublishedTrack, ...]


@dataclass(frozen=True, slots=True)
class NameCount:
    name: str
    switches: int
    misses: int


@dataclass(frozen=True, slots=True)
class IdentityReport:
    by_name: tuple[NameCount, ...]

    def switches(self, name: str) -> int:
        return _count(self, name).switches

    def misses(self, name: str) -> int:
        return _count(self, name).misses


def associate(gt: tuple[SimRobot, ...], pred: tuple[PublishedTrack, ...]) -> tuple[Binding, ...]:
    """门限 1.2 m，低于仿真最小间距，两辆真值抢不了同一个预测。"""
    if not gt or not pred:
        return ()
    cost = [[_pair_cost(robot, track) for track in pred] for robot in gt]
    pairs = min_cost_pairs(cost)
    bound: list[Binding] = []
    for gt_i, pred_i in pairs:
        robot = gt[gt_i]
        track = pred[pred_i]
        distance = _distance(robot, track)
        if distance <= SCORE_GATE_M:
            bound.append(Binding(label=track.label, name=robot.name, distance_m=distance))
    return tuple(bound)


def score_frames(frames: tuple[ScoreFrame, ...]) -> IdentityReport:
    """某个真值 name 绑到的槽位标签变了，该 name 计 1。第一次绑定不算切换。"""
    switches: dict[str, int] = {}
    misses: dict[str, int] = {}
    last_label: dict[str, str] = {}
    for frame in frames:
        bound_names: set[str] = set()
        for binding in associate(frame.gt, frame.pred):
            bound_names.add(binding.name)
            previous = last_label.get(binding.name)
            if previous is not None and previous != binding.label:
                switches[binding.name] = switches.get(binding.name, 0) + 1
            last_label[binding.name] = binding.label
        for robot in frame.gt:
            if robot.name not in bound_names:
                misses[robot.name] = misses.get(robot.name, 0) + 1
            switches.setdefault(robot.name, 0)
            misses.setdefault(robot.name, 0)
    names = tuple(sorted(set(switches) | set(misses)))
    return IdentityReport(
        by_name=tuple(
            NameCount(name=name, switches=switches.get(name, 0), misses=misses.get(name, 0))
            for name in names
        )
    )


def _pair_cost(robot: SimRobot, track: PublishedTrack) -> float:
    distance = _distance(robot, track)
    if distance > SCORE_GATE_M:
        return _REJECT
    return distance


def _distance(robot: SimRobot, track: PublishedTrack) -> float:
    return ((robot.x - track.x) ** 2 + (robot.y - track.y) ** 2) ** 0.5


def _count(report: IdentityReport, name: str) -> NameCount:
    for item in report.by_name:
        if item.name == name:
            return item
    return NameCount(name=name, switches=0, misses=0)
