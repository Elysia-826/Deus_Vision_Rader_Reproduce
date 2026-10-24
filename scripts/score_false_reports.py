# ─── How to run ───
#     python scripts/score_false_reports.py [identity_log.jsonl ...]
# ──────────────────

"""按身份数虚报/漏报——发布门的审计口径，和 score_identity_log.py 的最近邻互补。

虚报：pred 的槽名（R3/B1/BS…）必须对上同队同兵种的真值，且这台车在 2.5 m 内。
对不上、或 2.5 m 内没有这台车——哪怕位置贴着另一台车——都记 1 次虚报。
漏报：蓝相机可见的真值里，这一帧没有任何同名槽发布，记 1 次漏报。

会上裁判的虚报：默认 RADAR_SEND_LOST=0 时只有 confirmed 进 0x0305
（TENTATIVE 永不进，LOST 要 include_lost）——referee_bound_false 只数
confirmed，并把它拆成互斥三类（classify_false）：
  a. team_swap      2.5 m 内有同兵种、反队的真值（R1 贴着蓝英雄）
  b. wrong_position 同名真值在场但 ≥2.5 m
  c. no_robot       场上没有这个槽对应的车

score_identity_log.py 的 matched 是最近邻口径：槽名错了但位置贴近某台车
会被算成命中——那不能当发布门的成功线。本文件按身份判，才是。

多个日志路径逐个单独报数，不混总分（固定回放集每套序列一张表）。
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from locate.camera import camera_from_path  # noqa: E402
from locate.types import WorldXYZ  # noqa: E402
from track.truth import sim_robot_from_mapping  # noqa: E402
from track.types import SimRobot  # noqa: E402

DEFAULT_LOG = _ROOT / "scene" / "locate_out" / "identity_log.jsonl"
BLUE_JSON = _ROOT / "locate" / "calib" / "Camera_Radar_Blue.json"

# 可见性口径与 score_identity_log.py 完全一致：GT 投到蓝相机画面
# （±100 px 余量）内才算可见。漏报只对可见真值计数。
IMG_W, IMG_H = 4096, 3000
MARGIN = 100
GATE_M: Final = 2.5  # 同名真值必须在这个距离内，发布才算对

# confirmed 虚报的互斥三类分解（优先级 a > b > c，一条发布只进一类）。
# 默认 RADAR_SEND_LOST=0 时只有 confirmed 会进 0x0305（TENTATIVE 永不进，
# LOST 要 include_lost），所以「会上裁判的虚报」= confirmed 虚报。
FALSE_TEAM_SWAP: Final = "team_swap"  # a. 队伍认错：2.5 m 内有同兵种、反队的真值
FALSE_WRONG_POSITION: Final = "wrong_position"  # b. 同名真值在场，但 ≥2.5 m
FALSE_NO_ROBOT: Final = "no_robot"  # c. 场上没有这个槽对应的车


def _visible(pose, x: float, y: float, z: float) -> bool:
    try:
        pixel = pose.project(WorldXYZ(x, y, z))
    except Exception:  # noqa: BLE001 — 投不进画面的点就是不可见，不区分原因
        return False
    return -MARGIN <= pixel.u <= IMG_W + MARGIN and -MARGIN <= pixel.v <= IMG_H + MARGIN


@dataclass(frozen=True, slots=True)
class GtEntry:
    """一条真值。label 与槽名同一字符串空间（R1..R4/RS/B1..B4/BS）。"""

    label: str
    x: float
    y: float
    visible: bool


@dataclass(frozen=True, slots=True)
class PredEntry:
    slot: str
    x: float
    y: float
    state: str


@dataclass(frozen=True, slots=True)
class FrameCounts:
    false_reports: int
    misses: int
    correct: int
    errors: tuple[float, ...]
    false_slots: tuple[str, ...]
    correct_state_errors: tuple[tuple[str, float], ...]  # (state, err) 每条正确发布
    false_kind_state: tuple[tuple[str, str], ...]  # (kind, state) 每条虚报


def classify_false(gt: Sequence[GtEntry], pred: PredEntry) -> str:
    """一条虚报的互斥三类（a > b > c，一条发布只进一类）。

    只对已判虚报的 pred 调用（同名真值 2.5 m 内就不是虚报了）：
    a. team_swap —— 2.5 m 内有同兵种、反队的真值（R1 贴着蓝英雄）；
    b. wrong_position —— 同名真值在场但 ≥2.5 m（同名存在优先于 c）；
    c. no_robot —— 场上没有这个槽对应的车。
    """
    for g in gt:
        if (
            g.label != pred.slot
            and g.label[1:] == pred.slot[1:]
            and g.label[0] != pred.slot[0]
            and ((pred.x - g.x) ** 2 + (pred.y - g.y) ** 2) ** 0.5 < GATE_M
        ):
            return FALSE_TEAM_SWAP
    if any(g.label == pred.slot for g in gt):
        return FALSE_WRONG_POSITION
    return FALSE_NO_ROBOT


@dataclass(frozen=True, slots=True)
class Report:
    frames: int
    gt_total: int
    gt_visible: int
    published: int
    false_reports: int
    misses: int
    correct: int
    errors: tuple[float, ...]
    false_by_slot: tuple[tuple[str, int], ...]
    false_by_state: tuple[tuple[str, int], ...]
    state_counts: tuple[tuple[str, int], ...]
    referee_bound_false: int  # 会上裁判的虚报 = confirmed 虚报（默认发送路径）
    confirmed_false_kinds: tuple[tuple[str, int], ...]  # confirmed 虚报 a/b/c 分解
    err_by_state: tuple[tuple[str, float, float, int], ...]  # 每态正确发布的 (median, max, n)


def count_frame(gt: Sequence[GtEntry], preds: Sequence[PredEntry]) -> FrameCounts:
    """一帧的虚报/漏报。虚报按身份判：同队同兵种真值在 2.5 m 内才算对。"""
    errors: list[float] = []
    false_slots: list[str] = []
    correct_state_errors: list[tuple[str, float]] = []
    false_kind_state: list[tuple[str, str]] = []
    for pred in preds:
        best = min(
            (((pred.x - g.x) ** 2 + (pred.y - g.y) ** 2) ** 0.5 for g in gt if g.label == pred.slot),
            default=None,
        )
        if best is not None and best < GATE_M:
            errors.append(best)
            correct_state_errors.append((pred.state, best))
        else:
            false_slots.append(pred.slot)
            false_kind_state.append((classify_false(gt, pred), pred.state))
    published_names = {pred.slot for pred in preds}
    misses = sum(1 for g in gt if g.visible and g.label not in published_names)
    return FrameCounts(
        false_reports=len(false_slots),
        misses=misses,
        correct=len(preds) - len(false_slots),
        errors=tuple(errors),
        false_slots=tuple(false_slots),
        correct_state_errors=tuple(correct_state_errors),
        false_kind_state=tuple(false_kind_state),
    )


def _state_err_stats(
    pairs: Sequence[tuple[str, float]]
) -> tuple[tuple[str, float, float, int], ...]:
    by_state: dict[str, list[float]] = {}
    for state, err in pairs:
        by_state.setdefault(state, []).append(err)
    out: list[tuple[str, float, float, int]] = []
    for state in sorted(by_state):
        ordered = sorted(by_state[state])
        out.append((state, ordered[len(ordered) // 2], ordered[-1], len(ordered)))
    return tuple(out)


def count_frames(frames: Sequence[tuple[Sequence[GtEntry], Sequence[PredEntry]]]) -> Report:
    """逐帧计数后汇总。每套序列单独调用一次，不混总分。"""
    gt_total = gt_visible = published = 0
    false_reports = misses = correct = 0
    errors: list[float] = []
    false_by_slot: dict[str, int] = {}
    false_by_state: dict[str, int] = {}
    state_counts: dict[str, int] = {}
    correct_state_errors: list[tuple[str, float]] = []
    confirmed_false_kinds: dict[str, int] = {}
    for gt, preds in frames:
        counts = count_frame(gt, preds)
        gt_total += len(gt)
        gt_visible += sum(1 for g in gt if g.visible)
        published += len(preds)
        false_reports += counts.false_reports
        misses += counts.misses
        correct += counts.correct
        errors.extend(counts.errors)
        correct_state_errors.extend(counts.correct_state_errors)
        for slot in counts.false_slots:
            false_by_slot[slot] = false_by_slot.get(slot, 0) + 1
        for kind, state in counts.false_kind_state:
            false_by_state[state] = false_by_state.get(state, 0) + 1
            if state == "confirmed":
                confirmed_false_kinds[kind] = confirmed_false_kinds.get(kind, 0) + 1
        for pred in preds:
            state_counts[pred.state] = state_counts.get(pred.state, 0) + 1
    ordered = sorted(errors)
    return Report(
        frames=len(frames),
        gt_total=gt_total,
        gt_visible=gt_visible,
        published=published,
        false_reports=false_reports,
        misses=misses,
        correct=correct,
        errors=tuple(ordered),
        false_by_slot=tuple(sorted(false_by_slot.items())),
        false_by_state=tuple(sorted(false_by_state.items())),
        state_counts=tuple(sorted(state_counts.items())),
        referee_bound_false=false_by_state.get("confirmed", 0),
        confirmed_false_kinds=tuple(sorted(confirmed_false_kinds.items())),
        err_by_state=_state_err_stats(correct_state_errors),
    )


def parse_frames(
    lines: Iterable[str], is_visible: Callable[[SimRobot], bool]
) -> tuple[tuple[tuple[GtEntry, ...], tuple[PredEntry, ...]], ...]:
    """identity_log.jsonl 文本 → [(gt, preds), …]。visible 由注入的判定函数算。"""
    frames: list[tuple[tuple[GtEntry, ...], tuple[PredEntry, ...]]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        gt = tuple(
            GtEntry(
                label=robot.label,
                x=robot.x,
                y=robot.y,
                visible=is_visible(robot),
            )
            for robot in (sim_robot_from_mapping(item) for item in raw.get("gt", []))
        )
        preds = tuple(
            PredEntry(
                slot=str(item["slot"]),
                x=float(item["x"]),
                y=float(item["y"]),
                state=str(item.get("state", "")),
            )
            for item in raw.get("pred", [])
        )
        frames.append((gt, preds))
    return tuple(frames)


def _fmt_pair(items: Sequence[tuple[str, int]]) -> str:
    return " ".join(f"{key}={value}" for key, value in items) if items else "-"


def _fmt_state_err(items: Sequence[tuple[str, float, float, int]]) -> str:
    if not items:
        return "-"
    return " | ".join(
        f"{state} med={median:.3f}m max={maximum:.3f}m n={count}"
        for state, median, maximum, count in items
    )


def report_lines(log: Path, report: Report) -> list[str]:
    lines = [f"=== {log} ==="]
    lines.append(f"frames={report.frames}")
    lines.append(f"gt_total={report.gt_total} gt_visible={report.gt_visible}")
    lines.append(
        f"published={report.published} ({_fmt_pair(report.state_counts)})"
    )
    lines.append(f"false_reports={report.false_reports}")
    lines.append(f"  by_slot: {_fmt_pair(report.false_by_slot)}")
    lines.append(f"  by_state: {_fmt_pair(report.false_by_state)}")
    lines.append(
        f"  err_by_state (correct only): {_fmt_state_err(report.err_by_state)}"
    )
    lines.append(f"referee_bound_false (confirmed only)={report.referee_bound_false}")
    lines.append(
        f"  confirmed kinds: {_fmt_pair(report.confirmed_false_kinds)}"
    )
    lines.append(f"misses={report.misses}")
    lines.append(f"correct={report.correct}")
    if report.errors:
        ordered = report.errors
        lines.append(
            f"err median={ordered[len(ordered) // 2]:.3f}m max={ordered[-1]:.3f}m "
            f"n={len(ordered)}"
        )
    else:
        lines.append("err median=- max=- n=0")
    return lines


def main() -> None:
    logs = [Path(arg) for arg in sys.argv[1:]] or [DEFAULT_LOG]
    pose = camera_from_path(BLUE_JSON)

    def is_visible(robot: SimRobot) -> bool:
        return _visible(pose, robot.x, robot.y, robot.z)

    for log in logs:
        frames = parse_frames(log.read_text(encoding="utf-8").splitlines(), is_visible)
        report = count_frames(frames)
        for line in report_lines(log, report):
            print(line)


if __name__ == "__main__":
    main()
