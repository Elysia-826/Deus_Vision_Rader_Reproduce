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

confirmed 标记的裁判 d 分级准线（grade_confirmed，只数 confirmed）：
  accurate  d < 0.8 m    semi  0.8 ≤ d < 1.6 m
  error     d ≥ 1.6 m 或场上没有这台车
阈值 import referee/stub.py（规则手册 5.6.6 stub 复刻同源）——不用 2.5 m
当「对」。

confirmed 发布的混淆表（stick_class，只数 confirmed）：在全部真值里找
2.5 m 内最近的一台车（不管同名不同名）——贴错记「发布槽 → 实际槽」一对
（槽名保持 R1/B3/RS 原样），贴对不进表，附近没车单独计数；一条发布只进
一类，不用 0.8 m 再分一次。

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
from referee.stub import ACCURATE_M, SEMI_ACCURATE_M  # noqa: E402
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

# confirmed 发布的裁判 d 分级（阈值 import referee/stub.py，和规则手册 5.6.6
# 的 stub 复刻同源——不再用 2.5 m 当「对」，对的就是 <0.8 m）。
GRADE_ACCURATE: Final = "accurate"  # d < 0.8 m
GRADE_SEMI: Final = "semi"  # 0.8 m ≤ d < 1.6 m
GRADE_ERROR: Final = "error"  # d ≥ 1.6 m，或场上没有这台车（d 无定义）

# confirmed 发布贴着谁（混淆表口径）：2.5 m 内最近的一台真值，不管同名不同名。
# 一条发布只进一类；不用 0.8 m 再分一次（d 分级表已有那张尺子）。
STICK_WRONG: Final = "stick_wrong"  # 贴错：最近车槽名 ≠ 发布槽名，记「发布槽→实际槽」
STICK_RIGHT: Final = "stick_right"  # 贴对：最近车就是同名车且 <2.5 m
STICK_NONE: Final = "stick_none"  # 附近没车：2.5 m 内没有任何真值
# 贴错对的互斥三类构成（槽名保持 R1/B3/RS 原样，不翻译兵种）
WRONG_SAME_ROLE_OPP_TEAM: Final = "same_role_opp_team"  # 同兵种反队（R1→B1）
WRONG_SAME_TEAM_DIFF_ROLE: Final = "same_team_diff_role"  # 同队异兵种（R3→R1）
WRONG_BOTH_DIFF: Final = "both_diff"  # 又反队又异兵种（R1→B3）


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


def grade_confirmed(gt: Sequence[GtEntry], pred: PredEntry) -> tuple[str, float | None]:
    """confirmed 发布按裁判 d 分级（和 referee/stub.py 同源阈值）。

    只数 confirmed——lost/tentative 默认不进 0x0305，不混进准线。
    返回 (档, d)；场上没有同名真值时 d 无定义，直接判错误档。
    """
    best = min(
        (((pred.x - g.x) ** 2 + (pred.y - g.y) ** 2) ** 0.5 for g in gt if g.label == pred.slot),
        default=None,
    )
    if best is None:
        return GRADE_ERROR, None
    if best < ACCURATE_M:
        return GRADE_ACCURATE, best
    if best < SEMI_ACCURATE_M:
        return GRADE_SEMI, best
    return GRADE_ERROR, best


def stick_class(gt: Sequence[GtEntry], pred: PredEntry) -> tuple[str, str | None]:
    """confirmed 发布贴着哪台真值：2.5 m 内最近的一台（不管同名不同名）。

    贴对 → (STICK_RIGHT, 同名槽)；贴错 → (STICK_WRONG, 实际贴着的槽)，
    记「发布槽 → 实际槽」一对；附近没车 → (STICK_NONE, None)。一条发布
    只进一类。距离并列时按槽名字典序取，保证可复现。
    """
    nearest = min(
        ((((pred.x - g.x) ** 2 + (pred.y - g.y) ** 2) ** 0.5, g.label) for g in gt),
        default=None,
    )
    if nearest is None or nearest[0] >= GATE_M:
        return STICK_NONE, None
    _distance, label = nearest
    if label == pred.slot:
        return STICK_RIGHT, label
    return STICK_WRONG, label


def wrong_pair_kind(from_slot: str, to_slot: str) -> str:
    """贴错对的互斥三类构成。贴错保证两槽不同名，三类覆盖全部情况。"""
    team_diff = to_slot[0] != from_slot[0]
    role_diff = to_slot[1:] != from_slot[1:]
    if team_diff and not role_diff:
        return WRONG_SAME_ROLE_OPP_TEAM
    if role_diff and not team_diff:
        return WRONG_SAME_TEAM_DIFF_ROLE
    return WRONG_BOTH_DIFF


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
    confirmed_graded: int  # 进 d 准线的 confirmed 条数
    grade_counts: tuple[tuple[str, int], ...]  # accurate / semi / error 条数
    error_same_name: int  # 错误档里同名在场但 ≥1.6 m
    error_no_robot: int  # 错误档里场上没有这台车
    grade_d_median: float | None  # 中位（只算有定义的 d；无此车的错误档没有 d）
    grade_d_max: float | None
    grade_d_n: int  # 有定义 d 的条数
    stick_wrong: int  # 贴错：2.5 m 内最近真值是别的车
    stick_right: int  # 贴对：最近真值就是同名车
    stick_none: int  # 附近没车：2.5 m 内没有任何真值
    confusion_pairs: tuple[tuple[str, str, int], ...]  # 「发布槽→实际槽」按次数降序
    wrong_split: tuple[tuple[str, int], ...]  # 贴错三类互斥构成


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
    grade_counts: dict[str, int] = {}
    error_same_name = error_no_robot = 0
    grade_ds: list[float] = []
    stick_wrong = stick_right = stick_none = 0
    confusion: dict[tuple[str, str], int] = {}
    wrong_split: dict[str, int] = {}
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
            if pred.state != "confirmed":
                continue  # d 准线和混淆表都只数 confirmed：lost/tentative 不进 0x0305
            grade, d = grade_confirmed(gt, pred)
            grade_counts[grade] = grade_counts.get(grade, 0) + 1
            if d is None:
                error_no_robot += 1
            else:
                grade_ds.append(d)
                if grade is GRADE_ERROR:
                    error_same_name += 1
            stick, actual = stick_class(gt, pred)
            if stick is STICK_RIGHT:
                stick_right += 1
            elif stick is STICK_NONE:
                stick_none += 1
            else:
                stick_wrong += 1
                pair = (pred.slot, str(actual))
                confusion[pair] = confusion.get(pair, 0) + 1
                kind = wrong_pair_kind(pair[0], pair[1])
                wrong_split[kind] = wrong_split.get(kind, 0) + 1
    ordered = sorted(errors)
    grade_ordered = sorted(grade_ds)
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
        confirmed_graded=sum(grade_counts.values()),
        grade_counts=tuple(sorted(grade_counts.items())),
        error_same_name=error_same_name,
        error_no_robot=error_no_robot,
        grade_d_median=grade_ordered[len(grade_ordered) // 2] if grade_ordered else None,
        grade_d_max=grade_ordered[-1] if grade_ordered else None,
        grade_d_n=len(grade_ordered),
        stick_wrong=stick_wrong,
        stick_right=stick_right,
        stick_none=stick_none,
        confusion_pairs=tuple(
            (pair[0], pair[1], count)
            for pair, count in sorted(confusion.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        wrong_split=tuple(sorted(wrong_split.items())),
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
    total = report.confirmed_graded
    if total:
        bands = " ".join(
            f"{grade}={count} ({count / total * 100:.1f}%)"
            for grade, count in report.grade_counts
        )
    else:
        bands = "-"
    lines.append(f"confirmed_d_line: n={total} {bands}")
    lines.append(
        f"  error split: same_name_ge_1.6m={report.error_same_name} "
        f"no_robot={report.error_no_robot}"
    )
    if report.grade_d_median is not None:
        lines.append(
            f"  d median={report.grade_d_median:.3f}m max={report.grade_d_max:.3f}m "
            f"(defined d only, n={report.grade_d_n})"
        )
    else:
        lines.append("  d median=- max=- (no defined d)")
    lines.append(
        f"confirmed_stick: wrong={report.stick_wrong} right={report.stick_right} "
        f"none={report.stick_none}"
    )
    if report.confusion_pairs:
        pairs_txt = " ".join(
            f"{from_slot}->{to_slot}={count}"
            for from_slot, to_slot, count in report.confusion_pairs
        )
        lines.append(f"  confusion pairs (desc): {pairs_txt}")
    lines.append(f"  wrong split: {_fmt_pair(report.wrong_split)}")
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
