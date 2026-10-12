"""Stub 裁判：把规则手册 5.6.6 的「被标记进度」计分公式做成回放可对打的一端。

不是真裁判仿真器——没有图传/UI/复活机制，只实现和基准有关的部分：
  1. d 分级：准确 <0.8m，半准确 <1.6m，错误 ≥1.6m（对真值算，真值由 set_truth 注入）
  2. 连击 x：准确 +1.0 / 半准确 +0.5（上次好则累进），错误 -0.8（上次坏则连扣）
  3. 数据中断：累计 0.5s 没收到该机器人的坐标 → 追加一次「错误」判定
  4. P ∈ [0,150]，敌方 P≥100 记易伤秒，多台同时易伤不叠加（工会时间取并集）
  5. 1Hz 回发 0x020C（易伤掩码）和 0x020E（双倍机会数/触发中）
金标测试：tests/referee/test_stub.py::test_rulebook_example，逐值对照规则手册的
worked example（5Hz 全准 3 秒到 P=120 → 5Hz 全错 5 连扣 → 1s 中断两次 → 复准两帧）。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Final

from referee.commands import (
    CMD_MAP_POSITIONS,
    CMD_MARK_PROGRESS,
    CMD_RADAR_INFO,
    SLOT_NAMES,
    decode_map_positions,
    field_from_mark,
    slot_index_of,
)
from referee.frame import FrameParser, pack_frame
from referee.transport import RefereeTransport
from track.types import SimRobot, Team

ACCURATE_M: Final[float] = 0.8
SEMI_ACCURATE_M: Final[float] = 1.6
INTERRUPTION_S: Final[float] = 0.5
FEEDBACK_INTERVAL_S: Final[float] = 1.0
P_MAX: Final[float] = 150.0
DOUBLE_CHANCE_S: Final[float] = 60.0  # 累计 1 分钟易伤换 1 次双倍机会
DOUBLE_ACTIVE_S: Final[float] = 30.0


class Judgement(StrEnum):
    """一帧内某槽的判定。SILENT = 本帧没发包也没触发中断。"""

    ACCURATE = "accurate"
    SEMI = "semi"
    ERROR = "error"
    INTERRUPT = "interrupt"
    SILENT = "silent"


@dataclass(slots=True)
class _SlotState:
    p: float = 0.0
    streak: float = 0.0  # 当前连击 x
    last_good: bool = False  # 上次判定是准确/半准确
    last_data_s: float | None = None  # 上次收到坐标的时刻；None=还没开赛
    ever_sent: bool = False
    truth_x: float | None = None
    truth_y: float | None = None
    vuln_s: float = 0.0
    judge: Judgement = Judgement.SILENT
    last_d: float | None = None
    counts: dict[str, int] = field(default_factory=dict)

    def bump(self, judgement: Judgement) -> None:
        self.judge = judgement
        self.counts[judgement.value] = self.counts.get(judgement.value, 0) + 1


@dataclass(frozen=True, slots=True)
class StubSummary:
    """跑完一段回放后的汇总，score_referee.py 直接打印这个。"""

    slot_names: tuple[str, ...]
    final_p: tuple[float, ...]
    vuln_s: tuple[float, ...]
    counts: tuple[dict[str, int], ...]
    union_vuln_s: float
    doubled_vuln_s: float
    double_chances_earned: int
    frames: int


class StubReferee:
    """回环对端的假服务器。truth 由外部按帧注入（run_radar 的仿真真值）。"""

    def __init__(self, transport: RefereeTransport, *, own_team: Team) -> None:
        self._transport = transport
        self._parser = FrameParser()
        self._own = own_team
        self._slots = tuple(_SlotState() for _ in range(12))
        self._last_step_s: float | None = None
        self._last_feedback_s: float | None = None
        self._union_vuln_s = 0.0
        self._doubled_vuln_s = 0.0
        self._double_used = 0
        self._double_active_until_s: float | None = None
        self._frames = 0

    # ---------- 输入 ----------

    def set_truth(self, robots: tuple[SimRobot, ...]) -> None:
        """把当帧仿真真值换算到 0x0305 槽。不在场/没有真值的槽保持 None。"""
        for slot in self._slots:
            slot.truth_x = None
            slot.truth_y = None
        for robot in robots:
            index = slot_index_of(self._own, robot.team, robot.label)
            if index is None:
                continue
            self._slots[index].truth_x = robot.x
            self._slots[index].truth_y = robot.y

    def step(self, *, now_s: float) -> None:
        """一帧：收雷达包 → 中断判定 → 易伤秒积分 → 1Hz 回发反馈。"""
        if self._last_step_s is None:
            for slot in self._slots:
                slot.last_data_s = now_s
        dt = 0.0 if self._last_step_s is None else now_s - self._last_step_s
        self._last_step_s = now_s
        self._frames += 1
        for slot in self._slots:
            slot.judge = Judgement.SILENT

        self._consume_marks(now_s)
        self._charge_interruptions(now_s)
        self._integrate_vulnerability(dt, now_s)
        self._maybe_send_feedback(now_s)

    # ---------- 输出 ----------

    def summary(self) -> StubSummary:
        union_chances = int(self._union_vuln_s // DOUBLE_CHANCE_S)
        return StubSummary(
            slot_names=SLOT_NAMES,
            final_p=tuple(slot.p for slot in self._slots),
            vuln_s=tuple(slot.vuln_s for slot in self._slots),
            counts=tuple(dict(slot.counts) for slot in self._slots),
            union_vuln_s=self._union_vuln_s,
            doubled_vuln_s=self._doubled_vuln_s,
            double_chances_earned=union_chances,
            frames=self._frames,
        )

    def p_at(self, index: int) -> float:
        """槽位当前 P。测试和基准脚本读数用。"""
        return self._slots[index].p

    def count_at(self, index: int, judgement: Judgement) -> int:
        """槽位某类判定的累计次数。"""
        return self._slots[index].counts.get(judgement.value, 0)

    def mark_progress_mask(self) -> int:
        """0x020C：敌方 P≥100 置位，己方 P≥50 置位。"""
        mask = 0
        for index, slot in enumerate(self._slots[:6]):
            if slot.p >= 100.0:
                mask |= 1 << index
        for index, slot in enumerate(self._slots[6:]):
            if slot.p >= 50.0:
                mask |= 1 << (index + 6)
        return mask

    def radar_info_byte(self) -> int:
        """0x020E：bit0-1 剩余机会（挣到的-已用的），bit2 双倍生效中。"""
        earned = int(self._union_vuln_s // DOUBLE_CHANCE_S)
        remaining = max(0, min(3, earned - self._double_used))
        active = (
            self._double_active_until_s is not None
            and self._last_step_s is not None
            and self._double_active_until_s > self._last_step_s
        )
        return remaining | (0x04 if active else 0)

    def trigger_double(self) -> None:
        """收到雷达 0x0121 触发：扣一次机会，30s 双倍生效。没机会则忽略。"""
        if self._last_step_s is None:
            return
        earned = int(self._union_vuln_s // DOUBLE_CHANCE_S)
        if earned - self._double_used <= 0:
            return
        self._double_used += 1
        self._double_active_until_s = self._last_step_s + DOUBLE_ACTIVE_S

    def log_frame(self, path: Path, frame: int, *, now_s: float) -> None:
        """逐帧 JSONL：score_referee.py 的输入。只写有真值或发过包的槽。"""
        slots: dict[str, dict[str, float | bool | str | None]] = {}
        for name, slot in zip(SLOT_NAMES, self._slots, strict=True):
            if slot.truth_x is None and not slot.ever_sent:
                continue
            slots[name] = {
                "p": round(slot.p, 2),
                "sent": slot.judge in (Judgement.ACCURATE, Judgement.SEMI, Judgement.ERROR),
                "d": None if slot.last_d is None else round(slot.last_d, 3),
                "judge": slot.judge.value,
                "vuln": slot.p >= 100.0,
            }
        record = {"frame": frame, "t": round(now_s, 3), "slots": slots}
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ---------- 内部 ----------

    def _consume_marks(self, now_s: float) -> None:
        for frame in self._parser.feed(self._transport.read(0.0)):
            if frame.cmd_id != CMD_MAP_POSITIONS:
                continue
            marks = decode_map_positions(frame.payload)
            for index, mark in enumerate(marks):
                if mark is None:
                    continue  # (0,0) = 未发送，不重置中断计时
                slot = self._slots[index]
                slot.ever_sent = True
                slot.last_data_s = now_s
                x_m, y_m = field_from_mark(mark)
                slot.last_d = self._distance(slot, x_m, y_m)
                self._apply_judgement(slot, slot.last_d)

    def _charge_interruptions(self, now_s: float) -> None:
        for slot in self._slots:
            assert slot.last_data_s is not None
            while now_s - slot.last_data_s >= INTERRUPTION_S:
                slot.last_data_s += INTERRUPTION_S
                self._set_error(slot, Judgement.INTERRUPT)
                self._commit(slot)

    def _distance(self, slot: _SlotState, x_m: float, y_m: float) -> float:
        if slot.truth_x is None or slot.truth_y is None:
            return math.inf  # 服务器视角：场上没这辆车，必然全错
        return math.hypot(x_m - slot.truth_x, y_m - slot.truth_y)

    def _apply_judgement(self, slot: _SlotState, d: float) -> None:
        if d < ACCURATE_M:
            slot.streak = slot.streak + 1.0 if slot.last_good else 1.0
            slot.last_good = True
            slot.bump(Judgement.ACCURATE)
        elif d < SEMI_ACCURATE_M:
            slot.streak = slot.streak + 0.5 if slot.last_good else 0.5
            slot.last_good = True
            slot.bump(Judgement.SEMI)
        else:
            self._set_error(slot, Judgement.ERROR)
        self._commit(slot)

    def _set_error(self, slot: _SlotState, judgement: Judgement) -> None:
        """错误/中断只改连击；P 的提交统一走 _commit，防止双扣。"""
        slot.streak = slot.streak - 0.8 if not slot.last_good else -0.8
        slot.last_good = False
        slot.bump(judgement)

    def _commit(self, slot: _SlotState) -> None:
        slot.p = max(0.0, min(P_MAX, slot.p + slot.streak))

    def _integrate_vulnerability(self, dt: float, now_s: float) -> None:
        if dt <= 0.0:
            return
        any_enemy = False
        for slot in self._slots[:6]:
            if slot.p >= 100.0:
                slot.vuln_s += dt
                any_enemy = True
        if any_enemy:
            self._union_vuln_s += dt
            if self._double_active_until_s is not None and now_s <= self._double_active_until_s:
                self._doubled_vuln_s += dt

    def _maybe_send_feedback(self, now_s: float) -> None:
        if self._last_feedback_s is not None and now_s - self._last_feedback_s < FEEDBACK_INTERVAL_S:
            return
        self._last_feedback_s = now_s
        mask = self.mark_progress_mask()
        self._transport.write(
            pack_frame(seq=0, cmd_id=CMD_MARK_PROGRESS, payload=mask.to_bytes(2, "little"))
        )
        self._transport.write(
            pack_frame(seq=1, cmd_id=CMD_RADAR_INFO, payload=bytes((self.radar_info_byte(),)))
        )
