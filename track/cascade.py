"""固定兵种槽跟踪。历史图案是软代价；漏图案或短遮挡时点还在，名字不改。"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Final, assert_never

from locate.field import FIELD_X_MAX, FIELD_X_MIN, FIELD_Y_MAX, FIELD_Y_MIN
from track.assign import REJECT, min_cost_pairs
from track.kalman import PlanarKalman
from track.types import (
    FieldObservation,
    PatternStatus,
    PublishedTrack,
    Role,
    SlotId,
    Team,
    TrackPhase,
    all_slots,
)

MATCH_GATE_M: Final = 2.5
CONFIRM_HITS: Final = 2
COAST_FRAMES: Final = 5
DEAD_FRAMES: Final = 5  # 连续 5 帧（0.5s@10Hz）装甲全 dead 才判死；单帧误闪率 28% 不可信
REBIND_FRAMES: Final = 3
BIRTH_CONF: Final = 0.5
BOT_BONUS: Final = 0.8
PATTERN_BONUS: Final = 0.2
PATTERN_PENALTY: Final = 0.35
TEAM_PENALTY: Final = 0.5
# 身份时序聚合（港科大 type.py 同思路）：每槽记最近 30 帧的灯队/兵种票，
# 取最近 10 帧指数加权（τ=0.5）出分布。小陀螺下单帧 dead/错标翻不动已确认槽。
_HIST_LEN: Final = 30
_HIST_WINDOW: Final = 10
_HIST_TAU: Final = 0.5


def _weighted_argmax(hist: deque) -> tuple[object | None, float]:
    """最近 _HIST_WINDOW 帧的指数加权众数 + 归一化得分。空历史返回 (None, 0)。"""
    recent = list(hist)[-_HIST_WINDOW:]
    if not recent:
        return None, 0.0
    n = len(recent)
    scores: dict[object, float] = {}
    total = 0.0
    for i, value in enumerate(recent):
        weight = math.exp(-_HIST_TAU * (n - 1 - i))
        scores[value] = scores.get(value, 0.0) + weight
        total += weight
    best = max(scores, key=lambda k: scores[k])
    return best, scores[best] / total


@dataclass(slots=True)  # noqa: MUTABLE_OK — 状态机和滤波器跨帧原地更新
class _Slot:
    slot: SlotId
    phase: TrackPhase = TrackPhase.INACTIVE
    bot_id: int | None = None
    hit_count: int = 0
    miss_count: int = 0
    rebind_streak: int = 0
    dead_streak: int = 0
    seen: bool = False
    last_x: float = 0.0
    last_y: float = 0.0
    filt: PlanarKalman = field(default_factory=PlanarKalman)
    team_hist: deque[Team] = field(default_factory=lambda: deque(maxlen=_HIST_LEN))
    role_hist: deque[Role] = field(default_factory=lambda: deque(maxlen=_HIST_LEN))


class CascadeMatchTracker:
    """10 个槽。不读裁判，不改定位，不做猜点跳变。"""

    def __init__(self) -> None:
        self._slots = [_Slot(slot) for slot in all_slots()]

    def step(self, observations: tuple[FieldObservation, ...], dt_s: float) -> tuple[PublishedTrack, ...]:
        """预测，再全局分配，再发布仍确认或仍在外推的槽。"""
        self._predict(dt_s)
        pairs = self._match(observations)
        matched = {slot_i for slot_i, _obs_i in pairs}
        for slot_i, obs_i in pairs:
            self._absorb(self._slots[slot_i], observations[obs_i], dt_s)
        for index, slot in enumerate(self._slots):
            if index not in matched:
                self._mark_miss(slot)
        return self._publish()

    def _predict(self, dt_s: float) -> None:
        for slot in self._slots:
            match slot.phase:
                case TrackPhase.CONFIRMED | TrackPhase.LOST | TrackPhase.TENTATIVE:
                    slot.filt.predict(dt_s)
                case TrackPhase.INACTIVE:
                    continue
                case unreachable:
                    assert_never(unreachable)

    def _match(self, observations: tuple[FieldObservation, ...]) -> tuple[tuple[int, int], ...]:
        if not observations:
            return ()
        cost = [
            [_pair_cost(slot, obs) for obs in observations]
            for slot in self._slots
        ]
        return min_cost_pairs(cost)

    def _absorb(self, slot: _Slot, obs: FieldObservation, dt_s: float) -> None:
        status = pattern_status(slot, obs)
        # 本帧观测先成为该槽的历史票，再走状态机；pattern_status 查的是投票前分布。
        if obs.team is not None:
            slot.team_hist.append(obs.team)
        if obs.role is not None:
            slot.role_hist.append(obs.role)
        # 死亡时序投票：连续 DEAD_FRAMES 帧装甲全 dead 才真死（单帧误闪 28%）。
        if obs.armor_dead:
            slot.dead_streak += 1
        else:
            slot.dead_streak = 0
        match slot.phase:
            case TrackPhase.INACTIVE:
                slot.phase = TrackPhase.TENTATIVE
                slot.hit_count = 1
                slot.miss_count = 0
                slot.bot_id = obs.bot_id
                slot.rebind_streak = 0
                slot.filt.reset(obs.x, obs.y)
                _remember(slot, obs)
            case TrackPhase.TENTATIVE:
                if status is PatternStatus.AGREE:
                    slot.hit_count += 1
                if slot.hit_count >= CONFIRM_HITS and status is PatternStatus.AGREE:
                    slot.phase = TrackPhase.CONFIRMED
                _correct(slot, obs, dt_s)
                _note_bot(slot, obs, status)
            case TrackPhase.CONFIRMED | TrackPhase.LOST:
                slot.phase = TrackPhase.CONFIRMED
                slot.miss_count = 0
                _correct(slot, obs, dt_s)
                _note_bot(slot, obs, status)
            case unreachable:
                assert_never(unreachable)
        if slot.dead_streak >= DEAD_FRAMES:
            _deactivate(slot)  # 尸体停发：连着半秒装甲全熄，判死释放槽位

    def _mark_miss(self, slot: _Slot) -> None:
        match slot.phase:
            case TrackPhase.INACTIVE:
                return
            case TrackPhase.TENTATIVE:
                slot.hit_count -= 1
                if slot.hit_count <= 0:
                    _deactivate(slot)
            case TrackPhase.CONFIRMED:
                slot.phase = TrackPhase.LOST
                slot.miss_count = 1
            case TrackPhase.LOST:
                slot.miss_count += 1
                if slot.miss_count >= COAST_FRAMES:
                    _deactivate(slot)
            case unreachable:
                assert_never(unreachable)

    def _publish(self) -> tuple[PublishedTrack, ...]:
        published: list[PublishedTrack] = []
        for slot in self._slots:
            if _visible(slot):
                published.append(
                    PublishedTrack(
                        label=slot.slot.label,
                        team=slot.slot.team,
                        x=slot.filt.x,
                        y=slot.filt.y,
                        phase=slot.phase,
                        bot_id=slot.bot_id,
                        vx=slot.filt.vx,
                        vy=slot.filt.vy,
                    )
                )
        return tuple(published)


def pattern_status(slot: _Slot, obs: FieldObservation) -> PatternStatus:
    """槽的当前身份 = 历史分布；新槽无历史时退化为单帧观测。

    小陀螺下单帧 dead/错标不会翻案已确认槽；持续的新身份会随指数加权
    慢慢成为众数，槽判定随之翻转（车确实换了）。
    """
    team, _ = _weighted_argmax(slot.team_hist)
    role, _ = _weighted_argmax(slot.role_hist)
    if team is None or role is None:
        team, role = obs.team, obs.role
    if team is None or role is None:
        return PatternStatus.ABSENT
    if team is slot.slot.team and role is slot.slot.role:
        return PatternStatus.AGREE
    return PatternStatus.CONFLICT


def _pair_cost(slot: _Slot, obs: FieldObservation) -> float:
    status = pattern_status(slot, obs)
    match slot.phase:
        case TrackPhase.INACTIVE:
            if obs.armor_dead:
                return REJECT  # 装甲全熄的车不许开新槽：先当尸体，不当新车
            if status is not PatternStatus.AGREE or obs.pattern_conf < BIRTH_CONF:
                return REJECT
            return 1.0 - obs.pattern_conf
        case TrackPhase.TENTATIVE | TrackPhase.CONFIRMED | TrackPhase.LOST:
            return _active_cost(slot, obs, status)
        case unreachable:
            assert_never(unreachable)


def _active_cost(slot: _Slot, obs: FieldObservation, status: PatternStatus) -> float:
    dist = ((slot.filt.x - obs.x) ** 2 + (slot.filt.y - obs.y) ** 2) ** 0.5
    if dist > MATCH_GATE_M:
        return REJECT
    cost = dist
    if slot.bot_id is not None and obs.bot_id is not None and slot.bot_id == obs.bot_id:
        cost -= BOT_BONUS
    match status:
        case PatternStatus.AGREE:
            cost -= PATTERN_BONUS
        case PatternStatus.CONFLICT:
            cost += PATTERN_PENALTY
        case PatternStatus.ABSENT:
            cost += 0.0
        case unreachable:
            assert_never(unreachable)
    if obs.team is not None and obs.team is not slot.slot.team:
        cost += TEAM_PENALTY
    return cost


def _note_bot(slot: _Slot, obs: FieldObservation, status: PatternStatus) -> None:
    if obs.bot_id is None or slot.bot_id == obs.bot_id:
        slot.rebind_streak = 0
        if obs.bot_id is not None:
            slot.bot_id = obs.bot_id
        return
    if status is PatternStatus.CONFLICT:
        slot.rebind_streak = 0
        return
    slot.rebind_streak += 1
    if slot.rebind_streak >= REBIND_FRAMES:
        slot.bot_id = obs.bot_id
        slot.rebind_streak = 0


def _visible(slot: _Slot) -> bool:
    """发布条件 = 状态机还在讲这台车 + 位置还在场上 + 身份投票大致认同本槽。

    身份发布门：R3 槽吃着英雄的观测、发英雄的坐标当步兵坐标——比不发
    还糟 5 倍（错发 5Hz 每包扣 x，静默 0.5s 才扣一次中断）。槽自己的
    历史票是证据：近窗顶票 ≥0.7 都在说「我是别的兵种」就静默。
    """
    match slot.phase:
        case TrackPhase.CONFIRMED:
            pass
        case TrackPhase.LOST:
            if slot.miss_count >= COAST_FRAMES:
                return False
        case TrackPhase.TENTATIVE:
            pass
        case TrackPhase.INACTIVE:
            return False
        case unreachable:
            assert_never(unreachable)
    if not _on_field_publish(slot.filt.x, slot.filt.y):
        return False
    return _identity_publishable(slot)


_PUBLISH_GATE_FRACTION: Final = 0.7  # 近窗顶票占比 ≥0.7 在说别的兵种 → 静默
_PUBLISH_GATE_MIN_VOTES: Final = 3  # 至少 3 票才够格说「认同/不认同」


def _identity_publishable(slot: _Slot) -> bool:
    """槽自己的票强烈反对本槽身份时不发布——错发比静默伤害大 5 倍。

    跟背叛判死不同：不杀槽（Kalman 保留），只屏蔽输出。车换回来时
    近窗指数加权投票会在几帧内翻回正确身份，发布自动恢复。
    """
    if len(slot.role_hist) < _PUBLISH_GATE_MIN_VOTES:
        return True
    role, fraction = _weighted_argmax(slot.role_hist)
    if role is not None and role is not slot.slot.role and fraction >= _PUBLISH_GATE_FRACTION:
        return False
    return True


_PUBLISH_MARGIN_M: Final = 0.5  # 真车最大半长；观测侧的 2m 宽容度不该带进发布


def _on_field_publish(x: float, y: float) -> bool:
    return (
        FIELD_X_MIN - _PUBLISH_MARGIN_M <= x <= FIELD_X_MAX + _PUBLISH_MARGIN_M
        and FIELD_Y_MIN - _PUBLISH_MARGIN_M <= y <= FIELD_Y_MAX + _PUBLISH_MARGIN_M
    )

def _correct(slot: _Slot, obs: FieldObservation, dt_s: float) -> None:
    vx = (obs.x - slot.last_x) / dt_s if slot.seen and dt_s > 0.0 else None
    vy = (obs.y - slot.last_y) / dt_s if slot.seen and dt_s > 0.0 else None
    slot.filt.update(obs.x, obs.y, vx, vy)
    _remember(slot, obs)


def _remember(slot: _Slot, obs: FieldObservation) -> None:
    slot.seen = True
    slot.last_x = obs.x
    slot.last_y = obs.y


def _deactivate(slot: _Slot) -> None:
    slot.phase = TrackPhase.INACTIVE
    slot.bot_id = None
    slot.hit_count = 0
    slot.miss_count = 0
    slot.rebind_streak = 0
    slot.dead_streak = 0
    slot.seen = False
    slot.filt = PlanarKalman()
    slot.team_hist.clear()
    slot.role_hist.clear()
