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
# 双寿命：COAST 内的 LOST 槽还能吸收观测（正常跟踪）；过了 COAST 进
# 幽灵态——只外推发布（小地图/猜点窗口用），_pair_cost 一律 REJECT，
# 不许偷观测。直接拉长吸收寿命会把基准从 9.6s 打到 5.6s（僵尸槽在
# 0.5-0.8s 吸走路过的别人的车，位置/身份被污染），幽灵化没有这个代价。
COAST_FRAMES: Final = 5   # 吸收寿命：0.5s@10Hz
GHOST_FRAMES: Final = 12  # 发布寿命：1.2s@10Hz，猜点门（0.5s 起）的活窗
DEAD_FRAMES: Final = 5  # 连续 5 帧（0.5s@10Hz）装甲全 dead 才判死；单帧误闪率 28% 不可信
REBIND_FRAMES: Final = 3
BIRTH_CONF: Final = 0.5
BOT_BONUS: Final = 0.8
PATTERN_BONUS: Final = 0.2
PATTERN_PENALTY: Final = 0.35
TEAM_PENALTY: Final = 0.5
# 哨兵盟友提示代价（规则 §3.2.3：雷达只能收哨兵的数据）
ALLY_HINT_M: Final = 2.0     # 距哨兵报的盟友位置 <2m 视为「在盟友附近」
# 量级依据：哨兵 INS 是厘米级真值，灯条远距误读率仿真实测 71%——
# 位置先验的可信度远高于灯色。加罚必须压得住敌方槽的
# PATTERN_BONUS+距离小账（0.8 级），减罚要能在「敌我槽贴脸抢同一观测」
# 时顶掉己方槽的 TEAM_PENALTY+CONFLICT（0.85 级）。
ALLY_HINT_PENALTY: Final = 0.9  # 盟友附近的车想进敌方槽 → 加罚（更可能被误读）
ALLY_HINT_BONUS: Final = 0.45   # 盟友附近的车想进己方槽 → 减罚（可信度高）
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
    lost_s: float = 0.0  # 丢失后累计秒数；CONFIRMED 观测清零
    rebind_streak: int = 0
    dead_streak: int = 0
    seen: bool = False
    last_x: float = 0.0
    last_y: float = 0.0
    filt: PlanarKalman = field(default_factory=PlanarKalman)
    team_hist: deque[Team] = field(default_factory=lambda: deque(maxlen=_HIST_LEN))
    role_hist: deque[Role] = field(default_factory=lambda: deque(maxlen=_HIST_LEN))


class CascadeMatchTracker:
    """10 个槽。不读裁判，不改定位，不做猜点跳变。

    own_team：哨兵盟友提示的参照系。提示点报的是「己方机器人」，
    判定敌我槽要用本方阵营比，不能信观测灯条色（灯条就是被修的对象）。
    """

    def __init__(self, *, own_team: Team | None = None) -> None:
        self._slots = [_Slot(slot) for slot in all_slots()]
        self._own_team = own_team
        self._ally_hints: tuple[tuple[float, float], ...] = ()

    def set_ally_hints(self, positions: tuple[tuple[float, float], ...]) -> None:
        """哨兵回传的己方机器人位置提示（场地米制）。

        规则 §3.2.3：雷达只能接收哨兵的数据。哨兵把己方坐标发给雷达后，
        匹配代价里用这些提示区分敌我——观测在盟友提示点附近时，更可能
        是己方车：进己方槽减代价，进敌方槽加罚（盟友被灯条误读成
        敌方色的场景）。协议未发布前留空，融合层到位后直接调用此接口；
        需要 own_team 才生效，没有 own_team 的提示无从判定敌我。
        """
        self._ally_hints = positions

    def step(self, observations: tuple[FieldObservation, ...], dt_s: float) -> tuple[PublishedTrack, ...]:
        """预测，再全局分配，再发布仍确认或仍在外推的槽。"""
        self._predict(dt_s)
        pairs = self._match(observations)
        matched = {slot_i for slot_i, _obs_i in pairs}
        for slot_i, obs_i in pairs:
            self._absorb(self._slots[slot_i], observations[obs_i], dt_s)
        for index, slot in enumerate(self._slots):
            if index not in matched:
                self._mark_miss(slot, dt_s)
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
            [_pair_cost(slot, obs, self._ally_hints, self._own_team) for obs in observations]
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
                slot.lost_s = 0.0
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
                slot.lost_s = 0.0
                _correct(slot, obs, dt_s)
                _note_bot(slot, obs, status)
            case unreachable:
                assert_never(unreachable)
        if slot.dead_streak >= DEAD_FRAMES:
            _deactivate(slot)  # 尸体停发：连着半秒装甲全熄，判死释放槽位

    def _mark_miss(self, slot: _Slot, dt_s: float) -> None:
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
                slot.lost_s = dt_s
            case TrackPhase.LOST:
                slot.miss_count += 1
                slot.lost_s += dt_s
                if slot.miss_count >= GHOST_FRAMES:
                    _deactivate(slot)  # 幽灵期满：彻底注销
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
                        pos_uncertainty=slot.filt.pos_uncertainty,
                        lost_s=slot.lost_s,
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


def _pair_cost(
    slot: _Slot,
    obs: FieldObservation,
    ally_hints: tuple[tuple[float, float], ...],
    own_team: Team | None,
) -> float:
    status = pattern_status(slot, obs)
    match slot.phase:
        case TrackPhase.INACTIVE:
            if obs.armor_dead:
                return REJECT  # 装甲全熄的车不许开新槽：先当尸体，不当新车
            if status is not PatternStatus.AGREE or obs.pattern_conf < BIRTH_CONF:
                return REJECT
            return 1.0 - obs.pattern_conf
        case TrackPhase.TENTATIVE | TrackPhase.CONFIRMED | TrackPhase.LOST:
            if slot.phase is TrackPhase.LOST and slot.miss_count >= COAST_FRAMES:
                # 幽灵：只收身份完全吻合的观测（本尊回归，灯条+图案都对上）。
                # 别的车（异色/异兵种/无身份）一律 REJECT——直接拉长吸收
                # 寿命的版本里僵尸槽偷走过路车，基准 9.6s → 5.6s。
                if obs.team is not slot.slot.team or obs.role is not slot.slot.role:
                    return REJECT
            return _active_cost(slot, obs, status, ally_hints, own_team)
        case unreachable:
            assert_never(unreachable)


def _active_cost(
    slot: _Slot,
    obs: FieldObservation,
    status: PatternStatus,
    ally_hints: tuple[tuple[float, float], ...],
    own_team: Team | None,
) -> float:
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
    # 哨兵盟友提示：观测在己方提示点附近 → 真身大概率是己方车。
    # 判定基准是 slot.team vs own_team（提示的阵营语义），不是 obs.team——
    # 灯条色正是这个提示要修的误读源，不能反过来当判据：
    #   盟友附近 + 己方槽 → 位置说这是自家车 → 减罚
    #   盟友附近 + 敌方槽 → 自家车被误读成敌色想进敌方槽 → 加罚
    if ally_hints and own_team is not None:
        near_ally = any(math.hypot(obs.x - hx, obs.y - hy) < ALLY_HINT_M for hx, hy in ally_hints)
        if near_ally:
            if slot.slot.team is own_team:
                cost -= ALLY_HINT_BONUS
            else:
                cost += ALLY_HINT_PENALTY
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
        case TrackPhase.INACTIVE:
            return False
        case TrackPhase.LOST:
            if slot.miss_count >= GHOST_FRAMES:
                return False  # 幽灵期满在 _mark_miss 注销，这里兜底
        case TrackPhase.CONFIRMED | TrackPhase.TENTATIVE:
            pass
        case unreachable:
            assert_never(unreachable)
    if not _on_field_publish(slot.filt.x, slot.filt.y):
        return False
    return _identity_publishable(slot)


_PUBLISH_GATE_FRACTION: Final = 0.6  # 近窗顶票占比 ≥0.6 在说别的兵种 → 静默
# 0.7→0.6（2026-10-24 固定回放 A/B，三套序列数据见 docs/radar_operation.md §10.7）：
# 三序列虚报合计 1086→1022，漏报 +117 仍在各序列 +20% 预算内（2027 漏报
# 442→522，上限 530）。继续压（0.5）2027 漏报到 570 超预算，不采用。
# 角色票看不见队伍错认（R1 槽吃蓝英雄时图案「1」读得完全正确），所以靠这个
# 门虚报到不了 0——上限与结构原因记录在 §10.7。
_PUBLISH_GATE_MIN_VOTES: Final = 3  # 至少 3 票才够格说「认同/不认同」
# 注意：MIN_VOTES=2 数学上不可达——出生观测必须 AGREE（第一票永远同意本槽
# 身份），2 票时异见票最大占比 1.0/1.61=0.62 < 任何可用 FRACTION，A/B 实测
# 与基线逐帧一致；最早否决只能出现在第 3 票。


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
    slot.lost_s = 0.0
    slot.rebind_streak = 0
    slot.dead_streak = 0
    slot.seen = False
    slot.filt = PlanarKalman()
    slot.team_hist.clear()
    slot.role_hist.clear()
