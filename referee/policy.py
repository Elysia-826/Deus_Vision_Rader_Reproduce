"""双倍易伤触发策略 + 离线目标检测。规则给了两次机会，什么时候按由我们定。

当前原型策略：手里有机会、对方没在双倍中、且当前易伤的敌方车 ≥ 阈值，
就触发——易伤的车越多，翻倍窗口吃到的伤害期望越大。
更聪明的（预测易伤趋势、按目标血量加权）以后在这个文件里长大。

离线目标检测（§5.1.5.4）：敌方定位模块离线时，我方发的任何坐标
都判「准确」。0x020C 只给 1 bit/槽（P≥100），P 攒分过程看不见，
所以检测只能靠发送质量反推：P≥100 需要持续的准确标记流
（连击 +1/发），如果我们最近发的多是 LOST 外推（占比过半）而
易伤位还亮着，只有「离线模块把外推也判准确」能解释——此时把
该槽加入优先集合，质量门放行 LOST 标记，白捡 P 把易伤秒钉住。
"""

from __future__ import annotations

from collections import deque
from typing import Final, Mapping

from referee.commands import MarkProgress, RadarInfo

MIN_VULNERABLE_ENEMIES: Final = 2  # 同时易伤的敌方车达到几台才值得按

# 每槽发送历史长度（帧）。5Hz 发送 × 50 帧 = 最近 10s 的质量画像。
_SEND_WINDOW: Final = 50
# CONFIRMED 发送占比低于此值 + 易伤位亮 → 判离线。
# P≥100 至少需要 ~14 发连击准确的标记；LOST 外推常态下 0.5s 后
# 漂出准确圈，撑不起这种连击。占比过半都是外推还能保持 P≥100，
# 基本只有离线判准能解释。
OFFLINE_MAX_LOST_SHARE: Final = 0.5


def should_trigger_double(radar_info: RadarInfo, mark_progress: MarkProgress) -> bool:
    """判定表：没机会 / 已在双倍中 / 易伤车不够，都不按。"""
    if radar_info.double_chances <= 0:
        return False
    if radar_info.enemy_double_active:
        return False  # 规则允许排队第二次，但原型先做不浪费的打法
    vulnerable = sum(1 for index in range(6) if mark_progress.enemy_vulnerable(index))
    return vulnerable >= MIN_VULNERABLE_ENEMIES


class SendQualityTracker:
    """每帧记下各敌方槽实际发出去的标记质量（CONFIRMED=1 / LOST=0）。

    run_radar 主循环每帧 feed；0x020C 到达时把 confirmed_ratio 交给
    detect_offline_targets 判离线。
    """

    def __init__(self) -> None:
        self._history: list[deque[int]] = [deque(maxlen=_SEND_WINDOW) for _ in range(6)]

    def feed(self, slot: int, confirmed: bool) -> None:
        """记一帧该槽的发送质量。slot 是 0x0305 敌方槽下标 0..5。"""
        if 0 <= slot < 6:
            self._history[slot].append(1 if confirmed else 0)

    def confirmed_ratio(self, slot: int) -> float | None:
        """该槽最近窗口内 CONFIRMED 发送占比。没发过任何标记返回 None。"""
        if not 0 <= slot < 6 or not self._history[slot]:
            return None
        return sum(self._history[slot]) / len(self._history[slot])


def detect_offline_targets(
    mark_progress: MarkProgress,
    quality: SendQualityTracker,
) -> frozenset[int]:
    """易伤位亮 + 我们的 CONFIRMED 发送占比撑不起 P≥100 → 判模块离线。

    返回敌方槽下标集合，传给 marks_from_publishes 的 priority_slots：
    离线槽的 LOST 标记也放行（怎么发都判准确），把 P 钉在 ≥100，
    易伤秒持续积累。自纠偏：误判时外推标记吃 ERROR，P 跌破 100、
    易伤位灭，下一轮 0x020C 自然除名。
    """
    offline: set[int] = set()
    for index in range(6):
        if not mark_progress.enemy_vulnerable(index):
            continue  # 位没亮 → 不判（也没有白捡的收益）
        ratio = quality.confirmed_ratio(index)
        if ratio is None:
            continue  # 没发过标记 → 位亮是历史 P 在衰减途中，无从判断
        if ratio < 1.0 - OFFLINE_MAX_LOST_SHARE:
            offline.add(index)
    return frozenset(offline)
