"""双倍易伤触发策略 + 离线目标检测。规则给了两次机会，什么时候按由我们定。

当前原型策略：手里有机会、对方没在双倍中、且当前易伤的敌方车 ≥ 阈值，
就触发——易伤的车越多，翻倍窗口吃到的伤害期望越大。
更聪明的（预测易伤趋势、按目标血量加权）以后在这个文件里长大。

离线目标检测（§5.1.5.4）：敌方定位模块离线时，我方发的任何坐标
都判准确。如果某敌方槽已经易伤（P≥100）但我们最近的标记大多是
错误的（d>1.6m），说明它的定位模块可能离线——此时应该降低质量门
槛（LOST/低置信也发），白捡 P。
"""

from __future__ import annotations

from typing import Final

from referee.commands import MarkProgress, RadarInfo

MIN_VULNERABLE_ENEMIES: Final = 2  # 同时易伤的敌方车达到几台才值得按


def should_trigger_double(radar_info: RadarInfo, mark_progress: MarkProgress) -> bool:
    """判定表：没机会 / 已在双倍中 / 易伤车不够，都不按。"""
    if radar_info.double_chances <= 0:
        return False
    if radar_info.enemy_double_active:
        return False  # 规则允许排队第二次，但原型先做不浪费的打法
    vulnerable = sum(1 for index in range(6) if mark_progress.enemy_vulnerable(index))
    return vulnerable >= MIN_VULNERABLE_ENEMIES


def detect_offline_targets(mark_progress: MarkProgress, accurate_count: int) -> frozenset[int]:
    """检测可能的定位模块离线目标（§5.1.5.4）。

    启发式：敌方槽已易伤（P≥100）但我们几乎没有准确标记（accurate_count
    远低于 P 累积所需）→ 模块可能离线。返回敌方槽索引集合，加入
    marks_from_publishes 的 priority_slots 后可以白捡 P。

    accurate_count：最近 N 帧内我们对该槽发出的「准确」标记数。
    如果 P 已达 100+ 但 accurate_count < 5（连击公式到 100 需 ~14 发
    准确标记），说明有其他因素在帮 P 攒分——最可能就是模块离线。
    """
    offline: set[int] = set()
    for index in range(6):
        if not mark_progress.enemy_vulnerable(index):
            continue  # 还没易伤 → 不判断
        # P ≥ 100 需要 ≥14 发连续准确标记。如果我们只发了远少于 14 发
        # 准确标记但它还是易伤了，模块大概率离线。
        if accurate_count < 5:  # 远低于 P≥100 所需的连击数
            offline.add(index)
    return frozenset(offline)
