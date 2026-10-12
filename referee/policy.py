"""双倍易伤触发策略。规则给了两次机会，什么时候按由我们定。

当前原型策略：手里有机会、对方没在双倍中、且当前易伤的敌方车 ≥ 阈值，
就触发——易伤的车越多，翻倍窗口吃到的伤害期望越大。
更聪明的（预测易伤趋势、按目标血量加权）以后在这个文件里长大。
"""

from __future__ import annotations

from typing import Final

from referee.commands import MarkProgress, RadarInfo

MIN_VULNERABLE_ENEMIES: Final[int] = 2  # 同时易伤的敌方车达到几台才值得按


def should_trigger_double(radar_info: RadarInfo, mark_progress: MarkProgress) -> bool:
    """判定表：没机会 / 已在双倍中 / 易伤车不够，都不按。"""
    if radar_info.double_chances <= 0:
        return False
    if radar_info.enemy_double_active:
        return False  # 规则允许排队第二次，但原型先做不浪费的打法
    vulnerable = sum(1 for index in range(6) if mark_progress.enemy_vulnerable(index))
    return vulnerable >= MIN_VULNERABLE_ENEMIES
