from __future__ import annotations

from referee.commands import MarkProgress, RadarInfo
from referee.policy import (
    MIN_VULNERABLE_ENEMIES,
    SendQualityTracker,
    detect_offline_targets,
    should_trigger_double,
)


def _info(chances: int = 1, active: bool = False) -> RadarInfo:
    return RadarInfo(
        double_chances=chances,
        enemy_double_active=active,
        own_encryption_level=1,
        can_change_password=True,
    )


def _progress(*vulnerable_indices: int) -> MarkProgress:
    mask = 0
    for index in vulnerable_indices:
        mask |= 1 << index
    return MarkProgress(mask=mask)


def test_no_chances_no_trigger() -> None:
    assert should_trigger_double(_info(chances=0), _progress(0, 1, 2)) is False


def test_active_double_waits() -> None:
    assert should_trigger_double(_info(chances=1, active=True), _progress(0, 1, 2)) is False


def test_single_vulnerable_enemy_is_not_enough() -> None:
    assert should_trigger_double(_info(), _progress(3)) is False


def test_threshold_triggers() -> None:
    mask = _progress(*range(MIN_VULNERABLE_ENEMIES))
    assert should_trigger_double(_info(), mask) is True


def test_ally_marks_do_not_count() -> None:
    """己方特殊标识（bit6+）不算敌方易伤。"""
    assert should_trigger_double(_info(), MarkProgress(mask=0b0000_1111_0000_0000)) is False


# ---------- 离线目标检测（§5.1.5.4） ----------


def _quality_with(slot: int, confirmed: int, lost: int) -> SendQualityTracker:
    quality = SendQualityTracker()
    for _ in range(confirmed):
        quality.feed(slot, confirmed=True)
    for _ in range(lost):
        quality.feed(slot, confirmed=False)
    return quality


def test_offline_detect_flags_lost_heavy_vulnerable_slot() -> None:
    """易伤位亮 + 我们的发布大半是 LOST 外推 → 只有离线判准能解释。"""
    quality = _quality_with(slot=0, confirmed=2, lost=18)  # CONFIRMED 占比 10%
    assert detect_offline_targets(_progress(0), quality) == frozenset({0})


def test_offline_detect_spares_confirmed_heavy_slot() -> None:
    """易伤位亮但发布基本是 CONFIRMED → 正常准确流攒出的 P，不判离线。"""
    quality = _quality_with(slot=0, confirmed=18, lost=2)  # 占比 90%
    assert detect_offline_targets(_progress(0), quality) == frozenset()


def test_offline_detect_ignores_not_vulnerable_slots() -> None:
    """位没亮的槽不判：没有白捡收益，也没有可解释的信号。"""
    quality = _quality_with(slot=1, confirmed=0, lost=20)
    assert detect_offline_targets(_progress(0), quality) == frozenset()


def test_offline_detect_no_send_history_means_unknown() -> None:
    """一个标记都没发过 → 位亮是历史 P 在衰减途中，无从判断。"""
    assert detect_offline_targets(_progress(3), SendQualityTracker()) == frozenset()


def test_offline_detect_is_per_slot_not_global() -> None:
    """每个槽独立判：0 号槽外推占大头、1 号槽正常 → 只旗 0。"""
    quality = _quality_with(slot=0, confirmed=1, lost=9)
    for _ in range(10):
        quality.feed(1, confirmed=True)
    detected = detect_offline_targets(_progress(0, 1), quality)
    assert detected == frozenset({0})


def test_send_quality_ratio_is_per_slot_windowed() -> None:
    """占比按槽分别算；窗口内没有记录的槽返回 None。"""
    quality = _quality_with(slot=2, confirmed=3, lost=7)
    assert quality.confirmed_ratio(2) == 0.3
    assert quality.confirmed_ratio(0) is None
