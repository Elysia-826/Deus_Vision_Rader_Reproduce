from __future__ import annotations

from referee.commands import MarkProgress, RadarInfo
from referee.policy import MIN_VULNERABLE_ENEMIES, should_trigger_double


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
