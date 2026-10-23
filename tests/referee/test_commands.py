from __future__ import annotations

import pytest

from referee.commands import (
    CMD_MAP_POSITIONS,
    MapMark,
    PasswordAction,
    encode_map_positions,
    encode_radar_decision_content,
    encode_robot_interaction,
    mark_from_field,
    marks_from_publishes,
    parse_game_status,
    parse_mark_progress,
    parse_radar_info,
    parse_robot_hp,
)
from referee.errors import CommandDecodeError, FrameFormatError, PasswordFormatError
from track.types import PublishedTrack, Team, TrackPhase


def _publish(label: str, team: Team, x: float, y: float, phase: TrackPhase) -> PublishedTrack:
    return PublishedTrack(label=label, team=team, x=x, y=y, phase=phase, bot_id=1)


# ---------- 0x0305 ----------


def test_encode_map_positions_all_none_is_48_zero_bytes() -> None:
    assert encode_map_positions((None,) * 12) == b"\x00" * 48


def test_encode_map_positions_known_values_little_endian() -> None:
    payload = encode_map_positions((MapMark(x_cm=100, y_cm=200),) + (None,) * 11)
    assert payload[:4] == b"\x64\x00\xc8\x00"
    assert len(payload) == 48


def test_encode_map_positions_rejects_wrong_slot_count() -> None:
    with pytest.raises(FrameFormatError):
        encode_map_positions((None,) * 11)


def test_encode_map_positions_rejects_out_of_range() -> None:
    with pytest.raises(FrameFormatError):
        encode_map_positions((MapMark(x_cm=0x10000, y_cm=0),) + (None,) * 11)


# ---------- 坐标变换 ----------


def test_mark_from_field_center_maps_to_field_center() -> None:
    mark = mark_from_field(0.0, 0.0)
    assert mark == MapMark(x_cm=1400, y_cm=750)


def test_mark_from_field_clamps_outside_field() -> None:
    far = mark_from_field(99.0, -99.0)
    assert far == MapMark(x_cm=2800, y_cm=0)
    near = mark_from_field(-99.0, 99.0)
    assert near == MapMark(x_cm=0, y_cm=1500)


def test_mark_from_field_y_orientation_constant_flips_y(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import referee.commands as commands

    monkeypatch.setattr(commands, "_OUR_PLUS_Y_IS_REFEREE_PLUS_Y", False)
    flipped = commands.mark_from_field(0.0, 7.5)
    assert flipped.y_cm == 0
    monkeypatch.setattr(commands, "_OUR_PLUS_Y_IS_REFEREE_PLUS_Y", True)
    straight = commands.mark_from_field(0.0, 7.5)
    assert straight.y_cm == 1500


# ---------- 槽位映射（质量门） ----------


def test_marks_from_publishes_maps_enemy_and_ally_slots() -> None:
    publishes = (
        _publish("B3", Team.BLUE, -1.0, 2.0, TrackPhase.CONFIRMED),
        _publish("R1", Team.RED, 1.0, -2.0, TrackPhase.CONFIRMED),
    )
    marks = marks_from_publishes(Team.RED, publishes)
    enemy_inf3 = marks[2]
    assert enemy_inf3 is not None
    assert enemy_inf3 == mark_from_field(-1.0, 2.0)
    own_hero = marks[6]
    assert own_hero is not None
    assert own_hero == mark_from_field(1.0, -2.0)
    assert all(m is None for i, m in enumerate(marks) if i not in (2, 6))


def test_marks_from_publishes_sentry_maps_to_index_five() -> None:
    marks = marks_from_publishes(Team.RED, (_publish("BS", Team.BLUE, 3.0, 0.0, TrackPhase.CONFIRMED),))
    assert marks[5] == mark_from_field(3.0, 0.0)


def test_marks_from_publishes_filters_lost_by_default() -> None:
    lost = _publish("B1", Team.BLUE, 0.0, 0.0, TrackPhase.LOST)
    assert marks_from_publishes(Team.RED, (lost,))[0] is None
    assert marks_from_publishes(Team.RED, (lost,), include_lost=True)[0] is not None


def test_marks_from_publishes_priority_slot_bypasses_lost_gate() -> None:
    """§5.1.5.4 离线目标：敌方定位模块离线时任何坐标都判准确——
    priority_slots 里的槽放行 LOST 标记，白捡 P。不在集合里的照旧拦。"""
    lost_hero = _publish("B1", Team.BLUE, 0.0, 0.0, TrackPhase.LOST)
    lost_inf = _publish("B3", Team.BLUE, 1.0, 1.0, TrackPhase.LOST)
    marks = marks_from_publishes(
        Team.RED,
        (lost_hero, lost_inf),
        priority_slots=frozenset({2}),  # 敌方步3 槽判离线
    )
    assert marks[0] is None  # 不在优先集合 → LOST 照旧拦
    assert marks[2] is not None  # 优先槽 → LOST 放行
    assert marks[2] == mark_from_field(1.0, 1.0)


def test_marks_from_publishes_priority_still_blocks_tentative() -> None:
    """优先槽也不放行 TENTATIVE：两帧确认前的身份可能是错的，
    发错槽的标记对一个根本没被判定的机器人毫无意义。"""
    tentative = _publish("B1", Team.BLUE, 0.0, 0.0, TrackPhase.TENTATIVE)
    marks = marks_from_publishes(
        Team.RED, (tentative,), include_lost=True, priority_slots=frozenset({0})
    )
    assert marks[0] is None


def test_marks_from_publishes_never_sends_tentative() -> None:
    tentative = _publish("B1", Team.BLUE, 0.0, 0.0, TrackPhase.TENTATIVE)
    assert marks_from_publishes(Team.RED, (tentative,), include_lost=True)[0] is None


# ---------- 0x0121 / 0x0301 ----------


def test_radar_decision_content_layout() -> None:
    content = encode_radar_decision_content(2, PasswordAction.VERIFY_ENEMY, "ab12CD")
    assert content == b"\x02" + b"\x02" + b"ab12CD"
    assert len(content) == 8


def test_radar_decision_rejects_bad_passwords() -> None:
    with pytest.raises(PasswordFormatError):
        encode_radar_decision_content(0, PasswordAction.UPDATE_OWN, "short")
    with pytest.raises(PasswordFormatError):
        encode_radar_decision_content(0, PasswordAction.UPDATE_OWN, "12 456")
    with pytest.raises(PasswordFormatError):
        encode_radar_decision_content(0, PasswordAction.UPDATE_OWN, "密码密码六")


def test_robot_interaction_header_order() -> None:
    payload = encode_robot_interaction(0x0121, 9, 0x8080, b"\x00" * 8)
    assert payload[:6] == b"\x21\x01" + b"\x09\x00" + b"\x80\x80"
    assert len(payload) == 14


def test_robot_interaction_rejects_oversized_content() -> None:
    with pytest.raises(FrameFormatError):
        encode_robot_interaction(0x0121, 9, 0x8080, b"\x00" * 113)


# ---------- 下行解析 ----------


def _game_payload() -> bytes:
    return b"\x41" + (300).to_bytes(2, "little") + (1770000000).to_bytes(8, "little")


def test_parse_game_status() -> None:
    status = parse_game_status(_game_payload())
    assert status.game_type == 1
    assert status.game_progress == 4  # 比赛中
    assert status.stage_remain_time_s == 300
    assert status.unix_time == 1770000000


def test_parse_robot_hp_skips_reserved() -> None:
    raw = b"".join(value.to_bytes(2, "little") for value in (400, 300, 200, 100, 0, 60, 5000, 3000))
    hp = parse_robot_hp(raw)
    assert (hp.hero, hp.engineer, hp.infantry3, hp.infantry4) == (400, 300, 200, 100)
    assert (hp.sentry, hp.outpost, hp.base) == (60, 5000, 3000)


def test_parse_mark_progress_bits() -> None:
    progress = parse_mark_progress((0b0000_0000_0100_0001).to_bytes(2, "little"))
    assert progress.enemy_vulnerable(0) is True  # 敌英雄
    assert progress.enemy_vulnerable(2) is False  # 敌步3
    assert progress.ally_marked(0) is True  # 己英雄（bit6）
    assert progress.ally_marked(1) is False


def test_parse_radar_info_bitfields() -> None:
    raw = 0b0010_1110  # 机会 2 | 对方触发中 | 加密等级 1 | 可改密钥
    info = parse_radar_info(bytes((raw,)))
    assert info.double_chances == 2
    assert info.enemy_double_active is True
    assert info.own_encryption_level == 1
    assert info.can_change_password is True


def test_parsers_reject_wrong_lengths() -> None:
    with pytest.raises(CommandDecodeError):
        parse_game_status(b"\x00" * 10)
    with pytest.raises(CommandDecodeError):
        parse_robot_hp(b"\x00" * 15)
    with pytest.raises(CommandDecodeError):
        parse_mark_progress(b"\x00" * 3)
    with pytest.raises(CommandDecodeError):
        parse_radar_info(b"\x00\x00")


def test_cmd_ids_match_protocol() -> None:
    assert CMD_MAP_POSITIONS == 0x0305
