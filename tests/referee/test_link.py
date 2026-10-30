from __future__ import annotations

import pytest

from referee.commands import (
    CMD_GAME_STATUS,
    CMD_MAP_POSITIONS,
    CMD_MARK_PROGRESS,
    CMD_RADAR_INFO,
    CMD_ROBOT_HP,
    SUB_RADAR_DECISION,
    MapMark,
    encode_map_positions,
    mark_from_field,
    parse_radar_info,
)
from referee.frame import FrameParser, pack_frame
from referee.link import MIN_SEND_INTERVAL_S, RefereeLink
from referee.transport import LoopbackPair
from track.types import Team


def _red_link() -> tuple[RefereeLink, LoopbackPair]:
    pair = LoopbackPair()
    link = RefereeLink(pair.radar, own_team=Team.RED)
    return link, pair


def _peer_frames(pair: LoopbackPair) -> list[tuple[int, bytes]]:
    return [(f.cmd_id, f.payload) for f in FrameParser().feed(pair.peer.read(0.0))]


def test_send_map_writes_0x0305_frame() -> None:
    link, pair = _red_link()
    marks = (MapMark(x_cm=1400, y_cm=750),) + (None,) * 11
    assert link.send_map(marks, now_s=0.0) is True
    assert _peer_frames(pair) == [(CMD_MAP_POSITIONS, encode_map_positions(marks))]


def test_send_map_throttles_to_5hz() -> None:
    link, pair = _red_link()
    marks = (None,) * 12
    assert link.send_map(marks, now_s=0.0) is True
    assert link.send_map(marks, now_s=0.19) is False
    first = _peer_frames(pair)  # read 是破坏性的：每次取走全部
    assert first == [(CMD_MAP_POSITIONS, b"\x00" * 48)]
    assert link.send_map(marks, now_s=0.2) is True
    second = _peer_frames(pair)
    assert second == [(CMD_MAP_POSITIONS, b"\x00" * 48)]
    assert MIN_SEND_INTERVAL_S == pytest.approx(0.2)


def test_trigger_double_counts_monotonically() -> None:
    link, pair = _red_link()
    link.trigger_double()
    link.trigger_double()
    frames = _peer_frames(pair)
    assert len(frames) == 2
    for cmd_id, payload in frames:
        assert cmd_id == 0x0301
        assert payload[:2] == SUB_RADAR_DECISION.to_bytes(2, "little")
        assert payload[2:6] == b"\x09\x00\x80\x80"  # 红雷达 9 + 服务器
        assert payload[7] == 0  # 不动密钥
        assert payload[14] == 0  # 加固计数不随双倍递增
        assert len(payload) == 19
    assert [payload[6] for _, payload in frames] == [1, 2]


def test_verify_enemy_password_carries_key_and_current_counter() -> None:
    link, pair = _red_link()
    link.trigger_double()
    link.verify_enemy_password("Zz9x8y")
    frames = _peer_frames(pair)
    last = frames[-1][1]
    assert last[6] == 1  # 双倍计数保持 1，不因发密钥递增
    assert last[7] == 2  # 指令类型 = 提交对方密钥
    assert last[8:14] == b"Zz9x8y"
    assert last[14] == 0  # 加固计数不随密钥提交递增


def test_blue_link_uses_blue_radar_sender_id() -> None:
    pair = LoopbackPair()
    link = RefereeLink(pair.radar, own_team=Team.BLUE)
    link.trigger_double()
    (cmd_id, payload), *_ = _peer_frames(pair)
    assert payload[2:6] == b"\x6d\x00\x80\x80"  # 109 小端 + 0x8080


def test_poll_parses_all_downlink_commands() -> None:
    link, pair = _red_link()
    game = pack_frame(seq=0, cmd_id=CMD_GAME_STATUS, payload=b"\x41" + (120).to_bytes(2, "little") + (0).to_bytes(8, "little"))
    hp = pack_frame(seq=1, cmd_id=CMD_ROBOT_HP, payload=(100).to_bytes(2, "little") * 12)
    mask = pack_frame(seq=2, cmd_id=CMD_MARK_PROGRESS, payload=(0x0021).to_bytes(2, "little"))
    info = pack_frame(seq=3, cmd_id=CMD_RADAR_INFO, payload=bytes((0x03,)) + b"\x00" * 7)
    pair.peer.write(game + hp + mask + info)
    events = link.poll(timeout_s=0.0)
    assert len(events) == 4
    assert events[0].game_status is not None and events[0].game_status.game_progress == 4
    assert events[1].robot_hp is not None and events[1].robot_hp.heavy == 100
    assert events[2].mark_progress is not None and events[2].mark_progress.mask == 0x0021
    assert events[3].radar_info is not None and events[3].radar_info.double_chances == 3


def test_poll_ignores_unknown_and_corrupted_frames() -> None:
    link, pair = _red_link()
    unknown = pack_frame(seq=0, cmd_id=0x9999, payload=b"\x01")
    junk = b"\xa5\xff\xff\xff\xff\xff"
    bad_len = pack_frame(seq=1, cmd_id=CMD_RADAR_INFO, payload=b"\x00")  # 1B 拒收
    good = pack_frame(seq=2, cmd_id=CMD_RADAR_INFO, payload=bytes((0x05,)) + b"\x00" * 7)
    pair.peer.write(junk + unknown + bad_len + good)
    events = link.poll(timeout_s=0.0)
    assert len(events) == 1
    assert events[0].radar_info is not None
    assert events[0].radar_info == parse_radar_info(bytes((0x05,)) + b"\x00" * 7)


def test_seq_increments_and_wraps_across_sends() -> None:
    link, pair = _red_link()
    for _ in range(258):  # 2 圈 + 2：越过 255 回绕点
        link.trigger_double()
    seqs = [f.seq for f in FrameParser().feed(pair.peer.read(0.0))]
    assert seqs[:2] == [0, 1]
    assert seqs[254:258] == [254, 255, 0, 1]


def test_marks_from_field_roundtrip_through_link() -> None:
    link, pair = _red_link()
    marks = (mark_from_field(-12.0, 3.0),) + (None,) * 11
    link.send_map(marks, now_s=0.0)
    (cmd_id, payload), *_ = _peer_frames(pair)
    assert cmd_id == CMD_MAP_POSITIONS
    assert int.from_bytes(payload[0:2], "little") == 200
    assert int.from_bytes(payload[2:4], "little") == 1050
