from __future__ import annotations

import pytest

from referee.commands import (
    BLUE_RADAR_ID,
    BLUE_SENTRY_ID,
    CMD_ROBOT_INTERACTION,
    SUB_ROBOT_TO_ROBOT,
    MapMark,
    decode_sentry_guidance,
    encode_sentry_guidance,
)
from referee.link import RefereeLink
from referee.stub import StubReferee
from referee.transport import LoopbackPair
from track.types import Team


def _marks(*positions: tuple[int, int]) -> tuple[MapMark | None, ...]:
    out: list[MapMark | None] = [None] * 12
    for i, (x, y) in enumerate(positions):
        if i < 6:
            out[i] = MapMark(x_cm=x, y_cm=y)
    return tuple(out)


def test_sentry_guidance_roundtrip() -> None:
    marks = _marks((100, 200), (300, 400))
    content = encode_sentry_guidance(marks)
    assert content[0] == 2  # 数量
    decoded = decode_sentry_guidance(content)
    assert decoded == (MapMark(x_cm=100, y_cm=200), MapMark(x_cm=300, y_cm=400))


def test_sentry_guidance_all_none() -> None:
    content = encode_sentry_guidance((None,) * 12)
    assert content == b"\x00"  # 数量 0，无坐标
    assert decode_sentry_guidance(content) == ()


def test_sentry_guidance_only_takes_enemy_slots() -> None:
    """己方槽（6-11）不进引导——只发敌方位置。"""
    marks: list[MapMark | None] = [None] * 12
    marks[0] = MapMark(x_cm=100, y_cm=200)  # 敌方
    marks[6] = MapMark(x_cm=999, y_cm=999)  # 己方——不应出现
    content = encode_sentry_guidance(tuple(marks))
    decoded = decode_sentry_guidance(content)
    assert len(decoded) == 1
    assert decoded[0].x_cm == 100
    assert decoded[0].y_cm == 200


def test_link_sends_sentry_guidance() -> None:
    from referee.frame import FrameParser

    pair = LoopbackPair()
    link = RefereeLink(pair.radar, own_team=Team.BLUE)
    link.send_sentry_guidance(_marks((500, 600)))
    frames = FrameParser().feed(pair.peer.read(0.0))
    assert len(frames) == 1
    frame = frames[0]
    assert frame.cmd_id == CMD_ROBOT_INTERACTION
    # 解包 0x0301 外壳
    payload = frame.payload
    sub_cmd = int.from_bytes(payload[0:2], "little")
    sender = int.from_bytes(payload[2:4], "little")
    receiver = int.from_bytes(payload[4:6], "little")
    assert sub_cmd == SUB_ROBOT_TO_ROBOT
    assert sender == BLUE_RADAR_ID
    assert receiver == BLUE_SENTRY_ID
    # 解包内容
    decoded = decode_sentry_guidance(payload[6:])
    assert decoded == (MapMark(x_cm=500, y_cm=600),)
