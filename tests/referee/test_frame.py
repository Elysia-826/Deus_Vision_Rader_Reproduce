from __future__ import annotations

import pytest

from referee.errors import FrameFormatError
from referee.frame import Frame, FrameParser, pack_frame


def test_roundtrip_single_frame() -> None:
    raw = pack_frame(seq=3, cmd_id=0x0305, payload=b"\x01\x02\x03")
    (frame,) = FrameParser().feed(raw)
    assert frame == Frame(seq=3, cmd_id=0x0305, payload=b"\x01\x02\x03")


def test_multiple_frames_in_one_chunk() -> None:
    parser = FrameParser()
    raw = pack_frame(seq=0, cmd_id=0x0001, payload=b"a") + pack_frame(
        seq=1, cmd_id=0x0003, payload=b"bb"
    )
    frames = parser.feed(raw)
    assert [f.seq for f in frames] == [0, 1]
    assert [f.cmd_id for f in frames] == [0x0001, 0x0003]


def test_byte_by_byte_feeding() -> None:
    parser = FrameParser()
    raw = pack_frame(seq=9, cmd_id=0x020E, payload=b"\x07")
    collected: list[Frame] = []
    for byte in raw:
        collected += parser.feed(bytes((byte,)))
    assert collected == [Frame(seq=9, cmd_id=0x020E, payload=b"\x07")]


def test_garbage_prefix_is_skipped() -> None:
    parser = FrameParser()
    raw = b"\xde\xad\xbe\xef" + pack_frame(seq=1, cmd_id=0x0002, payload=b"")
    (frame,) = parser.feed(raw)
    assert frame.seq == 1
    assert frame.payload == b""


def test_corrupted_crc8_drops_only_that_frame() -> None:
    good = pack_frame(seq=0, cmd_id=0x0001, payload=b"x")
    bad = bytearray(good)
    bad[3] ^= 0xFF  # seq 坏 → CRC8 不匹配
    parser = FrameParser()
    frames = parser.feed(bytes(bad) + good)
    assert frames == (Frame(seq=0, cmd_id=0x0001, payload=b"x"),)


def test_corrupted_crc16_resynchronizes() -> None:
    good = pack_frame(seq=2, cmd_id=0x0003, payload=b"yz")
    bad = bytearray(good)
    bad[-1] ^= 0x55  # 尾 CRC16 坏
    parser = FrameParser()
    frames = parser.feed(bytes(bad) + good)
    assert frames == (Frame(seq=2, cmd_id=0x0003, payload=b"yz"),)


def test_oversize_length_field_is_rejected_and_resyncs() -> None:
    good = pack_frame(seq=0, cmd_id=0x0001, payload=b"k")
    bad = bytearray(good)
    bad[1:3] = (0xFF, 0xFF)  # data_length = 65535 超上限
    parser = FrameParser()
    frames = parser.feed(bytes(bad) + good)
    assert frames == (Frame(seq=0, cmd_id=0x0001, payload=b"k"),)


def test_pure_garbage_clears_buffer() -> None:
    parser = FrameParser()
    assert parser.feed(b"\x00\x01\x02\x03") == ()
    assert parser.feed(b"\xff" * 100) == ()


def test_seq_and_cmd_bounds_are_validated() -> None:
    with pytest.raises(FrameFormatError):
        pack_frame(seq=256, cmd_id=0x0001, payload=b"")
    with pytest.raises(FrameFormatError):
        pack_frame(seq=0, cmd_id=0x10000, payload=b"")
    with pytest.raises(FrameFormatError):
        pack_frame(seq=0, cmd_id=0x0001, payload=b"\x00" * 113)
