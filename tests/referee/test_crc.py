from __future__ import annotations

import pytest

from referee.crc import crc8, crc16


def _ref_crc8(data: bytes, init: int = 0xFF) -> int:
    """无表逐位参考实现（poly 反射形 0x8C），对照官方附录一的表驱动版本。"""
    value = init
    for byte in data:
        value ^= byte
        for _ in range(8):
            value = (value >> 1) ^ 0x8C if value & 1 else value >> 1
    return value


def _ref_crc16(data: bytes, init: int = 0xFFFF) -> int:
    value = init
    for byte in data:
        value ^= byte
        for _ in range(8):
            value = (value >> 1) ^ 0x8408 if value & 1 else value >> 1
    return value


@pytest.mark.parametrize(
    "data",
    [
        b"",
        b"\x00",
        b"\xA5\x03\x00\x00",
        b"\xA5\x30\x00\x07\x8C\x05\x03",
        bytes(range(64)),
        b"password" * 13,
    ],
)
def test_table_driven_matches_bitwise_reference(data: bytes) -> None:
    assert crc8(data) == _ref_crc8(data)
    assert crc16(data) == _ref_crc16(data)


def test_init_values_pass_through_on_empty_input() -> None:
    assert crc8(b"") == 0xFF
    assert crc16(b"") == 0xFFFF


def test_official_table_first_entries() -> None:
    """官方附录一表格首 8 项的抽样核对（防生成多项式抄错）。"""
    from referee.crc import _CRC8_TABLE, _CRC16_TABLE

    assert _CRC8_TABLE[:8] == (0x00, 0x5E, 0xBC, 0xE2, 0x61, 0x3F, 0xDD, 0x83)
    assert _CRC16_TABLE[:8] == (0x0000, 0x1189, 0x2312, 0x329B, 0x4624, 0x57AD, 0x6536, 0x74BF)
