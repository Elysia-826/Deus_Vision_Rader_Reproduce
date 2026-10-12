"""官方附录一的 CRC 校验。表由多项式生成，与《RM2026 通信协议》附录一的表格逐项一致。

CRC8：poly x⁸+x⁵+x⁴+1（LSB-first 反射形 0x8C），初值 0xff，校验帧头前 4 字节。
CRC16：poly 0x8408（CRC-CCITT 反射形），初值 0xffff，整包校验（帧头+cmd_id+data）。
"""

from __future__ import annotations

from typing import Final

CRC8_INIT: Final[int] = 0xFF
_CRC8_POLY_REFLECTED: Final[int] = 0x8C

CRC16_INIT: Final[int] = 0xFFFF
_CRC16_POLY_REFLECTED: Final[int] = 0x8408


def _build_table(poly: int) -> tuple[int, ...]:
    table: list[int] = []
    for index in range(256):
        value = index
        for _ in range(8):
            value = (value >> 1) ^ poly if value & 1 else value >> 1
        table.append(value)
    return tuple(table)


_CRC8_TABLE: Final[tuple[int, ...]] = _build_table(_CRC8_POLY_REFLECTED)
_CRC16_TABLE: Final[tuple[int, ...]] = _build_table(_CRC16_POLY_REFLECTED)


def crc8(data: bytes) -> int:
    """帧头 CRC8：对 SOF+len+seq 这 4 字节算，写进帧头第 5 字节。"""
    value = CRC8_INIT
    for byte in data:
        value = _CRC8_TABLE[value ^ byte]
    return value


def crc16(data: bytes) -> int:
    """整包 CRC16：覆盖帧头 5 字节 + cmd_id 2 字节 + data。小端写进帧尾。"""
    value = CRC16_INIT
    for byte in data:
        value = (value >> 8) ^ _CRC16_TABLE[(value ^ byte) & 0xFF]
    return value
